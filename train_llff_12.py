#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
import torchvision
from os import makedirs
from random import randint
from utils.graphics_utils import fov2focal
from utils.loss_utils import l1_loss, loss_depth_smoothness, patch_norm_mse_loss, patch_norm_mse_loss_global, ssim
# from utils.loss_utils import mssim as ssim
from gaussian_renderer import render, render_for_depth, render_for_opa  # , network_gui
import sys
from scene import RenderScene, Scene, GaussianModel
from scene.cameras import MiniCam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    print('Launch TensorBoard')
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# [SDS] helper using diffusers (local weights preferred, download if missing)
class SDSHelper:
    def __init__(self, model_path, device="cuda", prompt="a photo of a green fern plant"):
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            if not model_path:
                raise ValueError("sds_model_path must be provided when sds_enable=True")
            local_only = os.path.isdir(model_path)
            self.device = device
            cache_dir = os.getenv("HF_HOME", None)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                local_files_only=local_only,
                cache_dir=cache_dir,
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                local_files_only=local_only,
                cache_dir=cache_dir,
            ).to(device)
            self.vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                local_files_only=local_only,
                cache_dir=cache_dir,
            ).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="unet",
                local_files_only=local_only,
                cache_dir=cache_dir,
            ).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(
                model_path,
                subfolder="scheduler",
                local_files_only=local_only,
                cache_dir=cache_dir,
            )
            self.text_encoder.eval()
            self.vae.eval()
            self.unet.eval()
            for mod in (self.text_encoder, self.vae, self.unet):
                for param in mod.parameters():
                    param.requires_grad_(False)
            self.empty_prompt_emb = self._encode_prompt([""])
            self.text_z = self._encode_prompt([prompt])
            print(f"[SDS] model source = {'local' if local_only else 'huggingface-download'} : {model_path}")
        except Exception as exc:
            print(f"[SDS] load failed ({exc}), disable SDS and continue training")
            self.disabled = True
            return

    def _encode_prompt(self, prompts):
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            return self.text_encoder(inputs.input_ids.to(self.device))[0]

    def sds_loss(self, image, sds_resolution=512, guidance_scale=100.0):
        img = image.unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img, (sds_resolution, sds_resolution), mode="bilinear", align_corners=False
        )
        img = img.clamp(0.0, 1.0)
        img = img * 2.0 - 1.0
        with torch.cuda.amp.autocast(enabled=True):
            latents = self.vae.encode(img).latent_dist.sample()
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = latents * scale
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=latents.device).long()
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latents_in = torch.cat([latents_noisy, latents_noisy], dim=0)
        prompt_emb = torch.cat([self.empty_prompt_emb, self.text_z], dim=0)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            noise_pred = self.unet(latents_in, t, encoder_hidden_states=prompt_emb).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha = self.scheduler.alphas_cumprod[t].to(latents.device)
        while alpha.ndim < latents.ndim:
            alpha = alpha.view(-1, 1, 1, 1)
        weight = (1.0 - alpha)
        grad = weight * (noise_pred - noise)
        return (grad.detach() * latents).mean()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, near_range):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    scene_sprical = RenderScene(dataset, gaussians, spiral=True)
    gaussians.training_setup(opt)
    sds_helper = None
    if getattr(opt, "sds_enable", False):
        sds_helper = SDSHelper(
            getattr(opt, "sds_model_path", ""),
            device="cuda",
            prompt=getattr(opt, "sds_prompt", "a photo of a green fern plant"),
        )
        if getattr(sds_helper, "disabled", False):
            sds_helper = None
    if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)
        (model_params, _) = torch.load(checkpoint)
        gaussians.load_shape(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_sprical_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17) # LLFF

    time_accum = 0

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if not viewpoint_sprical_stack:
            viewpoint_sprical_stack = scene_sprical.getRenderCameras().copy()


        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # -------------------------------------------------- DEPTH --------------------------------------------
        if iteration > opt.hard_depth_start:
            render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, background)
            depth = render_pkg["depth"]

            # Depth loss
            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 0.1 * loss_l2_dpt

            
            if iteration > 3000:
                loss_hard += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 1 * loss_global

            loss_hard.backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        # -------------------------------------------------- pnt --------------------------------------------
        if iteration > opt.soft_depth_start :
            render_pkg = render_for_opa(viewpoint_cam, gaussians, pipe, background)
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            # Depth loss
            loss_pnt = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.1 * loss_l2_dpt

            if iteration > 3000:
                loss_pnt += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 1 * loss_global

            loss_pnt.backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        

        # ---------------------------------------------- Photometric --------------------------------------------
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg['alpha']  # [visibility_filter]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        # Reg
        loss_reg = torch.tensor(0., device=loss.device)
        shape_pena = (gaussians.get_scaling.max(dim=1).values / gaussians.get_scaling.min(dim=1).values).mean()
        scale_pena = ((gaussians.get_scaling.max(dim=1, keepdim=True).values)**2).mean()
        opa_pena = 1 - (opacity[opacity > 0.2]**2).mean() + ((1 - opacity[opacity < 0.2])**2).mean()

        loss_reg += opt.shape_pena*shape_pena + opt.scale_pena*scale_pena + opt.opa_pena*opa_pena
        loss += loss_reg
        # [SDS] extra prior loss with warmup
        interval = max(getattr(opt, "sds_interval", 1), 1)
        if sds_helper is not None and iteration > getattr(opt, "sds_start_iter", 7000) and (iteration % interval == 0):
            warmup = max(getattr(opt, "sds_warmup", 1000), 1)
            w_sds = getattr(opt, "sds_weight", 0.0) * min((iteration - opt.sds_start_iter) / float(warmup), 1.0)
            view_inv = torch.inverse(viewpoint_cam.world_view_transform)
            perturb_mag = 0.01 * scene.cameras_extent
            delta = (torch.rand(3, device=view_inv.device) - 0.5) * 2.0 * perturb_mag
            view_inv_pert = view_inv.clone()
            view_inv_pert[3, :3] += delta
            world_view_pert = torch.inverse(view_inv_pert)
            full_proj_pert = (world_view_pert.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
            sds_cam = MiniCam(
                viewpoint_cam.image_width,
                viewpoint_cam.image_height,
                viewpoint_cam.FoVy,
                viewpoint_cam.FoVx,
                viewpoint_cam.znear,
                viewpoint_cam.zfar,
                world_view_pert,
                full_proj_pert,
            )
            sds_render = render(sds_cam, gaussians, pipe, background)
            sds_image = sds_render["render"]
            loss_sds = sds_helper.sds_loss(sds_image, sds_resolution=getattr(opt, "sds_resolution", 512))
            if iteration % 200 == 0:
                print(f"[SDS] iter={iteration} loss_sds={loss_sds.item():.6f} w_sds={w_sds:.6f} interval={interval}")
            loss += (w_sds * interval) * loss_sds

        loss.backward()
        
        # ================================================================================

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            time_accum += iter_start.elapsed_time(iter_end)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, render(viewpoint_cam, gaussians, pipe, background)["color"])

            # Densification
            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  
                    size_threshold = max_dist = None
                            
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, opt.split_opacity_thresh, max_dist)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            if (iteration - 1) % 25 == 0:
                viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                mask_near = None
                if iteration > 2000:
                    for idx, view in enumerate(scene_sprical.getRenderCameras().copy()):
                        mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near_range
                        mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
                    gaussians.prune_points(mask_near.squeeze())


                ## render process
                # if (iteration + 25) > (opt.iterations):
                #     while viewpoint_sprical_stack:
                #         render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=False)
                #         iteration += 1
                #         viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                # render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=((iteration + 25) > (opt.iterations)))


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_latest.pth")
            

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), 'w') as opt_log_f:
        opt_log_f.write(str(Namespace(**vars(opt))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, depth_loss=torch.tensor(0), reg_loss=torch.tensor(0)):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss_patches/depth_kl_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/reg_loss', reg_loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'eval', 'cameras' : scene.getEvalCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    alpha = render_results["alpha"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
                    bg_mask_clone = bg_mask.clone()
                    for i in range(1, 50):
                        bg_mask[:, i:] *= bg_mask_clone[:, :-i]
                    white_mask = (gt_image.min(0, keepdim=True).values > 240/255)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/mask".format(viewpoint.image_name), bg_mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/white_mask".format(viewpoint.image_name), white_mask[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



def render_one_step(iteration, time, dataset, viewpoint, gaussians, renderFunc, renderArgs, save=False):
    torch.cuda.empty_cache()
    time_path = os.path.join(dataset.model_path, 'time')
    makedirs(time_path, exist_ok=True)
    render_results = renderFunc(viewpoint, gaussians, *renderArgs)
    image = torch.clamp(render_results["render"], 0.0, 1.0)
    torchvision.utils.save_image(image, os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))
    
    import matplotlib.font_manager as fm # to create font
    from PIL import Image, ImageDraw, ImageFont  
    
    img = Image.open(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))  
    draw = ImageDraw.Draw(img)  
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 30)  
    text = format(time, '.2f') + ' s'
    x = 10
    y = 0 
    draw.text((x-1, y), text, font=font, fill='black')  
    draw.text((x+1, y), text, font=font, fill='black')  
    draw.text((x, y-1), text, font=font, fill='black')  
    draw.text((x, y+1), text, font=font, fill='black')  
    draw.text((x-1, y-1), text, font=font, fill='black')  
    draw.text((x+1, y-1), text, font=font, fill='black')  
    draw.text((x-1, y+1), text, font=font, fill='black')  
    draw.text((x+1, y+1), text, font=font, fill='black')  
    draw.text((x, y), text, font=font, fill='white')

    img.save(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))


    torch.cuda.empty_cache()
    if save:
        # os.system(f"ffmpeg -i " + time_path + f"/%5d.png -q 2 " + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))
        os.system(f'ffmpeg -f image2 -pattern_type glob -i "' + time_path + f'/*.png" -q 2 ' + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--near", type=int, default=0)
    parser.add_argument("--sds_enable", action="store_true", default=False)
    parser.add_argument("--sds_model_path", type=str, default="")
    parser.add_argument("--sds_start_iter", type=int, default=7000)
    parser.add_argument("--sds_warmup", type=int, default=1000)
    parser.add_argument("--sds_weight", type=float, default=0.001)
    parser.add_argument("--sds_prompt", type=str, default="a photo of a green fern plant")
    parser.add_argument("--sds_resolution", type=int, default=512)
    parser.add_argument("--sds_interval", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    opt = op.extract(args)
    # [SDS] propagate SDS params into optimization args
    opt.sds_enable = getattr(args, "sds_enable", False)
    opt.sds_model_path = getattr(args, "sds_model_path", "")
    opt.sds_start_iter = getattr(args, "sds_start_iter", 7000)
    opt.sds_warmup = getattr(args, "sds_warmup", 1000)
    opt.sds_weight = getattr(args, "sds_weight", 0.001)
    opt.sds_prompt = getattr(args, "sds_prompt", "a photo of a green fern plant")
    opt.sds_resolution = getattr(args, "sds_resolution", 512)
    opt.sds_interval = getattr(args, "sds_interval", 1)
    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.near)

    # All done
    print("\nTraining complete.")
