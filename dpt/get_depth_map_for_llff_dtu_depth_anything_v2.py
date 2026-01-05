import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def iter_llff_scenes(root_path):
    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    for scene in scenes:
        yield os.path.join(root_path, scene)


def collect_images(scene_path):
    patterns = [
        os.path.join(scene_path, "images", "*.JPG"),
        os.path.join(scene_path, "images", "*.jpg"),
        os.path.join(scene_path, "images", "*.png"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def preprocess_image(image, image_processor):
    try:
        inputs = image_processor(images=image, return_tensors="pt", do_resize=False)
        return inputs["pixel_values"]
    except Exception:
        mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - mean) / std
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_t


def run_depth_anything_v2(root_path, model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    for scene_path in iter_llff_scenes(root_path):
        image_paths = collect_images(scene_path)
        if not image_paths:
            print("No images found in", scene_path)
            continue

        output_dir = os.path.join(scene_path, "depths_npy")
        ensure_dir(output_dir)

        print("Processing scene:", scene_path, "images:", len(image_paths))
        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            pixel_values = preprocess_image(image, image_processor).to(device)

            with torch.no_grad():
                outputs = model(pixel_values)
                pred = outputs.predicted_depth
                pred = F.interpolate(pred.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False)
                pred = pred.squeeze(1).squeeze(0)

            depth = pred.cpu().numpy().astype(np.float32)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(output_dir, base_name + ".npy")
            np.save(out_path, depth)
            if idx % 50 == 0:
                print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", type=str, default="LLFF")
    parser.add_argument("-r", "--root_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Small")
    args = parser.parse_args()

    if args.benchmark != "LLFF":
        raise ValueError("This script is intended for LLFF. Use --benchmark LLFF.")

    root_path = args.root_path
    if not root_path.endswith("/"):
        root_path = root_path + "/"

    run_depth_anything_v2(root_path, args.model_id)


if __name__ == "__main__":
    main()
