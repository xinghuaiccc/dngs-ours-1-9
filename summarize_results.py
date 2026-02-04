import os
import json

OUTPUT_ROOT = "output-new-2-new/dtu"
SCENES = [
    "scan34", "scan41", "scan45", "scan82", "scan114", "scan31", "scan8"
]

def summarize_results():
    """
    Summarizes the evaluation results from all scenes into a single JSON file.
    """
    print("📝 Summarizing results...")
    all_results = {}
    for scene in SCENES:
        scene_output_path = os.path.join(OUTPUT_ROOT, scene)
        result_file = os.path.join(scene_output_path, "results_eval_mask.json")
        if os.path.exists(result_file):
            print(f"Found result file for scene: {scene}")
            with open(result_file, "r") as f:
                all_results[scene] = json.load(f)
        else:
            print(f"⚠️  Result file not found for scene: {scene}, skipping.")
    
    summary_file = os.path.join(OUTPUT_ROOT, "all_scenes_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"✅ Summary saved to {summary_file}")
    print("\nSummary content:")
    with open(summary_file, "r") as f:
        print(f.read())

if __name__ == "__main__":
    summarize_results()
