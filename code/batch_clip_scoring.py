import torch
from PIL import Image
import requests
import json
import time
import os
from tqdm import tqdm

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = load_from_name(
        "ViT-H-14", device=device, download_root="./", use_modelscope=True
    )
    model.eval()
    return model, preprocess, device


def download_image_with_retry(url, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download image from {url}: {e}")
                return None
            time.sleep(1)  


def calculate_clip_score(model, preprocess, device, image_url, text):
    if not text or not text.strip():
        return 0.0

    image = download_image_with_retry(image_url)
    if image is None:
        return 0.0

    try:
        image_input = preprocess(image).unsqueeze(0).to(device)

        clean_text = text.strip()
        
        text_input = clip.tokenize([clean_text]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            cosine_score = (image_features @ text_features.T).item()
            
            
            return round(float(cosine_score), 4)

    except Exception as e:
        print(f"Error calculating CLIP score for {image_url}: {e}")
        return 0.0


def process_json_file(json_path, model, preprocess, device):
    print(f"Processing {json_path}...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = 0
    processed_items = 0
    clip_scores = []

    for poem_title, poem_data in data.items():
        if "frame_content" not in poem_data:
            continue

        frame_content = poem_data["frame_content"]
        if not frame_content:
            continue

        print(f"Processing poem: {poem_title} ({len(frame_content)} frames)")

        for i, frame in enumerate(tqdm(frame_content, desc=f"Processing {poem_title}")):
            total_items += 1

            img_path = frame.get("img_path", "")
            ext_content = frame.get("ext_content", "")
            
            if ext_content and ext_content.strip():
                text_for_clip = "唐朝古诗意象：" + ext_content
            else:
                content = frame.get("content", "")
                if content and content.strip():
                    text_for_clip = content
                else:
                    frame["clip_score"] = 0.0
                    continue
            
            if not img_path:
                frame["clip_score"] = 0.0
                continue

            clip_score = calculate_clip_score(
                model, preprocess, device, img_path, text_for_clip
            )
            frame["clip_score"] = clip_score

            if clip_score > 0:
                clip_scores.append(clip_score)

            processed_items += 1

            time.sleep(0.1)

    print(f"Processed {processed_items}/{total_items} items in {json_path}")
    
    if clip_scores:
        avg_score = sum(clip_scores) / len(clip_scores)
        min_score = min(clip_scores)
        max_score = max(clip_scores)
        print(f"  Average CLIP score: {avg_score:.4f}")
        print(f"  Min score: {min_score:.4f}, Max score: {max_score:.4f}")

    output_path = json_path.replace(".json", "_clip_scored.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")
    return {
        "output_path": output_path,
        "clip_scores": clip_scores,
    }


def main():
    json_files = [
        "<JSON_FILE_PATH_1>",
        "<JSON_FILE_PATH_2>",
        "<JSON_FILE_PATH_3>",
        "<JSON_FILE_PATH_4>",
        "<JSON_FILE_PATH_5>",
    ]

    all_clip_scores = []

    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()

    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} does not exist, skipping...")
            continue

        try:
            result = process_json_file(json_file, model, preprocess, device)
            all_clip_scores.extend(result["clip_scores"])
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    summary_stats = {
        "total_samples": len(all_clip_scores),
        "average_score": 0.0,
        "min_score": 0.0,
        "max_score": 0.0,
    }

    if all_clip_scores:
        avg_score = sum(all_clip_scores) / len(all_clip_scores)
        min_score = min(all_clip_scores)
        max_score = max(all_clip_scores)
        
        summary_stats["average_score"] = round(avg_score, 4)
        summary_stats["min_score"] = round(min_score, 4)
        summary_stats["max_score"] = round(max_score, 4)
        
        print(f"\nCLIP Cosine Similarity Scores:")
        print(f"  Total samples: {len(all_clip_scores)}")
        print(f"  Average: {avg_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")
        
        score_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        
        for score in all_clip_scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1
        
        print(f"\nScore Distribution:")
        for range_name, count in score_ranges.items():
            percentage = (count / len(all_clip_scores)) * 100
            print(f"  {range_name}: {count} ({percentage:.1f}%)")

    summary_file = "<CLIP_SCORE_SUMMARY_PATH>"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print(f"\nSummary statistics saved to: {summary_file}")
    print("\nAll files processed successfully!")


if __name__ == "__main__":
    main()