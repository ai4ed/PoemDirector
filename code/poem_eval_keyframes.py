import os
import json
import concurrent.futures
from pathlib import Path
import threading
import sys
from typing import List, Dict, Any, Optional
from ch_video_gen.utils.llm_utils import call_mlops_llm, call_llm
from ch_video_gen.utils.chemistry_utils import image_to_base64

# ============== Two independent scoring prompts ==============

# Dimension 1: Aesthetic Coherence scoring prompt
ch_get_aesthetic_coherence = """
You are a professional educational content reviewer. You need to score the *aesthetic coherence* of a Chinese classical poetry instructional video.
Based on the provided visual elements and the narration script, evaluate how consistent and harmonious the video is in terms of aesthetic style.

[Task]
Given 5–10 keyframe images and the full video script, score the video's aesthetic coherence.
"Aesthetic coherence" refers to the consistency and coordination of visual and audio elements (e.g., visual style, color palette, typography/subtitles, transitions/animation, background music/sound effects) and whether they match the poem's theme and artistic mood. Your score should reflect the overall multimodal perception and the learner's sense of immersion.

[Input]
1) Keyframe images (5–10):
- {keyframes_images}

2) Video script (narration):
- {video_script}

[Scoring rubric] (1–5)
- 5 Excellent: Highly unified and artistically compelling; all visual/audio elements align perfectly with the poem’s theme and emotional tone; smooth transitions and natural pacing; strong immersive aesthetic experience.
- 4 Good: Overall coherent aesthetic style; visuals, colors, music, and pacing are consistent and appropriate; transitions are natural; only minor details could be improved.
- 3 Fair: Generally coherent; occasional inconsistencies or slightly abrupt elements, but overall watchability is not severely affected.
- 2 Poor: Clear aesthetic inconsistencies; some scenes/styles deviate from the overall look; mismatched colors/music/transitions; noticeably harms aesthetic experience.
- 1 Very poor: Extremely incoherent; visual elements clash; unpleasant colors; audio atmosphere does not match visuals; confusing overall experience.

Note: When scoring, distinguish dialogue scenes—if a shot intentionally depicts character-to-character dialogue, do not penalize aesthetic coherence for that reason alone.

[Output format]
Return **only** a JSON object with:
- "Aesthetic Coherence Score" (1–5)
- "Aesthetic Coherence Rationale" (a detailed explanation)

Example:
```json
{
  "Aesthetic Coherence Score": 4,
  "Aesthetic Coherence Rationale": "The visual style matches the poem’s theme and the color palette fits the mood. The music generally complements the visuals, but a few transitions feel slightly abrupt, reducing immersion."
}
"""

# Dimension 2: Poetic Imagery Restoration scoring prompt
ch_get_poetic_imagery_restoration = """
You are a professional educational content reviewer. You need to assess the *poetic imagery restoration* in a Chinese classical poetry instructional video.
Based on the provided video script and keyframe images, evaluate whether the key poetic images (people, scenery, objects, atmosphere, emotions) are reproduced completely and accurately.

[Task]
Read the full script and examine the keyframe image list. Each image corresponds to a segment of the script.
Assess whether the video restores the poem’s major imagery accurately and completely. Focus on whether the imagery, scenes, characters, and emotional atmosphere described in the poem are sufficiently represented visually.

[Input]
1) Keyframe images (5–10):
- {keyframse_images_content}

2) Video script (narration):
- {video_script}

[Scoring rubric] (1–5)
- 5 Excellent: Key imagery is fully and accurately restored; visuals strongly match the poem’s imagery and mood with rich details; no important imagery is missing.
- 4 Good: Most key imagery is restored accurately; minor omissions or simplifications exist but do not significantly affect comprehension.
- 3 Fair: Some key imagery is shown, but there are noticeable omissions or inaccuracies; the overall restoration is acceptable but not strong.
- 2 Poor: Many key images are missing or inaccurately depicted; visuals only weakly match the poem’s imagery and mood.
- 1 Very poor: Imagery is largely not restored; visuals do not correspond to the poem’s key imagery; severe mismatch.

[Output format]
Return **only** a JSON object with:
- "Poetic Imagery Restoration Score" (1–5)
- "Poetic Imagery Restoration Rationale" (a detailed explanation)

Example:
```json
{
  "Poetic Imagery Restoration Score": 4,
  "Poetic Imagery Restoration Rationale": "The video depicts major images such as 'Chang'an city' and 'morning glow', but some imagery (e.g., 'a lone boat') is not fully shown and the visual expression is simplified."
}
"""

# ============== Helper functions ==============


def fill_template(template, data):
    for key, value in data.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def get_keyframe_images_from_json(video_data: Dict[str, Any]) -> List[str]:
    frame_content = video_data.get("frame_content", [])
    image_urls = []
    for frame in frame_content:
        image_urls.append(frame.get("img_path", ""))
    frame_content_content = []
    for frame in frame_content:
        frame_content_content.append(
            {
                "Image:": frame.get("img_path", ""),
                "Current segment:": frame.get("content", ""),
            }
        )

    return image_urls, frame_content_content


def message_set(data, prompt, img_path=None):
    task_prompt = fill_template(prompt, data)
    print(task_prompt)
    if img_path == None:
        return [
            {
                "role": "system",
                "content": "This is a brand-new conversation. If you remember any previous conversation, please forget it completely.",
            },
            {"role": "user", "content": task_prompt},
        ]
    else:
        temp_list = []
        temp_list.append({"type": "text", "text": task_prompt})
        for img in img_path:
            temp_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": img, "detail": "high"},
                }
            )
        return [
            {
                "role": "system",
                "content": "This is a brand-new conversation. If you remember any previous conversation, please forget it completely.",
            },
            {"role": "user", "content": temp_list},
        ]


def evaluate_single_dimension(
    data: Dict[str, Any],
    dimension_name: str,
    dimension_prompt: str,
    image_urls: List[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        if dimension_name == "Aesthetic Coherence":
            if not image_urls:
                return {"Aesthetic Coherence Score": 0, "Aesthetic Coherence Rationale": "No image data"}
            messages = message_set(data, dimension_prompt, image_urls)
        else:
            messages = message_set(data, dimension_prompt)

        reply = call_mlops_llm(messages, "call-claude-sonnet-4.5")
        response = reply["choices"][0]["message"]["content"].replace("json", "")
        response = response.split("```")[1]

        result = json.loads(response)
        return result

    except Exception as e:
        print(f"Error evaluating {dimension_name}: {e}")
        if dimension_name == "Aesthetic Coherence":
            return {"Aesthetic Coherence Score": 0, "Aesthetic Coherence Rationale": f"Evaluation failed: {str(e)}"}
        else:
            return {"Poetic Imagery Restoration Score": 0, "Poetic Imagery Restoration Rationale": f"Evaluation failed: {str(e)}"}


def evaluate_single_poem(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        image_urls, frame_content_content = get_keyframe_images_from_json(data)

        video_script = data.get("video_script", "")

        eval_data = {
            "video_script": video_script,
            "keyframes_images": "\n".join(
                [f"- Image {i+1}: {url}" for i, url in enumerate(image_urls)]
            ),
            "keyframse_images_content": frame_content_content,
        }

        results = {}

        print(f"    Evaluating Aesthetic Coherence for {data.get('poem_title', 'unknown')}")
        aesthetic_result = evaluate_single_dimension(
            eval_data, "Aesthetic Coherence", ch_get_aesthetic_coherence, image_urls
        )
        if aesthetic_result:
            results.update(aesthetic_result)

        print(f"    Evaluating Poetic Imagery Restoration for {data.get('poem_title', 'unknown')}")
        imagery_result = evaluate_single_dimension(
            eval_data, "Poetic Imagery Restoration", ch_get_poetic_imagery_restoration
        )
        if imagery_result:
            results.update(imagery_result)

        result = {**data, **results, "evaluation_status": "success"}

        save_path = data.get("save_path")
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"    Results saved to: {save_path}")
            except Exception as e:
                print(f"    Error saving to {save_path}: {e}")

        return result

    except Exception as e:
        print(f"Error evaluating poem: {e}")
        result = {**data, "evaluation_status": "failed", "error_message": str(e)}
        return result


def load_poem_data_combined(json_dir: str, all_videos_json_path: str) -> List[Dict[str, Any]]:
    poem_data_list = []

    try:
        # 1. Load frame_content from all_videos_processed.json
        frame_content_data = {}
        if os.path.exists(all_videos_json_path):
            with open(all_videos_json_path, "r", encoding="utf-8") as f:
                frame_content_data = json.load(f)

        # 2. Load individual poem JSON files
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and not f.endswith('_clip_scored.json') and not f.endswith('_scored.json') and not f.endswith('_evaluation_summary.json') and not f.endswith('_evaluation_results.json')]

        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    poem_data = json.load(f)

                if "video_script" in poem_data and poem_data["video_script"]:
                    poem_title = poem_data.get("title", "")
                    if not poem_title:
                        poem_title = json_file.replace("古诗词_", "").replace("_processed.json", "")

                    if poem_title in frame_content_data:
                        frame_data = frame_content_data[poem_title]
                        if "frame_content" in frame_data and frame_data["frame_content"]:
                            poem_data["frame_content"] = frame_data["frame_content"]
                            poem_data["poem_title"] = poem_title
                            poem_data["save_path"] = json_path
                            poem_data_list.append(poem_data)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

    except Exception as e:
        print(f"Error in combined loading: {e}")

    return poem_data_list


def load_poem_data_from_dir(json_dir: str) -> List[Dict[str, Any]]:
    poem_data_list = []

    try:
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and not f.endswith('_clip_scored.json') and not f.endswith('_scored.json')]

        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    poem_data = json.load(f)

                if "frame_content" in poem_data and poem_data["frame_content"]:
                    poem_data["save_path"] = json_path
                    poem_data_list.append(poem_data)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

    except Exception as e:
        print(f"Error reading directory {json_dir}: {e}")

    return poem_data_list


def evaluate_poems_concurrent(
    poem_data_list: List[Dict[str, Any]], max_workers: int = 5
) -> List[Dict[str, Any]]:
    results = []
    total = len(poem_data_list)
    completed = 0
    lock = threading.Lock()

    def process_with_progress(data):
        nonlocal completed
        result = evaluate_single_poem(data)

        with lock:
            completed += 1
            print(
                f"\nOverall Progress: {completed}/{total} ({completed/total*100:.1f}%) poems completed\n"
            )

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {
            executor.submit(process_with_progress, data): data
            for data in poem_data_list
        }

        for future in concurrent.futures.as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                data = future_to_data[future]
                print(f"Exception for {data.get('poem_title', 'unknown')}: {e}")

    return results


def generate_summary_report(
    results: List[Dict[str, Any]], output_path: str = "poem_evaluation_summary.json"
):
    summary = {
        "total_count": len(results),
        "success_count": sum(
            1 for r in results if r.get("evaluation_status") == "success"
        ),
        "failed_count": sum(
            1 for r in results if r.get("evaluation_status") == "failed"
        ),
        "average_scores": {"aesthetic_coherence": 0, "imagery_restoration": 0},
        "score_distribution": {
            "aesthetic_coherence": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "imagery_restoration": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        },
        "failed_poems": [],
    }

    success_results = [r for r in results if r.get("evaluation_status") == "success"]

    if success_results:
        aesthetic_scores = [
            r.get("Aesthetic Coherence Score", 0)
            for r in success_results
            if r.get("Aesthetic Coherence Score", 0) > 0
        ]
        imagery_scores = [
            r.get("Imagery Restoration Score", 0)
            for r in success_results
            if r.get("Imagery Restoration Score", 0) > 0
        ]

        if aesthetic_scores:
            summary["average_scores"]["aesthetic_coherence"] = round(
                sum(aesthetic_scores) / len(aesthetic_scores), 2
            )
        if imagery_scores:
            summary["average_scores"]["imagery_restoration"] = round(
                sum(imagery_scores) / len(imagery_scores), 2
            )

        for r in success_results:
            aes_score = round(r.get("Aesthetic Coherence Score", 0))
            img_score = round(r.get("Imagery Restoration Score", 0))

            if aes_score in [1, 2, 3, 4, 5]:
                summary["score_distribution"]["aesthetic_coherence"][aes_score] += 1
            if img_score in [1, 2, 3, 4, 5]:
                summary["score_distribution"]["imagery_restoration"][img_score] += 1

    for r in results:
        if r.get("evaluation_status") == "failed":
            summary["failed_poems"].append(
                {
                    "poem": r.get("poem_title", "unknown"),
                    "error": r.get("error_message", "Unknown error"),
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Poem Evaluation Summary:")
    print(f"{'='*60}")
    print(f"Total poems: {summary['total_count']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")

    if summary["success_count"] > 0:
        print(f"\nAverage Scores:")
        print(
            f"  Aesthetic Coherence: {summary['average_scores']['aesthetic_coherence']:.2f}"
        )
        print(
            f"  Imagery Restoration: {summary['average_scores']['imagery_restoration']:.2f}"
        )

        print(f"\nScore Distribution:")
        for dimension in ["aesthetic_coherence", "imagery_restoration"]:
            dist = summary["score_distribution"][dimension]
            print(
                f"  {dimension}: [1]={dist[1]}, [2]={dist[2]}, [3]={dist[3]}, [4]={dist[4]}, [5]={dist[5]}"
            )

    print(f"\nSummary report saved to: {output_path}")
    print(f"{'='*60}")


def main():
    base_path = (
        "./result/poem_baseline"
    )
    datasets = [
        "ijcai2026_poemdirector",
    ]
    max_workers = 30  


    for dataset in datasets:
        print("=" * 80)
        print(f"Processing dataset: {dataset}")
        print("=" * 80)


        json_dir = os.path.join(base_path, dataset, "json_results")
        processed_dir = dataset.replace("ijcai2026_", "") + "_processed"
        poem_data_json = os.path.join(
            base_path,
            dataset,
            processed_dir,
            "json_results",
            "all_videos_processed.json",
        )

        if not os.path.exists(poem_data_json):
            print(f"Warning: {poem_data_json} does not exist, skipping...")
            continue

        print(f"\nStep 1: Loading poem data from {json_dir} and {poem_data_json}...")
        poem_data_list = load_poem_data_combined(json_dir, poem_data_json)

        if not poem_data_list:
            print("No poem data found!")
            continue

        print(f"Loaded {len(poem_data_list)} poems")

        print(f"\nStep 2: Starting concurrent evaluation with {max_workers} workers...")
        print("Note: Each poem will be evaluated on 2 dimensions.")
        print("-" * 60)
        results = evaluate_poems_concurrent(poem_data_list, max_workers=max_workers)

        print("\nStep 3: Generating summary report...")
        generate_summary_report(
            results, output_path=os.path.join(json_dir, "poem_evaluation_summary.json")
        )

        print(f"\n Dataset {dataset} evaluation completed successfully!")
        print()

    print("\n All datasets processed successfully!")


if __name__ == "__main__":
    main()
