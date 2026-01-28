import os
import json
import concurrent.futures
from pathlib import Path
import threading
from typing import List, Dict, Any
from ch_video_gen.utils.llm_utils import call_mlops_llm, call_llm
from ch_video_gen.utils.chemistry_utils import image_to_base64

# Scoring prompt
ch_get_poem_video_review = """
You are a professional reviewer of classical Chinese poetry content. You need to evaluate the script quality of Chinese classical poetry instructional videos.

[Task]
Please read the narration script of the Chinese classical poetry explanatory video below and score it in terms of depth of explanation, knowledge accuracy, and instructional clarity.
Note: if the script contains obvious ASR errors (typos, missing words, garbled text, etc.), do NOT penalize the score for that; instead, score based on the intended content that can be reasonably inferred from the script.

[Video Script]
{video_script}

[Dimensions and Rubric] (all are scored from 1 to 5)

---

### 1. Depth of Explanation
Measures whether the script explains the poem in depth and helps viewers understand its cultural connotations, imagery, and artistic appeal.
Rubric:
- **5 Excellent**: Very comprehensive and deep; covers literal meaning, deeper mood/meaning, creation background, historical allusions, imagery analysis, and rhetorical devices; clear logic; insightful.
- **4 Good**: Fairly deep; covers meaning and background, and explains some imagery/devices, but slightly lacking in breadth or depth.
- **3 Fair**: Covers basic information; explains literal meaning and some background, but analysis is shallow and mentions little about imagery or artistic features.
- **2 Poor**: Superficial; only simple paraphrase, lacking background and deeper analysis.
- **1 Very Poor**: Only surface translation or no effective interpretation; no deep analysis of the poem's meaning.

---

### 2. Knowledge Accuracy
Measures whether the explanation aligns with the poem's intended meaning and historical/cultural facts.
Rubric:
- **5 Excellent**: Fully accurate; interpretations, background, and allusions are correct; information is reliable.
- **4 Good**: Mostly accurate; only very minor issues or optional additions.
- **3 Fair**: Largely correct, but some details are slightly off or not rigorously stated.
- **2 Poor**: Contains clear errors or misunderstandings that may mislead viewers.
- **1 Very Poor**: Severe/systematic errors, or the explanation is entirely incorrect regarding meaning/background.

---

### 3. Instructional Clarity
Measures whether the script is clear, fluent, well-structured, and easy for general viewers (especially K–12 students) to understand.
Rubric:
- **5 Excellent**: Clear structure, concise and easy language, well-layered, good pacing, key points highlighted; almost no comprehension barriers.
- **4 Good**: Clear and fluent; logic mostly explicit, with occasional dense or jumpy parts.
- **3 Fair**: Generally understandable, but some parts are vague, information is a bit cluttered, or logic is not tight.
- **2 Poor**: Scattered explanation, unclear expressions, weak logical connections; hard to follow.
- **1 Very Poor**: Disorganized structure, obscure language; viewers can hardly understand.

---

[Output Requirements]
Output ONLY a JSON object containing:
- depth_of_explanation
- depth_of_explanation_reason
- knowledge_accuracy
- knowledge_accuracy_reason
- instructional_clarity
- instructional_clarity_reason

Example:
```json
{
    "depth_of_explanation": 5,
    "depth_of_explanation_reason": "Covers literal meaning, background, imagery, and rhetoric with insightful analysis...",
    "knowledge_accuracy": 4,
    "knowledge_accuracy_reason": "Mostly accurate; only minor background details could be added...",
    "instructional_clarity": 5,
    "instructional_clarity_reason": "Well-structured and easy to follow for students..."
}
"""


def fill_template(template, data):
    for key, value in data.items():
        template = template.replace("{" + key + "}", "{" + str(value) + "}")
    return template


def message_set(data, prompt, img_path=None):
    task_prompt = fill_template(prompt, data)
    print(f"Processing: {data.get('save_path', 'Unknown file')}")

    if img_path == None:
        return [
            {
                "role": "system",
                "content": "You are a professional reviewer of classical Chinese poetry content. Evaluate the script quality of Chinese classical poetry instructional videos.",
            },
            {"role": "user", "content": task_prompt},
        ]
    else:
        return [
            {
                "role": "system",
                "content": "You are a professional reviewer of classical Chinese poetry content. Evaluate the script quality of Chinese classical poetry instructional videos.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(img_path)}"
                        },
                        "detail": "high",
                    },
                ],
            },
        ]


def load_json_files_from_dir(directory: str) -> List[Dict[str, Any]]:
    json_data_list = []
    dir_path = Path(directory)

    json_files = list(dir_path.glob("*.json"))

    print(f"Found {len(json_files)} JSON files in {directory}")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["save_path"] = str(json_file)
                data["result_path"] = str(
                    json_file.parent / f"{json_file.stem}_scored.json"
                )
                json_data_list.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return json_data_list


def evaluate_single_video(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        reply = call_mlops_llm(
            message_set(data, ch_get_poem_video_review), "call-claude-sonnet-4.5"
        )
        response = reply["choices"][0]["message"]["content"].replace("json", "")
        response = response.split("```")[1]

        score_result = json.loads(response)

        result = {**data, **score_result, "evaluation_status": "success"}

        with open(data.get("save_path"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    except Exception as e:
        print(f"Error evaluating {data.get('save_path', 'unknown')}: {e}")

        result = {
            **data,
            "depth_of_explanation": 0,
            "depth_of_explanation_reason": "Evaluation failed",
            "knowledge_accuracy": 0,
            "knowledge_accuracy_reason": "Evaluation failed",
            "instructional_clarity": 0,
            "instructional_clarity_reason": "Evaluation failed",
            "evaluation_status": "failed",
            "error_message": str(e),
        }

        return {}


def evaluate_videos_concurrent(
    json_data_list: List[Dict[str, Any]], max_workers: int = 5
) -> List[Dict[str, Any]]:
    results = []
    total = len(json_data_list)
    completed = 0
    lock = threading.Lock()

    def process_with_progress(data):
        nonlocal completed
        result = evaluate_single_video(data)

        with lock:
            completed += 1
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {
            executor.submit(process_with_progress, data): data
            for data in json_data_list
        }

        for future in concurrent.futures.as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                data = future_to_data[future]
                print(f"Exception for {data.get('save_path', 'unknown')}: {e}")

    return results


def generate_summary_report(
    results: List[Dict[str, Any]], output_path: str = "evaluation_summary.json"
):
    summary = {
        "total_count": len(results),
        "success_count": sum(
            1 for r in results if r.get("evaluation_status") == "success"
        ),
        "failed_count": sum(
            1 for r in results if r.get("evaluation_status") == "failed"
        ),
        "average_scores": {"depth_of_explanation": 0, "knowledge_accuracy": 0, "instructional_clarity": 0},
        "score_distribution": {
            "depth_of_explanation": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "knowledge_accuracy": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "instructional_clarity": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        },
        "failed_files": [],
    }

    success_results = [r for r in results if r.get("evaluation_status") == "success"]

    if success_results:
        depth_scores = [r["depth_of_explanation"] for r in success_results]
        accuracy_scores = [r["knowledge_accuracy"] for r in success_results]
        clarity_scores = [r["instructional_clarity"] for r in success_results]

        summary["average_scores"]["depth_of_explanation"] = sum(depth_scores) / len(depth_scores)
        summary["average_scores"]["knowledge_accuracy"] = sum(accuracy_scores) / len(
            accuracy_scores
        )
        summary["average_scores"]["instructional_clarity"] = sum(clarity_scores) / len(
            clarity_scores
        )

        for r in success_results:
            depth_score = r["depth_of_explanation"]
            accuracy_score = r["knowledge_accuracy"]
            clarity_score = r["instructional_clarity"]
            summary["score_distribution"]["depth_of_explanation"][depth_score] += 1
            summary["score_distribution"]["knowledge_accuracy"][accuracy_score] += 1
            summary["score_distribution"]["instructional_clarity"][clarity_score] += 1

    for r in results:
        if r.get("evaluation_status") == "failed":
            summary["failed_files"].append(
                {
                    "file": r.get("save_path", "unknown"),
                    "error": r.get("error_message", "Unknown error"),
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation Summary:")
    print(f"Total files: {summary['total_count']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")
    if summary["success_count"] > 0:
        print(f"Average depth_of_explanation score: {summary['average_scores']['depth_of_explanation']:.2f}")
        print(
            f"Average knowledge_accuracy score: {summary['average_scores']['knowledge_accuracy']:.2f}"
        )
        print(
            f"Average instructional_clarity score: {summary['average_scores']['instructional_clarity']:.2f}"
        )


def main():
    json_dir = "<PATH_TO_JSON_RESULTS>"  
    max_workers = 30  

    # 1. Load all JSON files
    print("Loading JSON files...")
    json_data_list = load_json_files_from_dir(json_dir)

    if not json_data_list:
        print("No JSON files found!")
        return

    print(f"Loaded {len(json_data_list)} files")

    # 2. Concurrent evaluation
    print(f"\nStarting concurrent evaluation with {max_workers} workers...")
    results = evaluate_videos_concurrent(json_data_list, max_workers=max_workers)

    # 3. Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        results, output_path=os.path.join(json_dir, "evaluation_summary.json")
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
