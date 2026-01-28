import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
import concurrent.futures
import threading
import subprocess
import traceback
from ch_video_gen.utils.llm_utils import call_mlops_llm
from ch_video_gen.utils.upload_oss import upload_video_from_bytesio


# Evaluation prompt
EVALUATION_PROMPT = """
You are a professional evaluator of educational videos about classical Chinese poetry. You will be given:

1) The full video (visuals, background music, sound effects, narration/recitation, etc.)
2) The full video script (voiceover/explanation text)

Please assess how well the video conveys the poem's original poetic imagery and emotions. Provide scores and analysis for two dimensions: [Poetic Imagery Alignment] and [Emotional Resonance].

If the script contains obvious ASR transcription errors (typos, garbled text, or incomplete sentences that are still inferable), do not intentionally lower the score for these technical artifacts. Evaluate based on your best understanding of the intended explanation and the final audiovisual effect.

--------------------
[Input]

1. Video script (full):
{video_script}

2. Video content:
(Watch the full video. Consider visuals, composition, color palette, pacing, transitions, background music ambience, and narration/recitation emotion. Use the script above to help interpret the video.)

--------------------
[Dimensions and Rubric] (1–5)

1. Poetic Imagery Alignment
Definition: How well the video's audiovisual atmosphere matches the poem's original imagery and aesthetic mood, enabling students to experience emotions consistent with the poem.

Consider: visual style, composition, colors, shot scale, motion and rhythm, music ambience, narration/recitation intonation, etc.

Scoring:
- 5 (Excellent): Highly aligned with the poem’s imagery. Vivid and expressive audiovisual presentation with a fully consistent aesthetic mood. Viewers feel immersed and strongly resonate with the poem’s atmosphere.
- 4 (Good): Largely consistent with the poem’s imagery and mood. The main emotions and aesthetics are well conveyed, with only minor details that could be more evocative.
- 3 (Fair): Generally matches the poem’s emotional tone without obvious mismatch, but the expressiveness is moderate. The poem-like atmosphere is present but not deeply recreated.
- 2 (Poor): Low alignment. The video does not effectively convey the poem’s imagery overall. Some elements attempt to correspond, but the result feels weak or unconvincing.
- 1 (Very Poor): Severely misaligned. The style or mood contradicts the poem (e.g., a sorrowful poem presented with a bright, playful cartoon tone). The poetic atmosphere fails to transmit.

2. Emotional Resonance
Definition: The extent to which the video evokes emotional resonance, conveying the poem’s emotional tone and atmosphere, leading primary/secondary students to feel emotionally engaged or empathic.

Consider: whether visuals, music, sound effects, narration emotion, pacing, and explanation style work in the same emotional direction and genuinely move the audience (rather than being superficial).

Scoring:
- 5 (Excellent): Strong emotional impact. The poem’s emotional tone is clearly and powerfully conveyed; viewers are emotionally engaged and empathize naturally.
- 4 (Good): Clear emotional tone and good engagement. Viewers can relate emotionally, with only minor room for stronger emotional driving force.
- 3 (Fair): The emotion is understandable but not very moving. Engagement exists but is limited or somewhat generic.
- 2 (Poor): Weak emotional delivery. The video feels emotionally flat; resonance is difficult to build.
- 1 (Very Poor): Emotion is largely absent or contradictory to the poem. Viewers feel little to no emotional fluctuation and may become confused about the poem’s sentiment.

--------------------
[Output]

Strictly output your evaluation in the following JSON format. Do not add any extra text:

```json
{
  "poetic_imagery_alignment": 4,
  "poetic_imagery_alignment_reason": "Write detailed reasons for the score here",
  "emotional_resonance": 5,
  "emotional_resonance_reason": "Write detailed reasons for the score here"
}
"""


def upload_video_with_retry(video_path: str, max_retries: int = 3) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Video size: {file_size_mb:.2f}MB")
    print(f"  Video path: {video_path}")
    video_path = compress_video_if_needed(video_path)
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()  
        print(f"  Read {len(video_bytes)} bytes from file")
    except Exception as e:
        raise Exception(f"Failed to read video file: {e}")

    filename = os.path.basename(video_path)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Upload attempt {attempt}/{max_retries}")

            video_data = BytesIO(video_bytes)

            video_url = upload_video_from_bytesio(video_data, filename)

            if video_url and video_url.startswith(("http://", "https://")):
                print(f"  ✓ Upload successful: {video_url}")
                return video_url
            else:
                raise ValueError(f"Invalid video URL: {video_url}")

        except Exception as e:
            print(f"  ✗ Upload failed (attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                wait_time = 2 * attempt
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                import traceback

                print(f"  Full error trace:\n{traceback.format_exc()}")
                raise Exception(
                    f"Failed to upload {filename} after {max_retries} attempts.\n"
                    f"Last error: {str(e)}"
                )


def compress_video_if_needed(video_path: str, max_size_mb: float = 30) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Checking video: {os.path.basename(video_path)}")
    print(f"  Current size: {file_size_mb:.2f}MB")

    if file_size_mb <= max_size_mb:
        print(f"  ✓ Size OK, no compression needed")
        return video_path

    min_compress_threshold = max_size_mb * 0.6
    if file_size_mb <= min_compress_threshold:
        print(f"  ✓ Size ({file_size_mb:.2f}MB) is reasonable, no compression needed")
        return video_path


    print(f"  Size exceeds {max_size_mb}MB, compressing...")

    video_path_obj = Path(video_path)
    compressed_path = str(
        video_path_obj.parent / f"{video_path_obj.stem}_compressed.mp4"
    )

    try:
        print(f"  Step 1: Probing video information...")
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=30
        )

        if probe_result.returncode != 0:
            raise Exception(f"Failed to probe video: {probe_result.stderr}")

        probe_data = json.loads(probe_result.stdout)
        duration = float(probe_data["format"]["duration"])
        print(f"    Video duration: {duration:.1f}s")

        target_size_mb = max_size_mb * 0.75
        target_bitrate = int((target_size_mb * 8 * 1024 * 1024) / duration)
        print(f"  Step 2: Calculating target bitrate: {target_bitrate} bps")

        print(f"  Step 3: Compressing video with fast preset...")
        compress_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-b:v",
            str(target_bitrate),
            "-maxrate",
            str(target_bitrate),
            "-bufsize",
            str(target_bitrate // 2),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",  
            "-y",
            compressed_path,
        ]

        try:
            compress_result = subprocess.run(
                compress_cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"    Fast preset timeout, trying ultrafast preset...")
            compress_cmd[17] = "ultrafast"  
            target_size_mb = max_size_mb * 0.5
            target_bitrate = int((target_size_mb * 8 * 1024 * 1024) / duration)
            compress_cmd[3] = str(target_bitrate)
            compress_cmd[5] = str(target_bitrate)
            compress_cmd[7] = str(target_bitrate // 2)
            print(f"  Retry with ultrafast preset, bitrate: {target_bitrate} bps")

            compress_result = subprocess.run(
                compress_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

        if compress_result.returncode != 0:
            raise Exception(f"FFmpeg compression failed: {compress_result.stderr}")

        if not os.path.exists(compressed_path):
            raise Exception("Compressed file not created")

        compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
        print(f"  ✓ Compression successful!")
        print(f"    Original: {file_size_mb:.2f}MB")
        print(f"    Compressed: {compressed_size_mb:.2f}MB")
        print(f"    Reduction: {(1 - compressed_size_mb/file_size_mb)*100:.1f}%")
        print(f"    Saved to: {compressed_path}")

        if compressed_size_mb > max_size_mb:
            print(
                f"   Compressed size ({compressed_size_mb:.2f}MB) still exceeds {max_size_mb}MB"
            )

        return compressed_path

    except subprocess.TimeoutExpired:
        print(f"  ✗ Compression timeout after ultrafast retry")
        raise Exception("Video compression timeout")

    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        if os.path.exists(compressed_path):
            try:
                os.remove(compressed_path)
                print(f"  Cleaned up incomplete compressed file")
            except:
                pass
        raise



def create_evaluation_messages(video_script: str, video_url: str) -> List[Dict]:
    prompt = EVALUATION_PROMPT.format(video_script=video_script)

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": video_url, "mime_type": "video/mp4"},
        },
    ]

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


def call_llm_with_retry(
    messages: List[Dict],
    model_name: str = "call-gemini-2.0-flash",
    max_retries: int = 5,
) -> Dict[str, Any]:
    import traceback

    last_error_details = None 

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Evaluation attempt {attempt}/{max_retries}")
            print(messages)
            reply = call_mlops_llm(messages, model_name)

            if not reply or "choices" not in reply:
                raise ValueError(f"Invalid API response: {reply}")

            if not reply["choices"] or len(reply["choices"]) == 0:
                raise ValueError(f"Empty choices in API response: {reply}")

            response = reply["choices"][0]["message"]["content"]
            print(f"  Raw response length: {len(response)} chars")
            print(f"  Raw response preview: {response[:200]}...")

            original_response = response  

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
                print("  Extracted JSON from ```json code block")
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                print("  Extracted JSON from ``` code block")

            try:
                result = json.loads(response)
            except json.JSONDecodeError as json_err:
                print(f"  JSON parsing failed:")
                print(f"    Error: {json_err}")
                print(
                    f"    Error position: line {json_err.lineno}, column {json_err.colno}"
                )
                print(f"    Cleaned response: {response[:500]}...")
                print(f"    Original response: {original_response[:500]}...")
                raise

            if "poetic_imagery_alignment" in result and "emotional_resonance" in result:
                print(f"  ✓ Evaluation successful")
                print(f"    poetic_imagery_alignment: {result['poetic_imagery_alignment']}")
                print(f"    emotional_resonance: {result['emotional_resonance']}")
                return result
            else:
                missing_fields = []
                if "poetic_imagery_alignment" not in result:
                    missing_fields.append("poetic_imagery_alignment")
                if "emotional_resonance" not in result:
                    missing_fields.append("emotional_resonance")

                error_msg = f"Invalid response format: missing fields {missing_fields}"
                print(f"  {error_msg}")
                print(f"    Received fields: {list(result.keys())}")
                print(
                    f"    Response: {json.dumps(result, ensure_ascii=False, indent=2)}"
                )
                raise ValueError(error_msg)

        except json.JSONDecodeError as e:
            error_details = {
                "attempt": attempt,
                "error_type": "JSONDecodeError",
                "error_message": str(e),
                "error_line": getattr(e, "lineno", None),
                "error_column": getattr(e, "colno", None),
                "response_length": len(response) if "response" in locals() else 0,
                "response_preview": response[:500] if "response" in locals() else "N/A",
                "traceback": traceback.format_exc(),
            }
            last_error_details = error_details

            print(f"  ✗ JSON parsing error (attempt {attempt}/{max_retries}):")
            print(f"    Error: {e}")
            print(
                f"    Position: line {error_details['error_line']}, col {error_details['error_column']}"
            )
            print(f"    Response preview: {error_details['response_preview']}")
            print(f"    Full traceback:\n{error_details['traceback']}")

        except Exception as e:
            error_details = {
                "attempt": attempt,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model_name": model_name,
                "messages_count": len(messages),
                "traceback": traceback.format_exc(),
            }
            last_error_details = error_details

            print(f"  ✗ Evaluation failed (attempt {attempt}/{max_retries}):")
            print(f"    Error type: {error_details['error_type']}")
            print(f"    Error message: {error_details['error_message']}")
            print(f"    Model: {model_name}")
            print(error_details["traceback"])
            if (
                "API" in str(e)
                or "connection" in str(e).lower()
                or "timeout" in str(e).lower()
            ):
                print(f"    This appears to be an API/network issue")

            print(f"    Full traceback:\n{error_details['traceback']}")

        if attempt < max_retries:
            wait_time = 2 * attempt
            print(f"  Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    error_summary = {
        "total_attempts": max_retries,
        "model_name": model_name,
        "last_error": last_error_details,
        "message": f"Failed to evaluate after {max_retries} attempts",
    }

    print(f"\n{'='*60}")
    print(f"All evaluation attempts failed!")
    print(f"{'='*60}")
    print(f"Total attempts: {max_retries}")
    print(f"Model: {model_name}")
    if last_error_details:
        print(f"Last error type: {last_error_details['error_type']}")
        print(f"Last error message: {last_error_details['error_message']}")
        print(f"\nFull details:")
        print(json.dumps(last_error_details, ensure_ascii=False, indent=2, default=str))
    print(f"{'='*60}\n")

    raise Exception(
        f"Failed to evaluate after {max_retries} attempts.\n"
        f"Last error: {last_error_details['error_type']} - {last_error_details['error_message']}\n"
        f"See logs for full details."
    )


def process_single_video(json_file_path: str) -> Dict[str, Any]:
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = Path(json_file_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"{'='*60}")

        if "video_path" not in data:
            raise ValueError("Missing 'video_path' in JSON data")

        video_path = data["video_path"]
        video_script = data.get("video_script", "")

        print(f"Video path: {video_path}")

        print("Step 1: Uploading video...")
        video_url = upload_video_with_retry(video_path, max_retries=3)

        print("Step 2: Creating evaluation messages...")
        messages = create_evaluation_messages(video_script, video_url)

        print("Step 3: Calling LLM for evaluation...")
        result = call_llm_with_retry(messages, max_retries=5)

        final_result = {
            **data,
            **result,
            "video_url": video_url,
            "video_name": video_name,
            "evaluation_status": "success",
            "json_file": json_file_path,
        }

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        print(f"✓ Successfully processed {video_name}")
        return final_result

    except Exception as e:
        print(f"✗ Failed to process {video_name}: {e}")
        return {
            "video_name": video_name,
            "json_file": json_file_path,
            "evaluation_status": "failed",
            "error_message": str(e),
        }


def load_json_files(json_dir: str) -> List[str]:
    json_dir_path = Path(json_dir)

    if not json_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {json_dir}")

    json_files = list(json_dir_path.glob("*.json"))

    json_files = [f for f in json_files if "summary" not in f.stem.lower()]

    print(f"Found {len(json_files)} JSON files in {json_dir}")
    return [str(f) for f in json_files]


def process_videos_concurrent(
    json_files: List[str], max_workers: int = 5
) -> List[Dict[str, Any]]:
    results = []
    total = len(json_files)
    completed = 0
    lock = threading.Lock()

    def process_with_progress(json_file):
        nonlocal completed
        result = process_single_video(json_file)

        with lock:
            completed += 1
            print(f"\n{'='*60}")
            print(f"Overall Progress: {completed}/{total} ({completed/total*100:.1f}%)")
            print(f"{'='*60}\n")

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_with_progress, json_file): json_file
            for json_file in json_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                json_file = future_to_file[future]
                print(f"Exception processing {json_file}: {e}")
                results.append(
                    {
                        "json_file": json_file,
                        "evaluation_status": "failed",
                        "error_message": str(e),
                    }
                )

    return results


def generate_summary_report(results: List[Dict[str, Any]], output_path: str):
    success_results = [r for r in results if r.get("evaluation_status") == "success"]
    failed_results = [r for r in results if r.get("evaluation_status") == "failed"]

    summary = {
        "total_count": len(results),
        "success_count": len(success_results),
        "failed_count": len(failed_results),
        "statistics": {},
        "failed_files": [],
    }

    if success_results:
        poetic_scores = [r.get("poetic_imagery_alignment", 0) for r in success_results]
        emotional_scores = [r.get("emotional_resonance", 0) for r in success_results]

        summary["statistics"] = {
            "poetic_imagery_alignment": {
                "average": round(sum(poetic_scores) / len(poetic_scores), 2),
                "distribution": {str(i): poetic_scores.count(i) for i in range(1, 6)},
            },
            "emotional_resonance": {
                "average": round(sum(emotional_scores) / len(emotional_scores), 2),
                "distribution": {
                    str(i): emotional_scores.count(i) for i in range(1, 6)
                },
            },
        }

    for r in failed_results:
        summary["failed_files"].append(
            {
                "file": r.get("video_name", r.get("json_file", "unknown")),
                "error": r.get("error_message", "Unknown error"),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total: {summary['total_count']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")

    if success_results:
        stats = summary["statistics"]
        print(f"\n poetic_imagery_alignment:")
        print(f"  Average: {stats['poetic_imagery_alignment']['average']}")
        print(f"  Distribution: {stats['poetic_imagery_alignment']['distribution']}")

        print(f"\n emotional_resonance:")
        print(f"  Average: {stats['emotional_resonance']['average']}")
        print(f"  Distribution: {stats['emotional_resonance']['distribution']}")

    if failed_results:
        print(f"\n Failed files:")
        for failed in summary["failed_files"][:5]:
            print(f"  - {failed['file']}: {failed['error'][:100]}")

    print(f"\nReport saved to: {output_path}")
    print(f"{'='*60}")


def main():
    json_dir = "<JSON_DIR>"
    max_workers = 30  
    print("=" * 60)
    print("Video Evaluation System")
    print("=" * 60)
    print(f"JSON directory: {json_dir}")
    print(f"Max workers: {max_workers}")
    print("=" * 60)

    try:
        print("\nStep 1: Loading JSON files...")
        json_files = load_json_files(json_dir)

        if not json_files:
            print("No JSON files found!")
            return

        print(f"\nStep 2: Processing {len(json_files)} videos...")
        results = process_videos_concurrent(json_files, max_workers=max_workers)

        print("\nStep 3: Generating summary report...")
        summary_path = os.path.join(json_dir, "evaluation_summary.json")
        generate_summary_report(results, summary_path)

        print("\n All done!")

    except Exception as e:
        print(f" Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
