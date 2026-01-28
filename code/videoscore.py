# run.py (OOM-safe + offline-friendly + DynamicCache hotfix)

# ---- hotfix for transformers DynamicCache API ----
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_usable_length = DynamicCache.get_seq_length
except Exception:
    pass
# -----------------------------------------------

import os
import re
import gc
from typing import List, Optional

import av
import numpy as np
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification


# ===================== Configuration =====================

VIDEO_DIR = "<PATH_TO_VIDEOS>"            
OUTPUT_EXCEL = "<OUTPUT_XLSX_PATH>"
ROUND_DIGIT = 3


MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"



MAX_NUM_FRAMES = 16


LOCAL_FILES_ONLY = False

REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8
For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


# ===================== Utilities =====================

def decode_hash_u(name: str) -> str:
    def repl(match):
        hexcode = match.group(1)
        try:
            return chr(int(hexcode, 16))
        except ValueError:
            return match.group(0)
    return re.sub(r"#U([0-9a-fA-F]{4})", repl, name)


def sample_indices(total_frames: Optional[int], max_frames: int) -> np.ndarray:
    if total_frames is None or total_frames <= 0:
        return np.arange(max_frames, dtype=int)

    if total_frames <= max_frames:
        return np.arange(total_frames, dtype=int)

    idx = np.linspace(0, total_frames - 1, max_frames).astype(int)
    return np.unique(idx)


def read_video_pyav(video_path: str, indices: np.ndarray) -> np.ndarray:
    container = av.open(video_path)
    frames = []

    indices = np.asarray(indices, dtype=int)
    indices = indices[indices >= 0]
    if len(indices) == 0:
        container.close()
        raise ValueError(f"indices is empty: {video_path}")

    index_set = set(indices.tolist())
    max_i = int(indices.max())

    for i, frame in enumerate(container.decode(video=0)):
        if i > max_i:
            break
        if i in index_set:
            frames.append(frame)

    container.close()

    if len(frames) == 0:
        raise ValueError(f"Failed to read frames from video: {video_path}")

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def _maybe_empty_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_one_video(video_path: str, model, processor, device, text_prompt: str, max_frames: int) -> List[float]:
    # 1) Get total frame count
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    container.close()

    indices = sample_indices(total_frames, max_frames)

    # 2) Read frames
    video_array = read_video_pyav(video_path, indices)
    frames = [Image.fromarray(x) for x in video_array]

    # 3) Pad the prompt with <image>
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=text_prompt)
    if eval_prompt.count("<image>") < len(frames):
        eval_prompt += "<image> " * (len(frames) - eval_prompt.count("<image>"))

    # 4) Build inputs
    inputs = processor(text=eval_prompt, images=frames, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    # 5) Inference: memory-friendly inference_mode + autocast
    if device.type == "cuda":
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**inputs)
    else:
        with torch.inference_mode():
            outputs = model(**inputs)

    logits = outputs.logits[0]  # shape [5]
    scores = [round(logits[i].item(), ROUND_DIGIT) for i in range(logits.shape[-1])]

    # 6) Release large objects (critical to avoid OOM on later videos)
    del inputs, outputs, frames, video_array, logits
    _maybe_empty_cuda_cache()

    return scores


def score_one_video_with_retry(video_path: str, model, processor, device, text_prompt: str) -> List[float]:
    frame_trials = [MAX_NUM_FRAMES, max(8, MAX_NUM_FRAMES // 2), 8]
    last_err = None

    for mf in frame_trials:
        try:
            return score_one_video(video_path, model, processor, device, text_prompt, max_frames=mf)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cuda out of memory" in msg or "out of memory" in msg:
                last_err = e
                print(f"  OOM: lowering frames to {mf} failed; retrying...")
                _maybe_empty_cuda_cache()
                continue
            raise 

    # All attempts failed
    raise last_err if last_err else RuntimeError("Unknown error")


# ===================== Main =====================

def main():
    print(f"Loading model：{MODEL_NAME}")
    print(f"MAX_NUM_FRAMES={MAX_NUM_FRAMES}, LOCAL_FILES_ONLY={LOCAL_FILES_ONLY}")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = Idefics2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        local_files_only=LOCAL_FILES_ONLY,
    ).eval()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1) Collect videos
    video_dir_abs = os.path.abspath(VIDEO_DIR)
    video_files = [
        os.path.join(video_dir_abs, f)
        for f in os.listdir(video_dir_abs)
        if f.lower().endswith(".mp4")
    ]
    video_files.sort()
    print(f"Found {len(video_files)} videos.")

    results = []

    # 2) Batch scoring
    for idx, video_path in enumerate(video_files, start=1):
        video_name = os.path.basename(video_path)
        poem_name_raw = os.path.splitext(video_name)[0]
        poem_name = decode_hash_u(poem_name_raw)

        print(f"[{idx}/{len(video_files)}] Scoring：{poem_name} ({video_name})")
        text_prompt = f"This is a video explaining the ancient Chinese poem '{poem_name}'."

        try:
            scores = score_one_video_with_retry(video_path, model, processor, device, text_prompt)
            visual, temporal, dynamic, text_align, factual = scores
            mean_score = round(sum(scores) / len(scores), ROUND_DIGIT)

            results.append({
                "filename": video_name,
                "poem_name": poem_name,
                "visual_quality": visual,
                "temporal_consistency": temporal,
                "dynamics": dynamic,
                "text_alignment": text_align,
                "factual_consistency": factual,
                "mean_score": mean_score,
            })

        except Exception as e:
            print(f" Scoring failed：{video_name}, reason：{e}")
            results.append({
                "filename": video_name,
                "poem_name": poem_name,
                "visual_quality": None,
                "temporal_consistency": None,
                "dynamics": None,
                "text_alignment": None,
                "factual_consistency": None,
                "mean_score": None,
                "error": str(e),
            })
            _maybe_empty_cuda_cache()

    # 3) Export to Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Excel saved:{os.path.abspath(OUTPUT_EXCEL)}")


if __name__ == "__main__":
    main()

