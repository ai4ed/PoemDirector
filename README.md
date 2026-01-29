# PoemDirector: Dataset and Evaluation Code

This repository contains the dataset and evaluation scripts for the paper **"PoemDirector: A Multi-Agent Context-Adaptive Instructional Mode Selection Framework for Chinese Classical Poetry Video Generation"**.

## 📂 Dataset

Due to file size limitations, the full video dataset (100 poems × 5 methods) is hosted on Hugging Face.

Please download and unzip **`poem_dataset.zip`** from the link below:

**[Download Dataset via Hugging Face](https://huggingface.co/datasets/userasadfasdf/poem_data/tree/main)**

Once unzipped, the dataset is organized by method:
* **`PoemDirector`**: Our proposed method.
* **`HAV`**: Handcrafted Animation Videos (Human Expert).
* **`Mootion`**: Commercial AI generation baseline.
* **`ESV`**: Explanatory Short Videos (Baseline).
* **`PRV`**: Poetry Recitation Videos (Baseline).

*Note: Files in each folder are named by poem title.*

## Evaluation Code

The `code/` directory contains scripts for the automated metrics reported in the paper.

| File                         | Function / Metric                                            |
| :--------------------------- | :----------------------------------------------------------- |
| **`poem_eval_script.py`**    | Evaluates **Depth of Explanation**, **Knowledge Accuracy**, and **Instructional Clarity** using LLMs (e.g., Claude 3.5 Sonnet) on extracted scripts. |
| **`poem_eval_keyframes.py`** | Evaluates **Aesthetic Coherence** and **Poetic Imagery Restoration** based on sampled keyframes. |
| **`poem_eval_video.py`**     | Evaluates **Emotional Resonance** using multimodal LLMs (e.g., Gemini 1.5 Pro) on the full video. |
| **`videoscore.py`**          | Calculates **VideoScore** for visual quality and dynamic degree assessment. |
| **`batch_clip_scoring.py`**  | Computes **CLIP scores** for vision-language alignment validation. |
