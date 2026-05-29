# PoemDirector: Dataset and Evaluation Code

This repository contains the dataset and evaluation scripts for the paper **"PoemDirector: A Multi-Agent Context-Adaptive Instructional Mode Selection Framework for Chinese Classical Poetry Video Generation"**.

## 📂 Dataset

Due to file size limitations, the full video dataset (100 poems × 5 methods) is hosted on Hugging Face.

Please download and unzip **`poem_dataset.zip`** from the link below:

**[Download Dataset via Hugging Face](https://huggingface.co/datasets/userasadfasdf/poem_data/tree/main)**

Once unzipped, the dataset is organized by method:
* **`Our_Method`**: Our proposed method.
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

## Demo Videos

We provide 5 representative demo poems for each method:
`demo1 = Chusai`, `demo2 = Denggao`, `demo3 = Fengqiao Yebo`, `demo4 = Jiangxue`, `demo5 = Liangzhouci`.

| Method | demo1 | demo2 | demo3 | demo4 | demo5 |
| :----- | :--- | :--- | :------- | :--- | :----- |
| `Our_Method` | [▶︎ demo1](generated_video/Our_Method/chusai_demo1.mp4) | [▶︎ demo2](generated_video/Our_Method/denggao_demo2.mp4) | [▶︎ demo3](generated_video/Our_Method/fengqiao_yebo_demo3.mp4) | [▶︎ demo4](generated_video/Our_Method/jiangxue_demo4.mp4) | [▶︎ demo5](generated_video/Our_Method/liangzhouci_demo5.mp4) |
| `HAV` | [▶︎ demo1](generated_video/HAV/chusai_demo1.mp4) | [▶︎ demo2](generated_video/HAV/denggao_demo2.mp4) | [▶︎ demo3](generated_video/HAV/fengqiao_yebo_demo3.mp4) | [▶︎ demo4](generated_video/HAV/jiangxue_demo4.mp4) | [▶︎ demo5](generated_video/HAV/liangzhouci_demo5.mp4) |
| `Mootion` | [▶︎ demo1](generated_video/Mootion/chusai_demo1.mp4) | [▶︎ demo2](generated_video/Mootion/denggao_demo2.mp4) | [▶︎ demo3](generated_video/Mootion/fengqiao_yebo_demo3.mp4) | [▶︎ demo4](generated_video/Mootion/jiangxue_demo4.mp4) | [▶︎ demo5](generated_video/Mootion/liangzhouci_demo5.mp4) |
| `ESV` | [▶︎ demo1](generated_video/ESV/chusai_demo1.mp4) | [▶︎ demo2](generated_video/ESV/denggao_demo2.mp4) | [▶︎ demo3](generated_video/ESV/fengqiao_yebo_demo3.mp4) | [▶︎ demo4](generated_video/ESV/jiangxue_demo4.mp4) | [▶︎ demo5](generated_video/ESV/liangzhouci_demo5.mp4) |
| `PRV` | [▶︎ demo1](generated_video/PRV/chusai_demo1.mp4) | [▶︎ demo2](generated_video/PRV/denggao_demo2.mp4) | [▶︎ demo3](generated_video/PRV/fengqiao_yebo_demo3.mp4) | [▶︎ demo4](generated_video/PRV/jiangxue_demo4.mp4) | [▶︎ demo5](generated_video/PRV/liangzhouci_demo5.mp4) |


https://github.com/user-attachments/assets/80cc6263-a2e5-4c1f-8c58-18bf0ccd89ed

