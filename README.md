# ⭐️ Zero-Shot Anomaly Detection Challenge

[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🔍 Overview
This repository contains our solution for the Zero-Shot Anomaly Detection Challenge, part of the [**Rayan International AI Contest**](https://ai.rayan.global). The challenge aims to develop a system that can detect anomalies in images using a zero-shot learning approach.

## 🎯 Challenge Objective
The challenge focuses on zero-shot unsupervised anomaly detection, requiring models to operate without any training data from the test distribution. The goal is to detect and localize anomalous regions in test images across both industrial and medical datasets. Unlike conventional approaches that rely on normal samples for training, this zero-shot setting demands effective performance when the only available information about the test-time distribution is implicit in the unlabeled test set itself.


## ⚙️ Constraints
* **Zero-Shot Learning Protocol:** The model must strictly adhere to a zero-shot setting, meaning it cannot observe any data—neither normal nor anomalous samples—from the test time distribution during its training phase.
* **Allowed Training Data:** The model can either be completely training-free or utilize a training phase restricted solely to a specific auxiliary dataset. The only permitted auxiliary dataset for this purpose is the entire MVTec-AD dataset.
* **Class Independence:** The problem is structured as a one-class classification (OCC) or one-model-per-category task. Each class within the dataset is treated independently, meaning there is no need to develop a unified model that functions across all classes simultaneously.
* **Execution Environment:** All code is executed within an isolated Docker container that does not have internet access. Consequently, any backbones or pre-trained models must be included in the repository and loaded offline.
* **Hardware and Time Limits:** The submission will be evaluated on a single NVIDIA GeForce RTX 4090 GPU. The total execution time for the inference phase, including evaluation, must not exceed 3 hours.
* **Output Specifications:** All generated anomaly scores (both image-level and pixel-level) must be normalized to a value between 0 and 1. Additionally, pixel-level anomaly score outputs must have a resolution of $224 \times 224$.

## 🚀 Our Approach
To address the zero-shot anomaly detection challenge, we adopted the **MuSc (Mutual Scoring)** framework, which achieves state-of-the-art results by leveraging the observation that normal image patches recur across test samples, whereas anomalous patches are rare. Our solution enhances this framework through specific architectural choices, hyperparameter optimization, and a novel post-processing strategy.

### Core Methodology
Our pipeline consists of three primary components derived from the MuSc framework:
* **Local Neighborhood Aggregation with Multiple Degrees (LNAMD):** Aggregates patch features at different neighborhood sizes to capture anomalies at multiple scales.
* **Mutual Scoring Mechanism (MSM):** Assigns patch-level anomaly scores by measuring how frequently a patch finds similar counterparts within the test set.
* **Re-scoring with Constrained Image-level Neighborhood (RsCIN):** Refines image-level anomaly scores by constructing a constrained neighborhood graph to enforce consistency among images with similar global features.

### Key Implementations & Refinements
* **Backbone Architecture:** We utilized a combined model design, employing **DINOv2** (`dinov2-vitl14`) for segmentation tasks and **ViT** (`ViT-L-14-336`) for classification tasks.
* **Hyperparameter Optimization:** We performed a systematic search over hyperparameters, selecting feature layers `{5, 11, 17, 23}` and specific score configurations that consistently improved performance on both image-level and pixel-level metrics.
* **Post-Processing with Binary Masking:** We identified that the baseline method struggled with background noise (e.g., grass or soil around photovoltaic modules) due to inconsistent textures. To resolve this, we introduced a classical computer vision filter to generate a binary mask that defines a narrow margin around the main object. Pixels outside this mask are set to the minimum anomaly value, significantly improving localization stability.

## 🏆 Results

Our solution for this Challenge achieved outstanding results. The evaluation metric for this challenge is a weighted average of multiple metrics (image-level AUROC, AP, F1 and pixel-level AUROC, AUPRO, AP, F1), with submissions tested on a private test dataset. Our approach achieved the **second highest score**.

The table below presents a summary of the Top 🔟 teams and their respective scores:

| **Rank** | **Team**                             | **Score (%)** |
|----------|--------------------------------------|------------------|
|🥇        | Pileh                                | 74.92            |
|🥈        | **No Trust Issues Here (Our Team)**  | **73.14**        |
|🥉        | AIUoK                                | 72.98            |
| 4        | Tempest                              | 70.88            |
| 5        | AI Guardians of Trust                | 66.29            |
| 6        | red_serotonin                        | 63.51            |
| 7        | CortexAI                             | 62.35            |
| 8        | GGWP                                 | 62.25            |
| 9        | AlphaQ                               | 62.15            |
| 10       | Persistence                          | 61.62            |


## 📄 Technical Report
For a detailed explanation of our methodologies, experiments, and results, please refer to our full [Technical Report](https://arxiv.org/abs/2512.01498).


## 🏃🏻‍♂️‍➡️ Steps to Set Up and Run

Follow these instructions to set up your environment and execute the training pipeline.

### 1. Clone the Repository
```bash
git clone git@github.com:safinal/zeroshot-anomaly-detection.git
cd zeroshot-anomaly-detection
```
### 2. Set Up the Environment
We recommend using a virtual environment to manage dependencies.

Using ```venv```:
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```
Using ```conda```:
```bash
conda create --name zeroshot-anomaly-detection python=3.8 -y
conda activate zeroshot-anomaly-detection
```
### 3. Install Dependencies
Install all required libraries from the ```requirements.txt``` file:
```bash
pip install -r requirements.txt
```
### 4. Run
```bash
python runner.py
```

## 🫶🏻 Acknowledgment
Our solution is highly inspired by the work presented in the paper [MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images](https://arxiv.org/abs/2401.16753), which served as the foundational basis for our implementation. We also extend our gratitude to the computer vision and anomaly detection research community for their invaluable contributions, which have highly paved the way for this solution.

## 🤝🏼 Contributions
We welcome contributions from the community to make this repository better!
