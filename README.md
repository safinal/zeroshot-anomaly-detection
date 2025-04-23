# ⭐️ Zero-Shot Anomaly Detection Challenge
*Contributors: [Ali Nafisi](https://safinal.github.io/)*

[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🔍 Overview
This repository contains our solution for the Zero-Shot Anomaly Detection Challenge, part of the [**Rayan International AI Contest**](https://ai.rayan.global). The challenge aims to develop a system that can detect anomalies in images using a zero-shot learning approach.

## 🎯 Challenge Objective
The challenge focuses on zero-shot unsupervised anomaly detection, requiring models to operate without any training data from the test distribution. The goal is to detect and localize anomalous regions in test images across both industrial and medical datasets. Unlike conventional approaches that rely on normal samples for training, this zero-shot setting demands effective performance when the only available information about the test-time distribution is implicit in the unlabeled test set itself.


## ⚙️ Constraints
The zero-shot approach imposes significant constraints:
- Models must not be trained on any samples from the test-time distribution (neither normal nor anomalous)
- Models can either be completely training-free or trained only on a specified auxiliary dataset
- Solutions must generalize well across diverse dataset classes (8 industrial and 2 medical classes)
- Each class within the dataset is treated independently
- Performance is evaluated using multiple metrics at both image-level (AUROC, AP, F1) and pixel-level (AUROC, AUPRO, AP, F1)

## 🚀 Our Approach
[must_completed]

## 🏆 Results

Our solution for this Challenge achieved outstanding results. The evaluation metric for this challenge is a weighted average of multiple metrics (image-level AUROC, AP, F1 and pixel-level AUROC, AUPRO, AP, F1), with submissions tested on a private test dataset. Our approach achieved the **highest score**.

The table below presents a summary of the Top 🔟 teams and their respective accuracy scores:

| **Rank** | **Team**                             | **Accuracy (%)** |
|----------|--------------------------------------|------------------|
|🥇        | **No Trust Issues Here (Our Team)**  | **73.16**        |
|🥈        | AIUoK                                | 73.04            |
|🥉        | Tempest                              | 71.61            |
| 4        | Sum of Squared Errors                | 71.55            |
| 5        | Scientific                           | 71.54            |
| 6        | AI Guardians of Trust                | 66.47            |
| 7        | red_serotonin                        | 65.92            |


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
