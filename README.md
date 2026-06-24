# YOLOv8n Custom Detection Model

This repository contains the architecture definition and configurations for a custom-trained **YOLOv8n** model (version `8.0.0.dev0`), optimized for general object detection tasks. 

The accompanying serialized state and parameters provide a complete roadmap of the neural network's architecture, layers, and historical training anchors.

## 🚀 Model Details

* **Base Architecture:** YOLOv8 Nano (`yolov8n.yaml`)
* **Framework:** PyTorch
* **License:** [AGPL-3.0 License](https://ultralytics.com/license)
* **Dataset Format:** COCO (`coco.yaml`)
* **Total Classes:** 80 (Standard COCO Classes, including person, bicycle, car, motorcycle, airplane, etc.)

---

## 🏗️ Neural Network Architecture

The model processes spatial features using a streamlined split-backbone system paired with multi-scale feature fusion heads (C2f & SPPF).

### 1. Backbone
* **Layer 0:** Standard Conv layer for primary spatial downsampling.
* **Layers 1–4:** Sequential feature extractors built upon advanced **Conv** and **C2f** modules.
* **Layer 5 (SPPF):** Spatial Pyramid Pooling Fast (`MaxPool2d` modules with a localized pool size of 5) to consolidate global contextual information.

### 2. Head & Feature Fusion
* Utilizes a combination of **Upsample** (`nearest` mode) and **Concat** layers to construct feature pyramids.
* Employs deep residual **Bottleneck** blocks inside the multi-scale feature pathways.
* **Detect Module:** Final prediction anchors utilizing Distribution Focal Loss (**DFL**) with a calculated `reg_max = 16`.

---

## ⚙️ Hyperparameters & Training Setup

The configuration was compiled utilizing standard `ultralytics` training protocols. Key parameters include:

| Parameter | Value / Setting | Description |
| :--- | :--- | :--- |
| **Optimizer** | `SGD` | Stochastic Gradient Descent |
| **Image Size** | Multi-scale adaptive | Scaled dynamically during training |
| **Initial LR (`lr0`)**| `0.01` | Baseline learning rate |
| **Momentum** | `0.937` | SGD acceleration momentum |
| **Weight Decay** | `0.0005` | L2 Regularization penalty factor |
| **Augmentations** | Mosaic, Mixup, Flip | Advanced data augmentation enabled |
| **Precision** | `FP16` (Half Storage) | Weights stored in half-precision tensors |

---

## 🛠️ Usage Guide

To load, infer, or continue training using this model setup, ensure you have the `ultralytics` package installed.

### Installation
```bash
pip install ultralytics torch
