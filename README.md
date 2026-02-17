# Face Anti-Spoofing Detection System

A deep learning-based face anti-spoofing system built using PyTorch and transfer learning.  
The model detects whether a face image is **Live (Real)** or **Spoof (Printed/Replay Attack)**.

---

##  Overview

This project implements a binary face anti-spoof classifier using a pretrained **ResNet backbone** (18/34/50) with a custom classification head.
The system is trained using supervised learning with data augmentation and validation-based evaluation.

---

## Model Architecture

The model uses transfer learning:

- Backbone: ResNet18 / ResNet34 / ResNet50 (ImageNet pretrained)
- Removed original fully connected layer
- Custom classifier head:
  - Linear → ReLU → Dropout → Linear (1 output logit)
- Loss: BCEWithLogitsLoss

### Why ResNet?
ResNet helps extract:
- Micro-texture differences
- Reflection artifacts
- Illumination inconsistencies
- Screen/paper attack patterns

---

##  Architecture Flow

Input Image (224x224 RGB)  
ResNet Backbone (Feature Extraction)  
Custom Classifier Head  
Output (Live / Spoof)

# RESULT
<p align="center">
  <img src="assets/images/Screenshot.png" width="400"/>
</p>
This can still be improved, as the model takes a little time to detect.
