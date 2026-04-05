# AISE Phase 2: Flood Segmentation via ResNet50-Unet

This repository contains the implementation for Theme 1 (Flood Segmentation) of the ANRF AISE Hackathon Phase 2. The solution utilizes a Deep Learning approach to segment flooded areas from satellite imagery, specifically optimized to handle class imbalance and high-resolution spatial features.

Key Features & Strategies
To achieve a competitive mIoU (>0.30), the following strategies were implemented:
* Backbone: ResNet50 encoder with ImageNet pre-trained weights for robust feature extraction.
* Input Resolution: Standardized 512x512 processing to balance local detail with global context.
* Loss Function: A hybrid Focal Loss + Dice Loss (50/50 split). Focal Loss was critical in forcing the model to learn the minority "flood" class, which is often ignored by standard Cross-Entropy.
* Normalization: Transitioned from per-image normalization to Global Scaling (0-1) to preserve spectral intensity differences.
* Post-Processing: Applied Test-Time Augmentation (TTA) with horizontal/vertical flips and morphological filtering to remove noise and disconnected components.

## 📁 Project Structure
├── models/             # Directory for model checkpoints (.pth)
├── src/                # Source code (Dataset, Model, Training)
├── LICENSE             # ANRF Open License
└── README.md           # Project Documentation
