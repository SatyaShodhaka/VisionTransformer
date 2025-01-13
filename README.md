# Eyes in the Wild: Leveraging Vision Transformers for Animal Detection in Camera Trap Images

This project explores the application of **Vision Transformers (ViTs)**, specifically DETR (Detection Transformer), for detecting and classifying wildlife in camera trap images. It utilizes the **Caltech Camera Traps (CCT-20)** dataset and enhances detection capabilities by employing **Conditional Wasserstein GANs (WGANs)** to translate infrared images to color images. The project contributes to wildlife conservation by providing a robust system for animal detection, addressing challenges like poor illumination, occlusion, and motion blur.

---

## Objectives
1. Implement the **DETR Vision Transformer** for object detection and instance segmentation tasks on the CCT-20 dataset.
2. Develop a **Conditional Wasserstein GAN** to colorize infrared images and evaluate its impact on detection accuracy.
3. Address class imbalances and evaluate DETR's performance across cis- and trans-locations in the dataset.
4. Compare the effectiveness of Vision Transformers with traditional computer vision models like CNNs.

---

## Dataset
We use the **Caltech Camera Traps-20 (CCT-20)** dataset, a subset of the Caltech Camera Traps dataset, which contains:
- **57,868 images** across 20 camera locations.
- **15 species classes** (e.g., deer, bobcats, coyotes) and an empty class.

### Challenges:
- **Class imbalance**: Uneven distribution of images across classes.
- **Environmental factors**: Poor illumination, occlusion, motion blur, and seasonal variations.
- **Small regions of interest (ROI)**: Many images have animals occupying a minor portion of the frame.

---

## Methodology

### 1. Pix2Pix Translation GAN (WGAN)
Infrared images in the dataset were colorized using a Conditional Wasserstein GAN. Key components:

#### ResUNet Generator:
- A U-Net architecture with **residual blocks** and **skip connections**.
- Downsampling layers with max-pooling and upsampling layers with feature concatenation.

#### PatchGAN Discriminator:
- CNN-based discriminator with **instance normalization** and **adaptive average pooling**.
- Uses Wasserstein loss to stabilize training.

**Training Details:**
- **Input**: 7,000 infrared and corresponding color images resized to 224x224.
- **Hardware**: Trained for 100 epochs on a Kaggle P100 GPU.

### Results:
- Generated images were evaluated using **PSNR** and **SSIM** metrics.
- WGAN-enhanced colorized images showed a notable improvement in detection accuracy.

---

### 2. DETR (Detection Transformer)
DETR integrates a **transformer encoder-decoder architecture** with a CNN backbone (ResNet-50) for object detection.

#### Key Features:
- **CNN Backbone**: Extracts image features for processing by the transformer.
- **Transformer Encoder-Decoder**: Captures long-range dependencies and global context.
- **Bipartite Matching Loss**: Matches predictions with ground truth using the Hungarian algorithm.

#### Training Setup:
- Learning rate: **1e-4** (model), **1e-5** (backbone).
- **Data Augmentation**: Horizontal flipping for underrepresented classes.
- Trained on NVIDIA L4 GPUs via Google Colab Pro.

#### Handling Class Imbalances:
Augmented the dataset by horizontally flipping images of underrepresented classes, resulting in a more balanced training set.

---

## Conclusion and Future Work

### Key Findings:
1. DETR demonstrates robust performance for animal detection, particularly on larger species.
2. Incorporating WGANs to preprocess infrared images improves detection accuracy by up to 7%.
3. DETR performs well in transfer learning, achieving comparable accuracy across cis- and trans-locations.

### Future Work:
1. Implement additional preprocessing techniques to address occlusion and motion blur.
2. Extend training epochs to further improve accuracy.
3. Explore other Vision Transformer models like **OSAT** for enhanced performance.

---

## File Structure
```
.
├── data/                             # Dataset folder
├── eccv_18_annotation_files/         # Annotation files
│   ├── cis_test_annotations.json
│   ├── cis_val_annotations.json
│   ├── train_annotations.json
│   ├── trans_test_annotations.json
│   └── trans_val_annotations.json
├── DETR.ipynb                        # Notebook for DETR implementation
├── Transformer.ipynb                 # Notebook for Vision Transformer experiments
├── eccv_18_annotations.tar.gz        # Compressed dataset annotations
└── README.md                         # Project documentation
```

---
## References
1. [Caltech Camera Traps Dataset](https://beery.cc/publications)
2. [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)
3. [Pix2Pix GAN](https://github.com/phillipi/pix2pix)


