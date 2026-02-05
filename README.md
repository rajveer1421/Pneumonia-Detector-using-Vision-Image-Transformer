# 🩺 Pneumonia Detection using Vision Transformer (from Scratch)

This project implements a **Vision Transformer (ViT) from scratch** to detect pneumonia from chest X-ray images.  
Instead of relying on pretrained architectures, the entire transformer pipeline — including **patch embeddings, multi-head self-attention, and transformer encoder blocks** — is implemented manually using TensorFlow/Keras.

The project explores how **self-attention can replace convolutions** for visual representation learning and evaluates transformer-based models on a real-world medical imaging task.

---

## Motivation

Convolutional Neural Networks (CNNs) dominate medical image analysis due to their strong inductive bias for locality.  
Vision Transformers, however, model **global spatial relationships** using self-attention, enabling long-range dependency modeling from early layers.

This project investigates:
- How Vision Transformers perform on medical X-ray images
- The role of self-attention in capturing global image context
- Whether combining CNNs with transformers improves stability on small datasets

---

## Dataset

- Chest X-ray images labeled as **Normal** and **Pneumonia**
- Images are resized, normalized, and augmented before training
- Dataset split into training and validation sets

*(Add dataset link or citation if publicly available)*

---

## Model Architecture

### VisionImageTransformer (Custom Implementation)

The model consists of the following components:

1. **Patch Extraction**
   - Input images are divided into fixed-size patches
   - Each patch is flattened and projected into an embedding space

2. **Positional Encoding**
   - Learnable positional embeddings are added to preserve spatial information

3. **Transformer Encoder Blocks**
   - Custom **Multi-Head Self-Attention**
   - Feed-Forward Neural Networks
   - Residual connections and Layer Normalization

4. **Hybrid CNN–Transformer Design**
   - Optional convolutional layers for early feature extraction
   - Transformer blocks model global dependencies on extracted features

5. **Classification Head**
   - MLP head for binary classification (Pneumonia vs Normal)

---

## Experiments & Results

- Achieved **~81% validation accuracy** on the chest X-ray dataset
- Compared multiple architectures:
  - CNN-only baseline
  - Pure Vision Transformer
  - Hybrid CNN + Vision Transformer

### Observations
- Self-attention captures global spatial patterns in chest X-ray images
- Hybrid CNN–Transformer models showed better stability on limited data
- Vision Transformers benefit from inductive bias when trained from scratch

---

## Limitations

- Performance limited by dataset size
- Training Vision Transformers from scratch is computationally expensive
- No large-scale pretraining used

---

## Future Work

- Pretraining ViT on larger medical imaging datasets
- Exploring self-supervised or contrastive learning methods
- Attention visualization and explainability (Grad-CAM, attention maps)
- Extending to multi-class or multi-label classification

---

## Deployment

The trained model is deployed using **Flask**, enabling:
- Uploading chest X-ray images via a web interface
- Real-time pneumonia prediction

This demonstrates an **end-to-end applied ML pipeline**, from model design to deployment.

---

## Tech Stack

- TensorFlow / Keras
- NumPy
- Flask
- HTML & CSS

---

## Key Takeaways

- Implemented a Vision Transformer **entirely from scratch**
- Gained practical understanding of transformer architectures beyond NLP
- Studied trade-offs between CNN inductive bias and attention-based modeling
- Applied transformers to a real-world medical imaging problem

---
