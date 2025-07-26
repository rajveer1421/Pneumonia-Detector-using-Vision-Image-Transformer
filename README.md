# Pneumonia-Detector-using-Vision-Image-Transformer
🩺 Pneumonia Detection using Vision Transformer (ViT) & Flask
This project is a deep learning-based web application to detect pneumonia from chest X-ray images. I built a custom Vision Transformer model (VisionImageTransformer) from scratch using TensorFlow/Keras, without relying on any pretrained architectures.

Unlike traditional CNNs, the model breaks down images into fixed-size patches and uses multi-head self-attention and transformer blocks to learn spatial patterns. I also incorporated CNN layers to improve feature extraction and hybrid learning.

The trained model is deployed using Flask, enabling users to upload X-ray images through a simple web interface and receive real-time predictions.

✨ Key Highlights
✅ Created Vision Transformer from scratch using Keras (no pretrained models)

📊 Achieved 81% validation accuracy on X-ray dataset

🧠 Learned how self-attention replaces convolutions in ViTs

🌐 Integrated the model into a Flask web app for easy interaction

💡 Gained hands-on experience with deployment, custom class handling, and image preprocessing

⚙️ Tech Stack
TensorFlow / Keras

NumPy

Flask

HTML & CSS

