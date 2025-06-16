
# 🧠 Detection of Choroidal Neovascularization Using Deep Learning

This repository provides a web application built with **Flask** for detecting Choroidal Neovascularization (CNV) in retinal images using multiple **deep learning models**.

---

## 🚀 Features

- Upload retinal OCT(A) images via a web interface  
- Choose from multiple pretrained CNN and transformer models  
- Get binary classification: CNV vs. Normal  
- Easily extensible architecture for new models

---

## ⚙️ Models Included

The following models are integrated into the application:

| Model                | Framework | Status   |
|---------------------|-----------|----------|
| CNN                 | Keras     | ✅ Active |
| ResNet18            | PyTorch   | ✅ Active |
| ResNet50            | Keras     | ✅ Active |
| VGGNet16            | Keras     | ✅ Active |
| VGGNet19            | Keras     | ✅ Active |
| InceptionV3         | Keras     | ✅ Active |
| MobileNetV2         | Keras     | ✅ Active |
| EfficientNetV2L     | Keras     | ✅ Active |
| Vision Transformer  | PyTorch   | 🔲 Optional (commented) |

---



## 🧪 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SahilChukka19/Detection-of-Choroidal-Neovasculariation-Using-Deep-Learning-Models.git
cd Detection-of-Choroidal-Neovasculariation-Using-Deep-Learning-Models
```
### ▶️ Running the Application
```bash
python app.py
```
Visit your app at: http://127.0.0.1:5000/

Upload a retinal image

Select the desired model from the dropdown

Click "Predict" to view classification output

## 📑 Results Interpretation
The models provide binary classification:

CNV: Choroidal Neovascularization present

Normal: No abnormality detected

## 🙋‍♂️ Contributors
### @SahilChukka19 – Core Development, Model Integration
### @Voxovoxo      – Model Intergration, Web Designing
