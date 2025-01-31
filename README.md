# **NIH Chest X-ray Dataset Analysis and Classification**

This repository contains a project focused on the classification of chest X-ray images using a subset of the **National Institutes of Health (NIH) Chest X-ray Dataset**. The primary objective is **binary classification** to detect the presence of **Infiltration** using both **TensorFlow/Keras** and **PyTorch** frameworks.

## **Dataset Overview**

### **Source**
- **Dataset Name**: NIH Chest X-ray Dataset Sample  
- **Dataset Size**: **112,120 images** (full dataset)  
- **Resolution**: 1024×1024  
- **Labels File**: `sample_labels.csv`  
- **Link**: [Kaggle - NIH Chest X-ray Dataset Sample](https://www.kaggle.com/)  

### **Classes**
The dataset contains **15 classes** (14 diseases + 1 "No findings" class). In this project, we perform **binary classification**:  
- **Class 1**: X-ray images labeled with **"Infiltration"**  
- **Class 0**: All other images  

## **Objectives**
- Perform **binary classification** to predict whether an X-ray image indicates **Infiltration**.  
- Address dataset imbalance using **data augmentation**.  
- Compare model performance between **TensorFlow/Keras** and **PyTorch**.  
- Optimize classification metrics such as **precision, recall, and F1-score**.  

## **Model Architecture**

### **TensorFlow/Keras**
- **Base Model**: **MobileNetV2** (pre-trained on ImageNet, frozen weights).  
- **Feature extractor** includes:  
  - `GlobalAveragePooling2D` or `GlobalMaxPooling2D`.  
  - Dense layer with **64 units (ReLU activation)**.  
  - Output layer with **1 unit (Sigmoid activation)**.  
- **Optimizer**: Adam (`lr=0.001`).  
- **Loss Function**: Binary Crossentropy.  
- **Metrics**: Accuracy, Precision, Recall, F1-score.  

### **PyTorch**
- **Base Model**: **MobileNetV2** (pre-trained on ImageNet, modified classification head).  
- **Classifier replaced** with a **single logit output for binary classification**.  
- **Loss Function**: BCEWithLogitsLoss.  
- **Optimizer**: Adam (`lr=0.001`).  

## **Data Management**
The dataset is **not included** in this repository due to its size and licensing restrictions. To replicate this project:  
1. Download the dataset from Kaggle.  
2. Upload it to **Google Drive**.  
3. Update the dataset paths in the scripts to match your Drive structure.  

Model training was conducted on **Google Colab** with dataset access via Google Drive.

## **Results and Performance**
Final model accuracy ranges between **66-72%**, depending on the framework and applied optimizations.

The **confusion matrix** and **ROC curve analysis** showed improvements in detecting the minority class after applying data augmentation.

## **Conclusions**
This project demonstrates a comparison between **TensorFlow/Keras** and **PyTorch** for deep learning-based **medical image classification**. Future improvements may include:  
- **Advanced transfer learning techniques**.  
- **Better dataset balancing strategies**.  
- **Hyperparameter tuning for further performance gains**.  
