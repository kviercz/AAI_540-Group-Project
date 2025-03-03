# AAI_540-Group-Project

![Copy of Copy of MLOpsDesign drawio](https://github.com/user-attachments/assets/aa8a5212-ebdd-424a-9175-7d881875c5e0)

# Emotion Classification Using FER-2013  

## Overview  
This project focuses on building a machine learning model to classify human emotions based on facial expressions. Using the **FER-2013 dataset**, we developed a **Convolutional Neural Network (CNN)** to predict one of seven emotion categories: **happiness, sadness, anger, surprise, disgust, fear, and neutral**. The goal is to explore deep learning techniques for emotion recognition while addressing dataset challenges such as class imbalance.  

## Project Details  
- **Dataset:** [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Model Architecture:** CNN with ReLU activation, categorical crossentropy loss  
- **Transfer Learning Models Considered:** ResNet, EfficientNet  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  
- **Final Model Accuracy:** **64.14%**  
- **Deployment:** **AWS SageMaker (Batch Inference via S3 & Lambda Triggers)**  

## Key Features  
**Preprocessing Pipeline:** Face detection, grayscale conversion, and image resizing (CascadeClassifier).  
**Data Augmentation:** Image flipping and minor transformations to reduce overfitting.  
**Automated CI/CD Workflow:** GitHub Actions for training validation and SageMaker CI/CD for deployment.  
**Model Tracking:** SageMaker Model Store for versioning and batch inference monitoring.  

## Repository Structure  
- **`/`** – Jupyter notebooks for training, evaluation, deployment, pipeline, and monitoring  
- **`lambda_functions/`** – Data preprocessing, lambda, and inference scripts  

## Deployment Process  
1. **Training:** The model is trained locally using **Google Colab (GPU)** due to AWS compute constraints.  
2. **Validation & CI/CD:** If validation accuracy is **above 50%**, GitHub Actions uploads the model as an artifact.  
3. **AWS SageMaker Deployment:**  
   - Model stored in **S3** and registered in **SageMaker Model Store**.  
   - **Inference:** S3 triggers Lambda to preprocess images and send them for predictions.  
   - Predictions are stored in S3 for evaluation.
4. **Model Quality Monitoring:** Scheduled monitoring runs periodically, logging performance metrics (accuracy, precision, recall) into AWS CloudWatch for analysis and alerting.

## Future Enhancements  
**Improve Model Performance:** Address dataset imbalance with techniques like **GANs or synthetic data augmentation**.  

## Authors  
**Project Team Group 5** – *University of San Diego*  
- **Dan Arday**  
- **Gabriel Colon**  
- **Kim Vierczhalek**  
