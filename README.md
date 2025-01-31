[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4PocFZrV)

Loris Salsi / 287715@student.pwr.edu.pl

# NIH Chest X-ray Dataset Analysis and Classification

This repository contains a project focused on the classification of chest X-ray images using the Random Sample of NIH Chest X-ray Dataset, a subset of the National Institutes of Health (NIH) Chest X-ray Dataset. The dataset includes 5,606 images labeled for various thoracic diseases, with a particular emphasis on Infiltration as the target class for binary classification. The project explores both TensorFlow/Keras and PyTorch frameworks for preprocessing, training, and evaluating machine learning models.

# Dataset Overview

## Source
Dataset Name: NIH Chest X-ray Dataset Sample
Dataset Size: 5% of the full NIH Chest X-ray Dataset
Number of Images: 5,606 (1024 x 1024 resolution)
Labels File: sample_labels.csv
Link: Kaggle - NIH Chest X-ray Dataset Sample 

## Class Descriptions
This dataset contains 15 classes (14 diseases and one for "No findings"). For this project, Infiltration is used as the target class for binary classification:
Class 1: Images with "Infiltration"
Class 0: All other images

# Objectives

- Perform binary classification to predict whether an X-ray image indicates Infiltration.
- Address dataset imbalance using augmentation and other techniques.
- Compare model performance across frameworks (TensorFlow/Keras and PyTorch).
- Experiment with data augmentation to enhance classification performance.

# Data Management

The dataset used for this project, a random sample of the NIH Chest X-ray Dataset, is not included in the GitHub repository due to its size and licensing restrictions. Instead, the dataset was stored and accessed via Google Drive during the development phase using Google Colab.

If you wish to replicate this project:
- Download the dataset from Kaggle.
- Upload the dataset to your Google Drive.
- Update the dataset paths in the scripts to match your Drive structure.
