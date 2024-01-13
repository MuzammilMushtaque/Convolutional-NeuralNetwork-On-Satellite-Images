# Satellite Image Classification using Multi-layer Convolutional Neural Network

## Overview

This project focuses on implementing a Multi-layer Convolutional Neural Network (CNN) model for the classification of satellite images into distinct categories: cloudy, desert, water, and green_area.

## Dataset

The dataset comprises a total of 5631 images, evenly distributed among classes, except for the desert class, which has 1131 images. The dataset is partitioned into training (85%), validation (10%), and testing (5%) sets.

## Machine Learning Evaluation

Initially, various standard machine learning techniques were explored to address the image classification problem. Notably, the RandomForestClassifier exhibited an impressive 95% accuracy with minimal computational time. However, to achieve a higher accuracy of 97%, the implementation of neural networks became crucial.

## Deep Learning Experiment

The deep learning experiment involved the creation of a multi-layer convolutional model with 16 output channels (named 'model'). The model was trained and validated using dedicated functions such as `loss_batch`, `evaluate`, and `fit`. The trained model, saved as 'DL_2.pth,' was then applied to the test dataset.

## Hyperparameter Tuning

Several hyperparameters significantly impacted and improved the accuracy of the deep learning model to 97%. These key factors include:
- Data normalization and augmentation
- Adam optimizer
- Optimal small learning rate (0.001)
- Reduced batch size (20)

## Application and Significance

This project holds particular relevance for remote sensing departments dealing with vast amounts of satellite imagery. The deep learning model plays a pivotal role in accurately distinguishing diverse Earth features, achieving a remarkable accuracy of 97%.

## Future Work

To enhance the given model further, future work may involve the implementation of advanced techniques such as Residual Connection or Batch Normalization. These additions could contribute to even more robust and accurate classification results.