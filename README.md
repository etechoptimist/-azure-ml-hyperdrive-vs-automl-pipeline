# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree program. In this project, I focused on creating and improving an Azure ML pipeline using the Python SDK and a custom Scikit-learn Logistic Regression model. I started by optimizing the model's hyperparameters with HyperDrive. Next, I used Azure AutoML to find the best-performing model for the same dataset. Finally, I compared the results from both methods to assess their performance and measure their effectiveness.

## Summary

### Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on data from a direct marketing campaign conducted by a bank.
 The problem is framed as a classification task with two possible outcomes: "yes" (the customer subscribes) or "no" (the customer does not subscribe). 
 This task emphasizes customer acquisition by targeting potential subscribers. By accurately predicting subscription likelihood, the model can help the bank improve 
 its marketing strategy and optimize resource allocation.


### Results

The best-performing model using HyperDrive was a Logistic Regression with a regularization strength of 1.0 and 150 maximum iterations, achieving an accuracy of 0.917 and an AUC_weighted of 0.932. In comparison, the AutoML model achieved significantly better results, with an AUC_weighted of 0.950 and a range of other metrics indicating superior performance. Given that AUC_weighted was our main metric, the AutoML model proved to be the better choice.

The AutoML model performed better overall, with LightGBM being identified as the best model. It achieved an AUC_weighted of 0.950, surpassing the Logistic Regression model from HyperDrive, which had an AUC_weighted of 0.932. This makes AutoML's LightGBM the superior model for this task based on our primary metric, AUC_weighted.

## Scikit-learn Pipeline

### Pipeline architecture

The pipeline starts with a data preprocessing step, where categorical variables are one-hot encoded, numerical mappings are applied, and the target variable is binarized, preparing the dataset for training. Hyperparameter tuning is conducted using Azure HyperDrive with a Logistic Regression model from Scikit-learn, optimizing parameters such as C (regularization strength) and max_iter (iterations) to maximize the AUC_weighted metric, while a Bandit policy ensures efficient early stopping of underperforming runs.

**What are the benefits of the parameter sampler you chose?**

The RandomParameterSampler is efficient because it explores a large hyperparameter space by randomly selecting combinations, which can uncover good configurations without the exhaustive effort of testing all possibilities. This approach is especially useful when the parameter space is large, as it saves time and computational resources compared to grid search.

**What are the benefits of the early stopping policy you chose?**

The BanditPolicy saves time and resources by stopping poorly performing runs early, allowing the experiment to focus on the most promising hyperparameter configurations.

## AutoML

The AutoML model generated is a classification model optimized for the primary metric AUC_weighted, which is chosen because it evaluates the model's ability to distinguish between classes while handling imbalanced datasets effectively by weighting the AUC for each class by its prevalence. The model performs automatic feature engineering (featurization='auto'), includes model explainability, and uses 5-fold cross-validation for robust evaluation. Additionally, early stopping is enabled to avoid overfitting, and the configuration supports ONNX-compatible models, with voting ensembles included to improve overall performance.

## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

The two models differ in both architecture and performance. The Logistic Regression model from HyperDrive achieved an accuracy of 0.917 and an AUC_weighted of 0.932, while the LightGBM model from AutoML achieved an accuracy of 0.915 and a significantly better AUC_weighted of 0.950. The performance difference likely stems from the architectures: Logistic Regression is a linear model, which may struggle with complex relationships in the data, while LightGBM is a tree-based ensemble model, better suited for capturing non-linear patterns and interactions. The advanced feature engineering and model selection techniques in AutoML further contributed to its superior performance.

## Future work

**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Expanding the search space for hyperparameters allows the model to explore a wider range of configurations, increasing the chances of finding more optimal settings that improve performance. Balancing the dataset by oversampling or generating synthetic data ensures the model is not biased toward majority classes, leading to better predictions for underrepresented outcomes. Additionally, experimenting with advanced models like neural networks can capture complex, non-linear relationships in the data, potentially improving accuracy and overall performance.

## Proof of cluster clean up

**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

![Cluster Deletion](data:image/png;base64,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)