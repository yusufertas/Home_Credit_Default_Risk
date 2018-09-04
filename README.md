# Home_Credit_Default_Risk

## Executive Summary

  This report details my work to obtain a credit repay probability model for review by Tekkredi. Using various dataframes and a couple of hundred features, I tried to predict the probabilities of each applicant in their ability to repay their loan. The data is comprised of training and test data, where the training data has a target variable named 'TARGET' to indicate whether the user has paid the loan or not. This data is not available in the test data, although it is matched to the system and an ROC AUC score is achieved for a successful submission. There also exists additional datasets from which we obtain critical features. This data is not available in the test data, although it is matched to the system and an ROC AUC score is achieved for a successful submission.

  An ROC AUC curve indicates the tradeoff between true positive rates and false positive rates. We want our model to have higher true positive rates, so we want a higher ROC AUC score which is the area that lies below the mentioned curve.
  
  After trying out many different models which I will discuss below, I opted for an XGBoost model with 50-fold cross validation to obtain our best model, the code of which will be shown in the html file in the repository.I started with using the training and test sets only and doing several variations on the basic model provided by one of the competition kernels:
  
  https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
  
This initial model was fit by Logistic Regression, MLP (Multi-Layer Perceptron) and XGBoost (A gradient boosted random forest model) with 100 features. The submitted results returned a very similar ROC scores of 0.728-0.735 on the test set. I extended this analysis to all the features in the training/test set, where I also provided one-hot encoding (a type of numerical label encoding creating 0-1 vectors) with 50-fold cross validation on the training data, obtaining a score of 0.76796184 (on the training set) with Logistic Regression (this work and many others were not submitted for the test set due to time constraints). 

  The natural extension was to use the other variables in the other dataframes provided. I first got a list of the 100 or so most important features from the previous analysis and then extended it with creating one-hot encoded features. Furthermore, I merged appropriate variables from appropriate additional datasets and built new quality features without modifying the original features in the training/test set (these modifications seem to reduce the ROC scores significantly). As a result of fitting Logistic Regression to this model with 50-fold cross validation, I was able to improve the ROC score to around 0.7691 (on the training set). I also checked that both Logistic Regression and XGBoost worked well with a higher number of features through an extensive ROC score search. Due to time constraints, MLP model was not fit at this stage. 
  
  The final model that was sent for submission (XGBoost with adjusted parameters on about 210 features with 50-fold cross validation) received a ROC score of **0.79006191** on the training set and **...** on the test set. The code in the .md file illustrates the code used to achive this ROC score from the very basic exploratory data analysis steps to final submission of the relevant dataframe.
