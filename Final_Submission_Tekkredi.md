
# Prepared by Yusuf Ertas for Review by Tekkredi

## Executive Summary

This report details my work to obtain a credit repay probability model for review by Tekkredi. Using various dataframes and a couple of hundred features, I tried to predict the probabilities of each applicant in their ability to repay their loan. The data is comprised of training and test data, where the training data has a target variable named 'TARGET' to indicate whether the user has paid the loan or not. This data is not available in the test data, although it is matched to the system and an ROC AUC score is achieved for a successful submission. There also exists additional datasets from which we obtain critical features. This data is not available in the test data, although it is matched to the system and an ROC AUC score is achieved for a successful submission.

An ROC AUC curve indicates the tradeoff between true positive rates and false positive rates. We want our model to have higher true positive rates, so we want a higher ROC AUC score which is the area that lies below the mentioned curve.

After trying out many different models which I will discuss below, I opted for an XGBoost model with 50-fold cross validation to obtain our best model, the code of which will be shown in the html file in the repository.I started with using the training and test sets only and doing several variations on the basic model provided by one of the competition kernels:

https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

This initial model was fit by Logistic Regression, MLP (Multi-Layer Perceptron) and XGBoost (A gradient boosted random forest model) with 100 features. The submitted results returned a very similar ROC scores of 0.728-0.735. I extended this analysis to all the features in the training/test set, where I also provided one-hot encoding (a type of numerical label encoding creating 0-1 vectors) with 50-fold cross validation on the training data, obtaining a score of 0.76796184 with Logistic Regression (this work and many others were not submitted for the test set due to time constraints).

The natural extension was to use the other variables in the other dataframes provided. I first got a list of the 100 or so most important features from the previous analysis and then extended it with creating one-hot encoded features. Furthermore, I merged appropriate variables from appropriate additional datasets and built new quality features without modifying the original features in the training/test set (these modifications seem to reduce the ROC scores significantly). As a result of fitting Logistic Regression to this model with 50-fold cross validation, I was able to improve the ROC score to around 0.7691. I also checked that both Logistic Regression and XGBoost worked well with a higher number of features through an extensive ROC score search. Due to time constraints, MLP model was not fit at this stage.

The final model that was sent for submission (XGBoost with adjusted parameters on about 210 features with 50-fold cross validation) received a ROC score of **0.79006191** on the training set and **0.75195** on the test set. The code in the html file illustrates the code used to achive this ROC score from the very basic exploratory data analysis steps to final submission of the relevant dataframe.

## Loading the data and Exploratory Look

In this section, we will load the relevant data and look at its only very basic properties such as how many rows and columns each file has. As can be seen most of the features we want to extract are in the training/test set. For more information on the basics of the data, we use as reference the following two kernels. One look at our data reveals that there are much more rows for the supplementary datasets than the training/test sets indicating a possibility groupby in pandas with mean() and sum() functions. 

* https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
* https://www.kaggle.com/ganeshn88/simple-exploration-of-all-200-variables

We will merge various datasets below with train and test sets appropraitely while using groupby operations. We will investigate this in the next section.


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# Training data
training = pd.read_csv('./Tekkredi/application_train.csv')
print('Training data shape: ', training.shape)
print(training.iloc[:,0:5].head())

# Testing data features
testing = pd.read_csv('./Tekkredi/application_test.csv')
print('Testing data shape: ', testing.shape)
print(testing.iloc[:,0:5].head())

# Bureau data
bureau = pd.read_csv('./Tekkredi/bureau.csv')
print('Bureau data shape: ', bureau.shape)

# previous_application data
previous_application = pd.read_csv('./Tekkredi/previous_application.csv',encoding="latin1")
print('previous_application data shape: ', previous_application.shape)


# credit_card_balance data
credit_card_balance = pd.read_csv('./Tekkredi/credit_card_balance.csv')
print('credit_card_balance data shape: ', credit_card_balance.shape)


# POS_CASH_balance data
POS_CASH_balance = pd.read_csv('./Tekkredi/POS_CASH_balance.csv',encoding="latin1")
print('POS_CASH_balance data shape: ', POS_CASH_balance.shape)
```

    Training data shape:  (307511, 122)
       SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR
    0      100002       1         Cash loans           M            N
    1      100003       0         Cash loans           F            N
    2      100004       0    Revolving loans           M            Y
    3      100006       0         Cash loans           F            N
    4      100007       0         Cash loans           M            N
    Testing data shape:  (48744, 121)
       SK_ID_CURR NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY
    0      100001         Cash loans           F            N               Y
    1      100005         Cash loans           M            N               Y
    2      100013         Cash loans           M            Y               Y
    3      100028         Cash loans           F            N               Y
    4      100038         Cash loans           M            Y               N
    Bureau data shape:  (1716428, 17)
    previous_application data shape:  (1670214, 37)
    credit_card_balance data shape:  (3840312, 23)
    POS_CASH_balance data shape:  (10001358, 8)
    

As mentioned in the executive summary, we bring together two lists (one larger which we will be using in our final submission) decided from our previous analysis using feature_importance_ attribute from the decision trees.


```python
training_list_1 = ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
       'FLAG_MOBIL','FLAG_CONT_MOBILE','FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',
       'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG','YEARS_BUILD_AVG','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR']
testing_list_1 =  ['SK_ID_CURR','NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
       'FLAG_MOBIL','FLAG_CONT_MOBILE','FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',
       'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG','YEARS_BUILD_AVG','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR']
larger_list = ['FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE','YEARS_BUILD_MODE',
       'ELEVATORS_MODE','FLOORSMIN_MODE',
       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
       'BASEMENTAREA_MEDI','YEARS_BUILD_MEDI','FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI','TOTALAREA_MODE',
       'WALLSMATERIAL_MODE']

training_list_large = training_list_1 + larger_list
test_list_large = testing_list_1 + larger_list

train = training[training_list_large]
test = testing[test_list_large]
```

### Converting Factor Variables to Labels (One-Hot Encoding)

In this section, we identify the numeric and factor variables and then convert them to vectors of 1's and 0's similar to obtaining dummy variables in econometric models.


```python
cols = train.columns
num_cols = train._get_numeric_data().columns
factor_list = list(set(cols) - set(num_cols))
print(factor_list)

train[factor_list] = train[factor_list].replace({np.nan:"a"})
test[factor_list] = test[factor_list].replace({np.nan:"a"})
```

    ['WALLSMATERIAL_MODE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'OCCUPATION_TYPE', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE']
    


```python
train = pd.get_dummies(train,columns=factor_list)
test = pd.get_dummies(test,columns=factor_list)

train.drop(['NAME_INCOME_TYPE_Maternity leave', 'NAME_FAMILY_STATUS_Unknown', 'CODE_GENDER_XNA'],axis=1,inplace=True)

print(train.shape)
print(test.shape)
```

    (307511, 181)
    (48744, 180)
    

### Testing the anomalies in the data

Number of children in both the training and test sets are between 0 and 6 as we expect. There is one more correction to the test set below which doesn't affect the number of rows.


```python
train['CNT_CHILDREN'] = train['CNT_CHILDREN'][train['CNT_CHILDREN']<=6]
test['CNT_CHILDREN'] = test['CNT_CHILDREN'][test['CNT_CHILDREN']<=6]
```


```python
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
```

## Merging other datasets with train/test data

In this section, we perform various groupby operations on relevant dataframes and obtain variables which we want to turn into features for use later. We use all the dataframes mentioned at the beginning.


```python
previous_application = previous_application[['SK_ID_CURR','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']]
previous_application.rename(columns={'AMT_CREDIT':'AMT_CREDIT_PA','AMT_ANNUITY':'AMT_ANNUITY_PA','AMT_GOODS_PRICE':
                                     'AMT_GOODS_PRICE_PA' },inplace=True)
```


```python
pa = pd.DataFrame(previous_application.groupby('SK_ID_CURR')[['AMT_CREDIT_PA','AMT_ANNUITY_PA','AMT_GOODS_PRICE_PA']].sum())

train1 = pd.merge(train,pa,how='left',on='SK_ID_CURR')
test1 = pd.merge(test,pa,how='left',on='SK_ID_CURR')

print(train1.shape)
print(test1.shape)
```

    (307511, 184)
    (48744, 183)
    


```python
bureau_cols = ['SK_ID_CURR','DAYS_CREDIT','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT',
              'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']
bureau=bureau[bureau_cols]
bureau.rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_BUREAU'},inplace=True)
bureau_dc = pd.DataFrame(bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].mean())
bureau_percentage = pd.DataFrame(bureau.groupby('SK_ID_CURR')[['AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT',
              'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY_BUREAU']].sum())

bureau_final = pd.merge(bureau_percentage,bureau_dc,left_index=True,right_index=True)
bureau_final = bureau_final.replace({np.nan:0})
train2 = pd.merge(train1, bureau_final, how='left',left_on='SK_ID_CURR',right_index=True) 
test2 = pd.merge(test1, bureau_final,how='left',left_on='SK_ID_CURR',right_index=True)

print(train2.shape)
print(test2.shape)
```

    (307511, 191)
    (48744, 190)
    


```python
train2['AMT_DAYS_CREDIT'] = train2['AMT_CREDIT'].div((-2)*train2['DAYS_CREDIT']+2)
test2['AMT_DAYS_CREDIT'] = test2['AMT_CREDIT'].div((-2)*test2['DAYS_CREDIT']+2)

train2['AMT_CREDIT_MAX_OVERDUE_RATIO'] = train2['AMT_CREDIT_MAX_OVERDUE'].div(train2['AMT_CREDIT'])
train2['AMT_CREDIT_SUM_RATIO'] = train2['AMT_CREDIT_SUM'].div(train2['AMT_CREDIT'])
train2['AMT_CREDIT_SUM_DEBT_RATIO'] = train2['AMT_CREDIT_SUM_DEBT'].div(train2['AMT_CREDIT'])
train2['AMT_CREDIT_SUM_LIMIT_RATIO'] = train2['AMT_CREDIT_SUM_LIMIT'].div(train2['AMT_CREDIT'])
train2['AMT_CREDIT_SUM_OVERDUE_RATIO'] = train2['AMT_CREDIT_SUM_OVERDUE'].div(train2['AMT_CREDIT'])

test2['AMT_CREDIT_MAX_OVERDUE_RATIO'] = test2['AMT_CREDIT_MAX_OVERDUE'].div(test2['AMT_CREDIT'])
test2['AMT_CREDIT_SUM_RATIO'] = test2['AMT_CREDIT_SUM'].div(test2['AMT_CREDIT'])
test2['AMT_CREDIT_SUM_DEBT_RATIO'] = test2['AMT_CREDIT_SUM_DEBT'].div(test2['AMT_CREDIT'])
test2['AMT_CREDIT_SUM_LIMIT_RATIO'] = test2['AMT_CREDIT_SUM_LIMIT'].div(test2['AMT_CREDIT'])
test2['AMT_CREDIT_SUM_OVERDUE_RATIO'] = test2['AMT_CREDIT_SUM_OVERDUE'].div(test2['AMT_CREDIT'])
                          

print(train2.shape)
print(test2.shape)
```

    (307511, 197)
    (48744, 196)
    


```python
POS_CASH_balance = POS_CASH_balance[['SK_ID_CURR','SK_DPD']]
POS_CASH_balance.rename(columns={'SK_DPD':'SK_DPD_CASH'},inplace=True)
POS_CASH_balance = pd.DataFrame(POS_CASH_balance.groupby('SK_ID_CURR')['SK_DPD_CASH'].mean())
credit_card_balance = credit_card_balance[['SK_ID_CURR','SK_DPD','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL',
                                          'AMT_DRAWINGS_CURRENT','AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
                                          'AMT_TOTAL_RECEIVABLE']]
credit_card_balance = pd.DataFrame(credit_card_balance.groupby('SK_ID_CURR')[['SK_DPD','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL',
                                          'AMT_DRAWINGS_CURRENT','AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
                                          'AMT_TOTAL_RECEIVABLE']].mean())

pos_cc = pd.merge(POS_CASH_balance,credit_card_balance,on='SK_ID_CURR')

train3 = pd.merge(train2,pos_cc,how='left',on='SK_ID_CURR')
test3 = pd.merge(test2,pos_cc,how='left',on='SK_ID_CURR')

train3['AMT_BALANCE_RATIO'] = train3['AMT_BALANCE'].div(train3['AMT_CREDIT'])
train3['AMT_CREDIT_LIMIT_ACTUAL_RATIO'] = train3['AMT_CREDIT_LIMIT_ACTUAL'].div(train3['AMT_CREDIT'])
train3['AMT_DRAWINGS_CURRENT_RATIO'] = train3['AMT_DRAWINGS_CURRENT'].div(train3['AMT_CREDIT'])
train3['AMT_PAYMENT_CURRENT_RATIO'] = train3['AMT_PAYMENT_CURRENT'].div(train3['AMT_CREDIT'])
train3['AMT_PAYMENT_TOTAL_CURRENT_RATIO'] = train3['AMT_PAYMENT_TOTAL_CURRENT'].div(train3['AMT_CREDIT'])
train3['AMT_TOTAL_RECEIVABLE_RATIO'] = train3['AMT_TOTAL_RECEIVABLE'].div(train3['AMT_CREDIT'])

test3['AMT_BALANCE_RATIO'] = test3['AMT_BALANCE'].div(test3['AMT_CREDIT'])
test3['AMT_CREDIT_LIMIT_ACTUAL_RATIO'] = test3['AMT_CREDIT_LIMIT_ACTUAL'].div(test3['AMT_CREDIT'])
test3['AMT_DRAWINGS_CURRENT_RATIO'] = test3['AMT_DRAWINGS_CURRENT'].div(test3['AMT_CREDIT'])
test3['AMT_PAYMENT_CURRENT_RATIO'] = test3['AMT_PAYMENT_CURRENT'].div(test3['AMT_CREDIT'])
test3['AMT_PAYMENT_TOTAL_CURRENT_RATIO'] = test3['AMT_PAYMENT_TOTAL_CURRENT'].div(test3['AMT_CREDIT'])
test3['AMT_TOTAL_RECEIVABLE_RATIO'] = test3['AMT_TOTAL_RECEIVABLE'].div(test3['AMT_CREDIT'])
```


```python
train3['Overdue_Days'] = (train3['AMT_CREDIT_SUM_OVERDUE'].div(train3['AMT_CREDIT'])).multiply((train3['SK_DPD_CASH']+train3['SK_DPD_CASH'])/2+1)
test3['Overdue_Days'] = (train3['AMT_CREDIT_SUM_OVERDUE'].div(train3['AMT_CREDIT'])).multiply((train3['SK_DPD_CASH']+train3['SK_DPD_CASH'])/2+1)
```

## Min-Max Scaling and Imputing NaN's

In this section, we normalize the numeric variables through a method called min-max scaling in scikit-learn. This transforms all the numeric variables to values between 0 and 1. In addition to this replacing NaN's with 0's or negative numbers at the earlier stages lead to reduced ROC scores, therefore we impute them using the mean. The idea is that many of the features contain variables that values of 0 and 1. Imputing with respect to the median may skew our results where such data contains many more 1's than 0's and vice versa.


```python
from sklearn.preprocessing import MinMaxScaler, Imputer
imputer = Imputer(strategy = 'mean')

y = train3['TARGET']

if 'TARGET' in train3.columns:
    trainfinal = train3.drop(columns = ['TARGET'])
else:
    trainfinal = train3.copy() 

testfinal = test3

# Fit on the training data
imputer.fit(trainfinal)

# Transform both training and testing data
train = imputer.transform(trainfinal)
test = imputer.transform(testfinal)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
print(train.shape)
```

    (307511, 210)
    

### Correlations

Here, we calculate the correlation of each feature in the model with the target variable to see whether any of them exceeds the absolute value of 0.9 which counts as very high correlation. It turns out that none of the features exceeds this correlation value and they are suitable for use in our final model.


```python
correlations = train3.corr()['TARGET'].sort_values(ascending=False)
print(correlations.head(6))
print(correlations.tail(6))
```

    TARGET                  1.000000
    DAYS_CREDIT             0.089729
    AMT_BALANCE             0.087425
    AMT_TOTAL_RECEIVABLE    0.086737
    DAYS_BIRTH              0.078239
    AMT_BALANCE_RATIO       0.071300
    Name: TARGET, dtype: float64
    NAME_INCOME_TYPE_Pensioner             -0.046209
    CODE_GENDER_F                          -0.054704
    NAME_EDUCATION_TYPE_Higher education   -0.056593
    EXT_SOURCE_1                           -0.155317
    EXT_SOURCE_2                           -0.160472
    EXT_SOURCE_3                           -0.178919
    Name: TARGET, dtype: float64
    

## Final Model (XGBoost, 50-fold cross validation, 210 features)

In the final model, we use the XGBoost model which is an adaptation of gradient boosted random forest (a collection of decision trees) where each tree corrects the error of the trees before itself. We use all of the 210 features since as mentioned in the executive summary, we perform a thorough search on the appropriate number of features for our model. 


```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


xgb = XGBClassifier(learning_rate=0.1,max_depth=5,min_child_weight=1,
                    gamma=0,subsample=0.8,objective= 'binary:logistic',nthread=4)
xgb.fit(train,y)
y_proba = xgb.predict_proba(test)[:,1]

```


```python
# Submission dataframe
submit = test2[['SK_ID_CURR']]
submit['TARGET'] = y_proba
```


```python
# Save the submission to a csv file
submit.to_csv('final_sub1.csv', index = False)
```
