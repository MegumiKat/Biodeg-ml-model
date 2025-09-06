# Biodegradability Prediction using Models

## Explanation for Code

This project aims to predict the biodegradability of various polymer compositions based on their chemical components using multiple machine learning regression models. 

The goal is to forecast biodegradability results (e.g., from 302B, 301F 30D, or 301F 60D tests) based on monomer composition data using different approaches, and evaluate their performance using code execution results.

### Files

- **clustering.py** : This script classifies the biodegradability level of polymer samples (Low / Medium / High) based on their chemical composition using multiple machine learning models with cross-validation and final prediction output.
- **final_verison_biodeg_modelling.py** : This script builds and compares multiple regression models—including Random Forest, Linear Regression, and others—to predict polymer biodegradability (302B, 301F30D, or 301F60D) based on 18-component chemical compositions, supporting both batch evaluation and manual input predictions.
- **generate_table.py** : This script visualizes the non-zero percentage range (min–max) and mean values of 18 polymer ingredients to highlight their distribution across samples.
- **redo.py** : This script loads and preprocesses polymer biodegradability data, applies multiple regression models to predict selected biodegradation test results (e.g., 302B, 301F30D, 301F60D), evaluates their performance using R² and cross-validation, visualizes model comparison, and performs PCA for 2D projection of feature space.

### Dataset

The dataset is stored in Excel file : *Final version-Biodeg data summary sheet.xlsx*

Columns include monomer composition percentages like CHDA, EG, SSIA, PET, etc.

Target biodegradability outputs:

- 302B test result
- 301F test result 30D
- 301F test result 60D

### Models

- Random Forest Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Multi-layer Perceptron (MLP) Regressor
- Gaussian Process Regressor

- PCA + Random Forest for dimensionality reduction & visualization
- 5-Fold Cross Validation comparison for each model’s performance

### How to Run

```python
# Step 1: Install dependencies (if not already done)
pip install -r requirements.txt

# Step 2: Run the script
python your_script_name.py
```

The script will prompt you to choose a biodegradability target:

```python
Which biodegradability do you want to predict ? 302B, 301F30D or 301F60D
```

You can input one of them.

Then it will :

- Preprocess the data
- Train multiple regression models
- Display accuracy (R²) for each model
- Perform cross-validation comparisons
- Show PCA-based visualization (optional)

## Example Output && Analysis (302B)

We only input 302B and the test data is
`[36.3, 45.45, 4.54, 0, 4.54, 0, 0, 9.09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

We use mean $R^2$ to evaluate models. 

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

- \( SS_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \): Residual sum of squares  
- \( SS_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \): Total sum of squares

### **clustering.py**

Predict in group.

#### Manual Grouping

```sh
Which biodegradability do you want to predict ? 302B, 301F30D or 301F60D
302b
302B test result
High      104
Medium     44
Low         2
Name: count, dtype: int64

========== Cross-validation accuracy (5-Fold) ==========

Random Forest        | Mean Accuracy: 0.8000 | Std: 0.0422
Logistic Regression  | Mean Accuracy: 0.7200 | Std: 0.0653
SVC                  | Mean Accuracy: 0.7667 | Std: 0.0298
KNN                  | Mean Accuracy: 0.7133 | Std: 0.0452
Decision Tree        | Mean Accuracy: 0.7533 | Std: 0.0581
Gradient Boosting    | Mean Accuracy: 0.7800 | Std: 0.0581
MLP                  | Mean Accuracy: 0.8000 | Std: 0.0422
Gaussian Process     | Mean Accuracy: 0.7533 | Std: 0.0340

========== Final model training + prediction (using RandomForest) ==========

Classification Report:
              precision    recall  f1-score   support
        High       0.93      0.90      0.92        31
      Medium       0.62      0.71      0.67         7
    accuracy                           0.87        38
   macro avg       0.78      0.81      0.79        38
weighted avg       0.88      0.87      0.87        38

The predicted degradation level of the sample is: High
```

Since manual grouping results in a large disparity in the number of data in each group, we use `qcut` to automatically group.

#### Auto Grouping

##### Test 1

```sh
302B test result
Low       50
Medium    50
High      50
Name: count, dtype: int64
cutpoints:
Group 1: 0.2600 ~ 0.6267
Group 2: 0.6267 ~ 0.8633
Group 3: 0.8633 ~ 1.0000

========== Cross-validation accuracy (5-Fold) ==========

Random Forest        | Mean Accuracy: 0.6800 | Std: 0.0618
Logistic Regression  | Mean Accuracy: 0.6000 | Std: 0.1211
SVC                  | Mean Accuracy: 0.6000 | Std: 0.1054
KNN                  | Mean Accuracy: 0.5667 | Std: 0.0558
Decision Tree        | Mean Accuracy: 0.6667 | Std: 0.0471
Gradient Boosting    | Mean Accuracy: 0.6600 | Std: 0.0533
MLP                  | Mean Accuracy: 0.6267 | Std: 0.0389
Gaussian Process     | Mean Accuracy: 0.5933 | Std: 0.0533
```

##### Test 2

```sh
302B test result
Low       50
Medium    50
High      50
Name: count, dtype: int64
cutpoints:
Group 1: 0.2600 ~ 0.6267
Group 2: 0.6267 ~ 0.8633
Group 3: 0.8633 ~ 1.0000

========== Cross-validation accuracy (5-Fold) ==========

Random Forest        | Mean Accuracy: 0.6733 | Std: 0.0573
Logistic Regression  | Mean Accuracy: 0.5933 | Std: 0.1289
SVC                  | Mean Accuracy: 0.6000 | Std: 0.1054
KNN                  | Mean Accuracy: 0.5667 | Std: 0.0558
Decision Tree        | Mean Accuracy: 0.6667 | Std: 0.0471
Gradient Boosting    | Mean Accuracy: 0.6533 | Std: 0.0542
MLP                  | Mean Accuracy: 0.6267 | Std: 0.0490
Gaussian Process     | Mean Accuracy: 0.5933 | Std: 0.0533
```

##### Test 3

```sh
302B test result
Low       50
Medium    50
High      50
Name: count, dtype: int64
cutpoints:
Group 1: 0.2600 ~ 0.6267
Group 2: 0.6267 ~ 0.8633
Group 3: 0.8633 ~ 1.0000

========== Cross-validation accuracy (5-Fold) ==========

Random Forest        | Mean Accuracy: 0.6933 | Std: 0.0389
Logistic Regression  | Mean Accuracy: 0.6000 | Std: 0.1211
SVC                  | Mean Accuracy: 0.6000 | Std: 0.1054
KNN                  | Mean Accuracy: 0.5667 | Std: 0.0558
Decision Tree        | Mean Accuracy: 0.6733 | Std: 0.0573
Gradient Boosting    | Mean Accuracy: 0.6600 | Std: 0.0573
MLP                  | Mean Accuracy: 0.6333 | Std: 0.0632
Gaussian Process     | Mean Accuracy: 0.5933 | Std: 0.0533
```

#### Result

According to the results of multiple runs above, it is obvious that RF is more applicable than other methods. Then we use the RF to predict.

```sh
========== Final model training + prediction (using RandomForest) ==========

Classification Report:
              precision    recall  f1-score   support
        High       0.80      0.86      0.83        14
         Low       0.67      0.86      0.75         7
      Medium       0.86      0.71      0.77        17

    accuracy                           0.79        38
   macro avg       0.77      0.81      0.78        38
weighted avg       0.80      0.79      0.79        38

The predicted degradation level of the sample is: High
```

### **final_verison_biodeg_modelling.py**

Predict for specific value

#### Test 4

```sh
size of train set: (112, 18)
size of test set: (38, 18)
accuracy of the tree is 43.68512251777161
accuracy of the linear regression is -68.34472330419447
accuracy of the ridge regression is -68.32994206764333
accuracy of the lasso regression is -2.3860405300103027
accuracy of the elastic net regression is 7.504143707158262
accuracy of the svr is 40.821513439552845
accuracy of the knn is 19.25682767576229
accuracy of the decision tree is 35.89109462402481
accuracy of the gradient boosting is 31.824377273054527
accuracy of the mlp is -605.7911899365549
accuracy of the gaussian process is -412.8937864605236
accuracy of the Random Forest after PCA (2D) is 14.720817592371272

 =================== 10-Fold Cross Validation R² Comparison: ===================

Random Forest        | Mean R²: 0.0229 | Std: 0.5406
Linear Regression    | Mean R²: -6.0661 | Std: 11.6136
Ridge                | Mean R²: -6.0181 | Std: 11.5731
Lasso                | Mean R²: -0.3566 | Std: 0.5528
ElasticNet           | Mean R²: -0.4450 | Std: 0.8496
SVR                  | Mean R²: -0.1229 | Std: 0.7309
KNN                  | Mean R²: -0.1923 | Std: 0.6211
Decision Tree        | Mean R²: -0.1063 | Std: 0.5095
Gradient Boosting    | Mean R²: -0.1047 | Std: 0.5735
MLP                  | Mean R²: -117.0823 | Std: 222.8258
Gaussian Process     | Mean R²: -6.8170 | Std: 6.2621
et voila
[87.05734244]
```

From test 4 we can see the accuracy of the RF is better than others

#### Test 5

```sh
size of train set: (112, 18)
size of test set: (38, 18)
accuracy of the tree is 46.638563321653116
accuracy of the linear regression is -68.34472330419447
accuracy of the ridge regression is -68.32994206764333
accuracy of the lasso regression is -2.3860405300103027
accuracy of the elastic net regression is 7.504143707158262
accuracy of the svr is 40.821513439552845
accuracy of the knn is 19.25682767576229
accuracy of the decision tree is 35.89109462402481
accuracy of the gradient boosting is 32.80904429674
accuracy of the mlp is -520.7788680751456
accuracy of the gaussian process is -412.8937864605236
accuracy of the Random Forest after PCA (2D) is 11.83649944132621

 =================== 10-Fold Cross Validation R² Comparison: ===================

Random Forest        | Mean R²: 0.0711 | Std: 0.4789
Linear Regression    | Mean R²: -6.0661 | Std: 11.6136
Ridge                | Mean R²: -6.0181 | Std: 11.5731
Lasso                | Mean R²: -0.3566 | Std: 0.5528
ElasticNet           | Mean R²: -0.4450 | Std: 0.8496
SVR                  | Mean R²: -0.1229 | Std: 0.7309
KNN                  | Mean R²: -0.1923 | Std: 0.6211
Decision Tree        | Mean R²: -0.0231 | Std: 0.4090
Gradient Boosting    | Mean R²: -0.1597 | Std: 0.5965
MLP                  | Mean R²: -150.2467 | Std: 164.5842
Gaussian Process     | Mean R²: -6.8170 | Std: 6.2621
et voila
[86.98209425]
```

From test 5 we can see the accuracy of the RF is better than others.

#### Test 6

```sh
size of train set: (112, 18)
size of test set: (38, 18)
accuracy of the tree is 46.066327398753735
accuracy of the linear regression is -68.34472330419447
accuracy of the ridge regression is -68.32994206764333
accuracy of the lasso regression is -2.3860405300103027
accuracy of the elastic net regression is 7.504143707158262
accuracy of the svr is 40.821513439552845
accuracy of the knn is 19.25682767576229
accuracy of the decision tree is 48.10914669452053
accuracy of the gradient boosting is 30.891927464141645
accuracy of the mlp is -6104.05353796462
accuracy of the gaussian process is -412.8937864605236
accuracy of the Random Forest after PCA (2D) is 12.098476118469215

 =================== 10-Fold Cross Validation R² Comparison: ===================

Random Forest        | Mean R²: 0.0414 | Std: 0.4949
Linear Regression    | Mean R²: -6.0661 | Std: 11.6136
Ridge                | Mean R²: -6.0181 | Std: 11.5731
Lasso                | Mean R²: -0.3566 | Std: 0.5528
ElasticNet           | Mean R²: -0.4450 | Std: 0.8496
SVR                  | Mean R²: -0.1229 | Std: 0.7309
KNN                  | Mean R²: -0.1923 | Std: 0.6211
Decision Tree        | Mean R²: -0.0768 | Std: 0.5153
Gradient Boosting    | Mean R²: -0.1008 | Std: 0.6066
MLP                  | Mean R²: -362.3662 | Std: 536.7459
Gaussian Process     | Mean R²: -6.8170 | Std: 6.2621
et voila
[88.48681173]
```

From test 6 we can see the accuracy of the RF is better than others.

