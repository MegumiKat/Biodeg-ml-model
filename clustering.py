import pandas as pd
import numpy as np   # numerical calcul
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit # divided in train and test split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression 
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report  # to evaluate the model
from sklearn.metrics import r2_score # give the r2 of the model

# Lire le fichier Excel
df = pd.read_excel("Final version-Biodeg data summary sheet.xlsx")
df.rename(columns={'Composition (%)': 'CHDA', 'Unnamed: 2': 'EG', 'Unnamed: 3': 'SSIA', 'Unnamed: 4': 'SSIT' , 'Unnamed: 5': 'PDA' , 'Unnamed: 6': 'Quat-PDA' , 'Unnamed: 7': 'PET', 'Unnamed: 8': 'LA' , 'Unnamed: 9': 'TMA'}, inplace=True)
df.rename(columns={'Unnamed: 10': 'CHDM', 'Unnamed: 11': 'FUGLA', 'Unnamed: 12': 'CHGLA', 'Unnamed: 13': 'BHET' , 'Unnamed: 14': 'NPG' , 'Unnamed: 15': 'TA' , 'Unnamed: 16': 'FDCA', 'Unnamed: 17': 'FDME' , 'Unnamed: 18': 'Ad'}, inplace=True)
df.rename(columns={'301F test result\n(28D)': '301F test result 30D', '301F test result\n(60D)': '301F test result 60D'}, inplace=True)

# Supprimer les lignes inutiles (les 3 suivantes après la ligne fusionnée)
df = df.drop(index=[0])


#df.fillna({"Calories": 0}, inplace=True) # replace NaN by 0 in the selected columns (possible to put many columns in one code line ?)
df.fillna({"CHDA": 0, "EG": 0, "SSIA":0, "SSIT":0, "PDA":0, "Quat-PDA":0, "PET": 0, "LA": 0, "TMA": 0, "CHDM": 0, "FUGLA": 0, "CHGLA": 0, "BHET": 0, "NPG": 0, "TA": 0, "FDCA": 0, "FDME": 0, "Ad": 0}, inplace=True)
print(df)
print("Which biodegradability do you want to predict ? 302B, 301F30D or 301F60D")
choice = input()

if choice == "302B" or choice == "302b":
  df = df.dropna(subset=['302B test result'])  #or '302B_test_result'
  features = ["CHDA","EG","SSIA","SSIT","PDA","Quat-PDA","PET","LA","TMA","CHDM","FUGLA","CHGLA","BHET","NPG","TA","FDCA","FDME","Ad"]
  X = df[features]
  y_raw = df["302B test result"]
elif choice == "301F30D" or choice == "301f30d" or choice == "301F30d" or choice == "301f30D" or choice == "30d" or choice == "30D":
  df = df.dropna(subset=['301F test result 30D'])  #or '301F_test_result_30D'
  features = ["CHDA","EG","SSIA","SSIT","PDA","Quat-PDA","PET","LA","TMA","CHDM","FUGLA","CHGLA","BHET","NPG","TA","FDCA","FDME","Ad"]
  X = df[features]
  y_raw = df["301F test result 30D"]
elif choice == "301F60D" or choice == "301f60d" or choice == "301F60d" or choice == "301f60D" or choice == "60d" or choice == "60D" :
  df = df.dropna(subset=['301F test result 60D'])  #or '301F_test_result_60D'
  features = ["CHDA","EG","SSIA","SSIT","PDA","Quat-PDA","PET","LA","TMA","CHDM","FUGLA","CHGLA","BHET","NPG","TA","FDCA","FDME","Ad"]
  X = df[features]
  y_raw = df["301F test result 60D"]
else :
  print("please choose a name among the 3 possibilities or be sure to write it as the same way as in the previous list")
  #exit()  #it is really useful to crash the program since there is no action after ?


# def label_degradation(value):
#     if value < 0.3:
#         return "Low"
#     elif value < 0.6:
#         return "Medium"
#     else:
#         return "High"

# y = y_raw.apply(label_degradation)

y, bins= pd.qcut(y_raw, q=3, labels=["Low", "Medium", "High"], retbins=True)


print(y.value_counts())
print("cutpoints:")
for i in range(len(bins)-1):
    print(f"Group {i+1}: {bins[i]:.4f} ~ {bins[i+1]:.4f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e6))
models = {
   "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(solver='saga', max_iter=2000),
    "SVC": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    "Gaussian Process": GaussianProcessClassifier(kernel=kernel)
}

print("\n========== Cross-validation accuracy (5-Fold) ==========\n")
cv_results = []
for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    # scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"{name:<20} | Mean Accuracy: {mean_score:.4f} | Std: {std_score:.4f}")
    cv_results.append((name, mean_score, std_score))


print("\n========== Final model training + prediction (using RandomForest) ==========\n")
best_model = models["Random Forest"]
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
# ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test)
# plt.title("Confusion Matrix - Random Forest")
# plt.grid(False)
# plt.show()


sample = [[36.3, 45.45, 4.54, 0, 4.54, 0, 0, 9.09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 可替换
sample_scaled = scaler.transform(sample)
sample_pred = best_model.predict(sample_scaled)
print(f"\nThe predicted degradation level of the sample is: {sample_pred[0]}")