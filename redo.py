import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# 模型库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# --------------------------
# 📌 1. 读取数据与清洗
# --------------------------
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop(index=[0])
    df.rename(columns={
        'Composition (%)': 'CHDA', 'Unnamed: 2': 'EG', 'Unnamed: 3': 'SSIA', 'Unnamed: 4': 'SSIT',
        'Unnamed: 5': 'PDA', 'Unnamed: 6': 'Quat-PDA', 'Unnamed: 7': 'PET', 'Unnamed: 8': 'LA',
        'Unnamed: 9': 'TMA', 'Unnamed: 10': 'CHDM', 'Unnamed: 11': 'FUGLA', 'Unnamed: 12': 'CHGLA',
        'Unnamed: 13': 'BHET', 'Unnamed: 14': 'NPG', 'Unnamed: 15': 'TA', 'Unnamed: 16': 'FDCA',
        'Unnamed: 17': 'FDME', 'Unnamed: 18': 'Ad',
        '301F test result\n(28D)': '301F test result 30D',
        '301F test result\n(60D)': '301F test result 60D'
    }, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --------------------------
# 📌 2. 选择目标
# --------------------------
def select_target(df):
    print("Which biodegradability do you want to predict? 302B / 301F30D / 301F60D")
    choice = input().strip().lower()
    target_map = {
        '302b': '302B test result',
        '301f30d': '301F test result 30D',
        '30d': '301F test result 30D',
        '301f60d': '301F test result 60D',
        '60d': '301F test result 60D'
    }
    target_col = target_map.get(choice)
    if not target_col or target_col not in df.columns:
        raise ValueError("Invalid choice. Please choose from 302B / 301F30D / 301F60D")
    
    df = df.dropna(subset=[target_col])
    features = ["CHDA","EG","SSIA","SSIT","PDA","Quat-PDA","PET","LA","TMA","CHDM","FUGLA","CHGLA",
                "BHET","NPG","TA","FDCA","FDME","Ad"]
    X = df[features]
    y = df[target_col]
    return X, y

# --------------------------
# 📌 3. 模型定义
# --------------------------
def get_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=0.5),
        "Lasso": Lasso(alpha=0.5),
        "ElasticNet": ElasticNet(alpha=0.5),
        "SVR": SVR(kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=5),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
        "MLP": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
        "Gaussian Process": GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-6)
    }

# --------------------------
# 📌 4. 训练模型 & 输出 R² & CV
# --------------------------
def evaluate_models(models, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    results = []

    print("\n📊 Model Performance (R² Scores):")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"{name:<20} | R²: {r2:.3f} | CV Mean: {np.mean(cv_scores):.3f} | CV Std: {np.std(cv_scores):.3f}")
        results.append((name, r2, np.mean(cv_scores), np.std(cv_scores)))
    
    return results

# --------------------------
# 📌 5. 可视化模型对比
# --------------------------
def plot_model_results(results):
    df_results = pd.DataFrame(results, columns=["Model", "R2", "CV Mean", "CV Std"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="R2", y="Model", data=df_results, palette="viridis")
    plt.title("Model R² Score Comparison")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# 📌 6. PCA 可视化
# --------------------------
def pca_analysis(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm")
    plt.title("PCA 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X_pca

# --------------------------
# ✅ 主程序入口
# --------------------------
def main():
    df = load_and_prepare_data("Final version-Biodeg data summary sheet.xlsx")
    X, y = select_target(df)
    models = get_models()
    results = evaluate_models(models, X, y)
    plot_model_results(results)
    pca_analysis(X, y)

# --------------------------
# 🚀 运行
# --------------------------
if __name__ == "__main__":
    main()