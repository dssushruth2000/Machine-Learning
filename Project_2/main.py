import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def load_dataset(dataset_id):
    return fetch_openml(data_id=dataset_id, as_frame=True)

dataset = load_dataset(44987)
X, y = dataset.data, dataset.target


nominal_features = list(X.select_dtypes(include=['object', 'category']).columns)
preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), nominal_features),
    ('scale', StandardScaler(), [col for col in X.columns if col not in nominal_features])
])

def plot_combined_learning_curves(models, titles, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for model, title, color in zip(models, titles, colors):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
        train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)
        test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)
        plt.plot(train_sizes, train_scores_mean, 'o-', color=color, label=f"{title} - Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color=color, linestyle='dashed', label=f"{title} - Cross-validation score")

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.show()
    
# Task 1
knn_models = [make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=k)) for k in [3, 5, 7]]
knn_titles = [f"KNN with k={k}" for k in [3, 5, 7]]
plot_combined_learning_curves(knn_models, knn_titles, X, y, cv=10, n_jobs=-1)    

# Task 2
knn_param_grid = {'kneighborsregressor__n_neighbors': [1, 3, 5, 7, 10]}
knn_pipeline = make_pipeline(preprocessor, KNeighborsRegressor())
knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
knn_grid_search.fit(X, y)
best_knn_model = knn_grid_search.best_estimator_


dt_param_grid = {
    'decisiontreeregressor__max_depth': [None, 5, 10, 15, 20],
    'decisiontreeregressor__min_samples_leaf': [1, 2, 3, 4, 5],
    'decisiontreeregressor__min_samples_split': [2, 5, 10]
}
dt_pipeline = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=42))
dt_grid_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid_search.fit(X, y)
best_dt_model = dt_grid_search.best_estimator_


lr_model = make_pipeline(preprocessor, LinearRegression())

models = [best_knn_model, best_dt_model, lr_model]
titles = ['Tunned KNN', 'Tunned Decision Tree', 'Linear Regression']
plot_combined_learning_curves(models, titles, X, y, cv=10, n_jobs=-1)


def get_last_rmse(model, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)
    test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)
    return train_sizes[-1], train_scores_mean[-1], test_scores_mean[-1]


# Task 1
results_task1 = []
k_values = [3, 5, 7]
for k in k_values:
    knn_model = make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=k))
    train_size, rmse_train, rmse_test = get_last_rmse(knn_model, X, y, cv=10, n_jobs=-1)
    results_task1.append({'k': k, 'Train Size': train_size, 'RMSE (Training)': rmse_train, 'RMSE (Test)': rmse_test})

results_df_task1 = pd.DataFrame(results_task1)
print("Task 1 Results:")
print(results_df_task1)

# Task 2
results_task2 = []
models_task2 = [best_knn_model, best_dt_model, lr_model]
model_names = ['Tunned KNN', 'Tunned Decision Tree', 'Linear Regression']
for model, name in zip(models_task2, model_names):
    train_size, rmse_train, rmse_test = get_last_rmse(model, X, y, cv=10, n_jobs=-1)
    results_task2.append({'Model': name, 'Train Size': train_size, 'RMSE (Training)': rmse_train, 'RMSE (Test)': rmse_test})

results_df_task2 = pd.DataFrame(results_task2)
print("Task 2 Results:")
print(results_df_task2)
