from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


regression_data = fetch_openml(data_id=550, parser='auto')
X_reg = regression_data.data
y_reg = regression_data.target.astype(float)


classification_data = fetch_openml(data_id=1462, parser='auto')
X_clf = classification_data.data
y_clf_series = pd.Series(classification_data.target)


encoder = OneHotEncoder(sparse=False, drop='first')
y_clf_encoded = encoder.fit_transform(y_clf_series.values.reshape(-1, 1))


X_train_val_reg, X_test_reg, y_train_val_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.1, random_state=42)
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_val_reg, y_train_val_reg, test_size=0.3, random_state=42)

X_train_val_clf, X_test_clf, y_train_val_clf, y_test_clf = train_test_split(X_clf, y_clf_encoded, test_size=0.1, random_state=42)
X_train_clf, X_val_clf, y_train_clf_encoded, y_val_clf_encoded = train_test_split(X_train_val_clf, y_train_val_clf, test_size=0.3, random_state=42)


scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_val_reg = scaler_reg.transform(X_val_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(X_train_clf)
X_val_clf = scaler_clf.transform(X_val_clf)
X_test_clf = scaler_clf.transform(X_test_clf)



models_reg = [
    # Model 1
    Sequential([
        Dense(64, input_dim=X_train_reg.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ]),
    # Model 2
    Sequential([
        Dense(128, input_dim=X_train_reg.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ]),
    # Model 3
    Sequential([
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='sigmoid'),
        Dropout(0.2),
        Dense(1)
    ]),
    # Model 4
    Sequential([
        Dense(128, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(1)
    ]),
]

models_clf = [
    # Model 1
    Sequential([
        Dense(64, input_dim=X_train_clf.shape[1], activation='sigmoid'),  
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ]),
    # Model 2
    Sequential([
        Dense(64, input_dim=X_train_clf.shape[1], activation='sigmoid'),  
        Dense(1, activation='sigmoid')
    ]),
    # Model 3
    Sequential([
        Dense(512, input_dim=X_train_clf.shape[1], activation='sigmoid'), 
        Dense(256, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ]),
    # Model 4
    Sequential([
        Dense(256, input_dim=X_train_clf.shape[1], activation='sigmoid'), 
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ]),
]



def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, is_regression=True):
    results = []
    best_val_metrics = []  

    for i, model in enumerate(models, 1):
        print(f"Training {'regression' if is_regression else 'classification'} model {i}")

        model.compile(optimizer='adam', 
                      loss='mean_squared_error' if is_regression else 'binary_crossentropy', 
                      metrics=['mse' if is_regression else 'accuracy'])
        

        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)

        plt.figure(figsize=(10, 5))
        metric_key = 'mse' if is_regression else 'accuracy'
        plt.plot(history.history[metric_key], label='Training ' + ('MSE' if is_regression else 'Accuracy'))
        plt.plot(history.history['val_' + metric_key], label='Validation ' + ('MSE' if is_regression else 'Accuracy'))
        plt.title(f"{'Regression' if is_regression else 'Classification'} Model {i}")
        plt.xlabel('Epochs')
        plt.ylabel('MSE' if is_regression else 'Accuracy')
        plt.legend()
        plt.show()

        if is_regression:
            best_val_metric = min(history.history['val_mse'])
        else:
            best_val_metric = max(history.history['val_accuracy'])

        best_val_metrics.append(best_val_metric)

        results.append((f'Model {i}', best_val_metric))

    results_df = pd.DataFrame(results, columns=['Model', ('Minimum Validation ' if is_regression else 'Maximum Validation ') + ('Error' if is_regression else 'Accuracy')])

    return results_df


regression_results_df = train_and_evaluate_models(models_reg, X_train_reg, y_train_reg, X_val_reg, y_val_reg, is_regression=True)
print(regression_results_df)


classification_results_df = train_and_evaluate_models(models_clf, X_train_clf, y_train_clf_encoded, X_val_clf, y_val_clf_encoded, is_regression=False)
print(classification_results_df)