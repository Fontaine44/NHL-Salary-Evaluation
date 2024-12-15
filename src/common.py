from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_mformatter(precision=0):
    # Million formatter to enhance axis visualization
    return FuncFormatter(lambda x, pos: f'{x/1e6:.{precision}f}M')

def load_dataset(preprocess=True):
    df = pd.read_csv('data/dataset.csv')

    if preprocess:
        df = preprocess_dataset(df)

    return df

def preprocess_dataset(df):
    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=["team", "nationality", "shootsCatches", "position"], drop_first=True)

    # Transfrom bool columns to int
    df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)

    # Drop info columns
    df = df.drop(columns=["name", "playerId", "salary", "capPercentage"])

    return df

def split_dataset(df):
    X_data = df.drop(columns="adjustedSalary")
    y_data = df["adjustedSalary"]

    return X_data, y_data

def split_train_test(X_data, y_data):
    # Separate into training and testing data
    X_train = X_data[X_data["season"] != 2023].to_numpy()
    X_test = X_data[X_data["season"] == 2023].to_numpy()

    # Labels
    y_train = y_data[X_data["season"] != 2023].to_numpy()
    y_test = y_data[X_data["season"] == 2023].to_numpy()

    return X_train, y_train, X_test, y_test

def standard_scaler(X_train, X_test):
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Init models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_features="sqrt", random_state=12345)
    svr = SVR(kernel="linear", C=1000, epsilon=0.01)
    knr = KNeighborsRegressor(n_neighbors=10, algorithm="auto", weights="distance", p=2)

    models = [
        ("Linear Regression", lr),
        ("Random Forest", rf),
        ("Support Vector", svr),
        ("K-Nearest Neighbors", knr)
    ]
    predictions = []
    data = []

    for name, model in models:
        # Train
        model, train_time = train(model, X_train, y_train)

        # Predict
        y_pred = np.round(model.predict(X_test), 0)
        predictions.append(y_pred)

        # Evaluate
        metrics = evaluate(y_test, y_pred)

        data.append((name,) + metrics + (train_time,))
        
    results_df = pd.DataFrame(
        data,
        columns=["Model", "R2", "MAE", "Top-100 MAE", "Top-50 MAE", "SMAPE", "Train time (sec)"]
    )

    results_df.set_index("Model", inplace=True)
    
    return results_df, predictions

def train(model, X, y):
    # Train the model and measure training time
    start_time = time()
    model.fit(X, y)
    end_time = time()

    train_time = np.round(end_time - start_time, 2)

    return model, train_time

def evaluate(y_test, y_pred):
    # Compute the metrics
    r2 = r2_score(y_test, y_pred)
    mae = mae_score(y_test, y_pred)
    top_100_mae = top_k_mae_score(100, y_test, y_pred)
    top_50_mae = top_k_mae_score(50, y_test, y_pred)
    smape = smape_score(y_test, y_pred)

    return r2, mae, top_100_mae, top_50_mae, smape

def r2_score(y_test, y_pred):
    # Compute R2 score

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return np.round(r2, 4)

def mae_score(y_true, y_pred):
    # Compute MAE
    mae = np.mean(np.abs(y_true - y_pred))
    mae = np.round(mae, 0)
    return "{:,.0f}".format(mae)

def top_k_mae_score(k, y_test, y_pred):
    # Compute MAE for the top-k highest paid players
    top_k_indices = np.argsort(y_test)[-k:]
    top_k_mae = np.mean(np.abs(y_test[top_k_indices] - y_pred[top_k_indices]))
    top_k_mae = np.round(top_k_mae, 0)
    return "{:,.0f}".format(top_k_mae)

def smape_score(y_test, y_pred):
    # Compute Symmetric Mean Absolute Percentage Error (SMAPE)
    num = np.abs(y_test - y_pred)
    den = (np.abs(y_test) + np.abs(y_pred)) / 2 + 1e-10

    smape = np.mean(num / den)
    smape = np.round(smape, 4)
    return smape

def plot_metrics(results_df, x_name, metrics=["R2", "MAE", "Top-100 MAE", "Top-50 MAE", "SMAPE", "Train time (sec)"], n_rows=2, n_cols=3):
    # Plot all metrics in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    # Iterate over each metric and plot in a subplot
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for model in results_df.index.unique():
            # Select metric values and x values for the current model
            x_values = results_df[results_df.index == model][x_name]
            metric_values = results_df[results_df.index == model][metric]
            if metric in ["MAE", "Top-100 MAE", "Top-50 MAE"]:
                metric_values = metric_values.str.replace(',', '').astype(float)
            
            # Plot
            ax.plot(x_values, metric_values, label=model, marker='.')

        # Format MAE to show in millions
        if metric in ["MAE", "Top-100 MAE", "Top-50 MAE"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(get_mformatter(2)))
        
        ax.set_title(f"{metric} vs {x_name}")
        ax.set_ylabel(f"{metric}")
        ax.set_xlabel(x_name)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
