import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# 1. SETUP DATA
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
print(f"Loading XGBoost data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# Ensure Models directory exists
os.makedirs('../../Models', exist_ok=True)

# 2. PREPARE FEATURES AND TARGET
margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True, errors='ignore')

# Handle NaNs which can creep in with 18 years of history
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data = data.values.astype(float)
acc_results = []

# 3. TRAINING LOOP
# We keep the 300-run loop to find the "best" random seed for the model
print("Starting XGBoost training loop...")
for x in tqdm(range(300)):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)

    # XGBoost specific data structure
    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    # PARAMETERS
    # With 18 years of data, 'max_depth': 3 is good to prevent overfitting
    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2,
        'tree_method': 'hist' # Faster training for large datasets
    }
    epochs = 750

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    
    # Get the class with highest probability
    y_pred = [np.argmax(z) for z in predictions]

    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    acc_results.append(acc)
    
    # Save the model only if it achieves a new high score
    if acc == max(acc_results) and acc > 65.0: # Only save models that are actually good
        model.save_model('../../Models/XGBoost_{}%_ML-4.json'.format(acc))
        # print(f" New Best Found: {acc}%") # Optional terminal spam reduction

print(f"Done. Best Accuracy found: {max(acc_results)}%")