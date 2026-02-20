import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# 1. LOAD UPDATED DATASET
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
print(f"Loading XGBoost Totals data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# Ensure Models directory exists
os.makedirs('../../Models', exist_ok=True)

# 2. PREPARE DATA
OU_target = data['OU-Cover']
total_line = data['OU']

# Drop non-predictive info but preserve features
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], 
          axis=1, inplace=True, errors='ignore')

# Re-add the OU line as a feature
data['OU'] = np.asarray(total_line)

# Handle potential missing values in historical data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data_values = data.values.astype(float)
acc_results = []

# 3. TRAINING LOOP
print("Starting XGBoost Totals loop (100 iterations)...")
for x in tqdm(range(100)):
    x_train, x_test, y_train, y_test = train_test_split(data_values, OU_target, test_size=.1)

    # Note: XGBoost requires labels to be 0, 1, 2 for multi-class
    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test)

    # PARAMETERS
    # Note: max_depth 20 is very high (potential overfitting). 
    # Added tree_method='hist' for performance with 18 years of data.
    param = {
        'max_depth': 6, # Reduced from 20 to 6 for better generalization across 18 years
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist' 
    }
    epochs = 750

    model = xgb.train(param, train, epochs)

    predictions = model.predict(test)
    y_pred = [np.argmax(z) for z in predictions]

    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    acc_results.append(acc)
    
    # Save model if it hits a new record
    if acc == max(acc_results) and acc > 52.0:
        print(f" New Best Totals Accuracy: {acc}%")
        model.save_model(f'../../Models/XGBoost_{acc}%_UO-9.json')

print(f"Done. Best Over/Under Accuracy found: {max(acc_results)}%")