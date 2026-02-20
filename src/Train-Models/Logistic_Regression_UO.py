import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 1. UPDATE: Using the complete 18-year dataset
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
print(f"Loading Totals data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# 2. HANDLE PUSHES
# In your Create_Games script, a 'Push' is marked as 2. 
# Logistic Regression works best with binary (0 or 1). We remove Pushes to improve accuracy.
data = data[data['OU-Cover'] != 2]

OU_target = data['OU-Cover']
total_line = data['OU']

# 3. FEATURE PREPARATION
# We drop non-predictive info but KEEP 'OU' as a feature because the line itself 
# often influences the betting behavior/outcome.
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], 
          axis=1, inplace=True, errors='ignore')

# Re-add the OU line as a specific feature for the model to consider
data['OU'] = total_line

# 4. CLEANING & CONVERSION
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)
OU_target = OU_target.loc[data.index]

X = data.values.astype(float)
y = OU_target.values.astype(int)

# 5. SPLIT AND TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Training on {len(X_train)} games. Predicting Over/Under outcomes...")

# Using liblinear solver which is often better for binary classification with betting data
model = LogisticRegression(max_iter=2000, solver='liblinear')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. RESULTS
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Under', 'Over'])

print(f"--- Totals Model Results ({dataset_table}) ---")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)