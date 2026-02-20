import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 1. UPDATE: Using your new complete dataset name
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")

print(f"Loading data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# 2. PREPARE TARGETS AND FEATURES
# We use 'Home-Team-Win' as our Y (what we want to predict)
margin = data['Home-Team-Win']

# Drop non-predictive columns and targets from the feature set
# Note: Added 'Days-Rest-Home' and 'Days-Rest-Away' to the columns we KEEP
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True, errors='ignore')

# 3. DATA CLEANING (Critical for large historical datasets)
# Replace any Infinity values with NaN, then drop rows with missing stats
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)
# Ensure the labels (margin) match the cleaned data rows
margin = margin.loc[data.index]

# Convert to float for sklearn
data = data.values.astype(float)

# 4. SPLIT AND TRAIN
# Increased random_state for consistency across runs
X_train, X_test, y_train, y_test = train_test_split(data, margin, test_size=0.1, random_state=42)

print(f"Training on {len(X_train)} games. Testing on {len(X_test)} games.")

# Using a higher 'max_iter' because 20,000+ rows might need more time to converge
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 5. RESULTS
report = classification_report(y_test, y_pred)

print(f"--- Results for {dataset_table} ---")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)