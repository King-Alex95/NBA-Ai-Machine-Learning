import sqlite3
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# 1. SETUP LOGGING AND MODELS
current_time = str(int(time.time()))
os.makedirs('../../Logs', exist_ok=True)
os.makedirs('../../Models', exist_ok=True)

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
# Patience increased to 15 given the larger dataset size
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

# CHANGED: Updated extension to .h5 to fix the ValueError
mcp_save = ModelCheckpoint('../../Models/Trained-Model-ML-' + current_time + '.h5', 
                           save_best_only=True, monitor='val_loss', mode='min')

# 2. LOAD UPDATED DATASET
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
print(f"Loading Neural Network data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# 3. PREPARE DATA
margin = data['Home-Team-Win']
# Ensure we drop all non-feature columns
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU', 'OU-Cover'], 
          axis=1, inplace=True, errors='ignore')

# Handle missing values that might exist in historical data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data = data.values.astype(float)

# Normalize features (scaling them between 0 and 1) so the Neural Network learns faster
x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(margin)

# 4. NEURAL NETWORK ARCHITECTURE
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(x_train.shape[1],))) 
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dropout(0.2)) 
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax)) 

# 5. COMPILE AND TRAIN
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print(f"Starting training on {len(x_train)} games...")
model.fit(x_train, y_train, 
          epochs=100, 
          validation_split=0.15, 
          batch_size=64, 
          callbacks=[tensorboard, earlyStopping, mcp_save])

# CHANGED: Updated print statement to reflect .h5
print(f'Done. Best model saved to: ../../Models/Trained-Model-ML-{current_time}.h5')