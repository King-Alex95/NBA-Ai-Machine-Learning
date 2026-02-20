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

# Separate log directory for OU to keep TensorBoard organized
tensorboard = TensorBoard(log_dir='../../Logs/OU_{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1, mode='min')

# CHANGED: Updated extension to .h5 to fix the ValueError
mcp_save = ModelCheckpoint('../../Models/Trained-Model-OU-' + current_time + '.h5', 
                           save_best_only=True, monitor='val_loss', mode='min')

# 2. LOAD UPDATED DATASET
dataset_table = "dataset_2007_2026_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
print(f"Loading Neural Network OU data from {dataset_table}...")
data = pd.read_sql_query(f"select * from \"{dataset_table}\"", con, index_col="index")
con.close()

# 3. PREPARE DATA
# Keeping 'OU' line as a feature but using 'OU-Cover' as the target
OU_target = data['OU-Cover']
total_line = data['OU']

# Drop non-predictive info
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], 
          axis=1, inplace=True, errors='ignore')

# Re-add the OU line so the network knows the "barrier" it needs to cross
data['OU'] = total_line

# Clean missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data = data.values.astype(float)

# Normalize and prepare targets
x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(OU_target).astype(int)

# 4. NEURAL NETWORK ARCHITECTURE
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dropout(0.1)) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
# 3 nodes: 0 = Under, 1 = Over, 2 = Push
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax)) 

# 5. COMPILE AND TRAIN
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print(f"Starting training on {len(x_train)} games for Over/Under...")
model.fit(x_train, y_train, 
          epochs=100, 
          validation_split=0.15, 
          batch_size=64, 
          callbacks=[tensorboard, earlyStopping, mcp_save])

# CHANGED: Updated print statement to reflect .h5
print(f'Done. Best OU model saved to: ../../Models/Trained-Model-OU-{current_time}.h5')