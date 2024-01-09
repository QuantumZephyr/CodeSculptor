# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:56:59 2023

@author: Administrator
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import tensorflow as tf
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold

seed_value = 1001
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

data = pd.read_excel('C:/Users/Administrator/Desktop/daima/daima/data.xlsx')

features = data[['K1', 'K2', 'K3', 'K4', 'K5', 'KI', 'K_1', 'K_2', 'K_3', 'K_4', 'K_5', 'K_I', 'Hrxn']]
target = data['Activation Energies']

smiles_fingerprints = data['Reactant_300'].apply(lambda x: [int(c) for c in x])

smiles_fingerprint_df = pd.DataFrame(smiles_fingerprints.tolist(), columns=['SMILES_Fingerprint_' + str(i) for i in range(len(smiles_fingerprints.iloc[0]))])

psmi_fingerprints = data['Product_300'].apply(lambda x: [int(c) for c in x])

psmi_fingerprint_df = pd.DataFrame(psmi_fingerprints.tolist(), columns=['psmi_Fingerprint_' + str(i) for i in range(len(psmi_fingerprints.iloc[0]))])

features = pd.concat([features, smiles_fingerprint_df, psmi_fingerprint_df], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kf = KFold(n_splits=10, shuffle=True, random_state=60)

r2_scores = []
mae_scores = []
rmse_scores = []
training_r2_scores = []

validation_r2_scores = []
validation_mae_scores = []
validation_rmse_scores = []

highest_r2 = 0.0
min_mae = float('inf')
min_rmse = float('inf') 

all_true_values = []
all_predicted_values = []

model = Sequential()
model.add(Dense(500, activation='elu', kernel_initializer='he_normal', input_dim=scaled_features.shape[1]))
model.add(Dropout(0.05))
model.add(Dense(250, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(125, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(62, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adamax')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

for _ in range(8):
    for train_indices, test_indices in kf.split(scaled_features):
        X_train, X_test = scaled_features[train_indices], scaled_features[test_indices]
        y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

        val_split = int(0.05 * len(X_train))
        X_val, y_val = X_train[:val_split], y_train[:val_split]
        X_train, y_train = X_train[val_split:], y_train[val_split:]

        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

        predictions = model.predict(X_test).flatten()
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        val_predictions = model.predict(X_val).flatten()
        val_r2 = r2_score(y_val, val_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = mean_squared_error(y_val, val_predictions, squared=False)

        y_train_pred = model.predict(X_train).flatten()
        training_r2 = r2_score(y_train, y_train_pred)
        training_r2_scores.append(training_r2)

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        validation_r2_scores.append(val_r2)
        validation_mae_scores.append(val_mae)
        validation_rmse_scores.append(val_rmse)

        if r2 > highest_r2:
            highest_r2 = r2
        if mae < min_mae:
            min_mae = mae
        if rmse < min_rmse:
            min_rmse = rmse

        all_true_values.extend(y_test)
        all_predicted_values.extend(predictions)

mean_r2 = sum(r2_scores) / len(r2_scores)
mean_mae = sum(mae_scores) / len(mae_scores)
mean_rmse = sum(rmse_scores) / len(rmse_scores)
mean_training_r2 = sum(training_r2_scores) / len(training_r2_scores)

mean_validation_r2 = sum(validation_r2_scores) / len(validation_r2_scores)
mean_validation_mae = sum(validation_mae_scores) / len(validation_mae_scores)
mean_validation_rmse = sum(validation_rmse_scores) / len(validation_rmse_scores)

print("Average R2 Score:", mean_r2)
print("Average MAE:", mean_mae)
print("Average RMSE:", mean_rmse)
print("Average Training R2 Score:", mean_training_r2)

print("Average Validation R2 Score:", mean_validation_r2)
print("Average Validation MAE:", mean_validation_mae)
print("Average Validation RMSE:", mean_validation_rmse)

print("Highest R2:", highest_r2)
print("Minimum MAE:", min_mae)
print("Minimum RMSE:", min_rmse)

plt.figure(figsize=(6, 5))
plt.scatter(y_test,predictions, s=14,c=(111/255,25/255,111/255) , label='Test Set', alpha=0.8)
plt.scatter(y_val, val_predictions, s=14, c=(0/255,120/255,80/255), label='Validation Set', alpha=0.8)
plt.plot([min(all_true_values), max(all_true_values)], [min(all_true_values), max(all_true_values)], linestyle='--', lw=2,c='red', label='y=x',alpha=0.8)  # 对角线
plt.xlabel('True Value (eV)', fontsize=18)
plt.ylabel('Predicted Value (eV)', fontsize=18)
plt.title('Neural Network Predictions', fontsize=20)
plt.xlim([0, 7])
plt.ylim([0, 7])
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()


