# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:24:09 2023

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

highest_r2 = 0.0
min_mae = float('inf')
min_rmse = float('inf') 

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

training_loss = []
validation_loss = []

for _ in range(8):
    for train_indices, test_indices in kf.split(scaled_features):
        X_train, X_test = scaled_features[train_indices], scaled_features[test_indices]
        y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

        # Train the model (with Early Stopping callback)
        history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.05, callbacks=[early_stopping], verbose=0)

        # Store the training and validation loss for each epoch
        training_loss.extend(history.history['loss'])
        validation_loss.extend(history.history['val_loss'])

        predictions = model.predict(X_test).flatten()
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        y_train_pred = model.predict(X_train).flatten()
        training_r2 = r2_score(y_train, y_train_pred)
        training_r2_scores.append(training_r2)

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        if r2 > highest_r2:
            highest_r2 = r2
        if mae < min_mae:
            min_mae = mae
        if rmse < min_rmse:
            min_rmse = rmse

plt.figure(figsize=(4, 3))
epochs = range(0, len(training_loss))
plt.plot(epochs, training_loss, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
