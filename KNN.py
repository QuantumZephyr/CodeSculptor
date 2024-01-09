# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:31:54 2023

@author: Administrator
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

seed_value = 1001
random_state = 60
np.random.seed(seed_value)
random.seed(seed_value)

data = pd.read_excel('C:/Users/Administrator/Desktop/daima/daima/data.xlsx')

features = data[['K1', 'K3', 'K4', 'K5', 'KI', 'K_3', 'K_4', 'K_5', 'K_I', 'Hrxn']]
target = data['Activation Energies']

smiles_fingerprints = data['Reactant_700'].apply(lambda x: [int(c) for c in x])

smiles_fingerprint_df = pd.DataFrame(smiles_fingerprints.tolist(), columns=['SMILES_Fingerprint_' + str(i) for i in range(len(smiles_fingerprints.iloc[0]))])

psmi_fingerprints = data['Product_700'].apply(lambda x: [int(c) for c in x])

psmi_fingerprint_df = pd.DataFrame(psmi_fingerprints.tolist(), columns=['psmi_Fingerprint_' + str(i) for i in range(len(psmi_fingerprints.iloc[0]))])

features = pd.concat([features, smiles_fingerprint_df, psmi_fingerprint_df], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

r2_scores = []
mae_scores = []
rmse_scores = []

knn_model = KNeighborsRegressor(n_neighbors=13,  metric='minkowski', p=1)

for _ in range(8): 
    for train_indices, test_indices in kf.split(scaled_features):
        X_train, X_test = scaled_features[train_indices], scaled_features[test_indices]
        y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

        knn_model.fit(X_train, y_train)

        predictions = knn_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

mean_r2 = sum(r2_scores) / len(r2_scores)
mean_mae = sum(mae_scores) / len(mae_scores)
mean_rmse = sum(rmse_scores) / len(rmse_scores)

print("Average R2 Score:", mean_r2)
print("Average MAE:", mean_mae)
print("Average RMSE:", mean_rmse)

plt.figure(figsize=(6, 5))
plt.rcParams['font.family'] = 'Arial'
plt.scatter(y_test, predictions,s=8, c='red', label='prediction vs. true', alpha=0.7)
plt.plot([0, 8], [0, 8], 'k-', lw=2, label='y=x')
plt.xlabel('true value (eV)',fontsize=12)
plt.ylabel('prediction value (eV)',fontsize=12)
plt.title('KNN')

mae_value = mean_absolute_error(y_test, predictions)
plt.text(5.5, 1.5, f'MAE = {mae_value:.2f} eV', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.legend()
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
