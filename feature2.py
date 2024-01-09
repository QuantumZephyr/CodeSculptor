# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:53:15 2023

@author: Administrator
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = pd.read_excel('C:/Users/Administrator/Desktop/daima/daima/data.xlsx')

features = data[['K1', 'K2', 'K3', 'K4', 'K5', 'KI', 'K_1', 'K_2', 'K_3', 'K_4', 'K_5', 'K_I', 'Hrxn']]
target = data['Activation Energies']

smiles_fingerprints = data['Reactant_700'].apply(lambda x: [int(c) for c in x])

smiles_fingerprint_df = pd.DataFrame(smiles_fingerprints.tolist(), columns=['rf_' + str(i) for i in range(len(smiles_fingerprints.iloc[0]))])

psmi_fingerprints = data['Product_700'].apply(lambda x: [int(c) for c in x])

psmi_fingerprint_df = pd.DataFrame(psmi_fingerprints.tolist(), columns=['pf_' + str(i) for i in range(len(psmi_fingerprints.iloc[0]))])

features = pd.concat([features, smiles_fingerprint_df, psmi_fingerprint_df], axis=1)

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5)

gbr.fit(features, target)

feature_importances = gbr.feature_importances_ * 100

feature_names = features.columns

sorted_indices = np.argsort(feature_importances)[::-1]

top_n = 60
top_indices = sorted_indices[:top_n]

colors = plt.cm.RdYlGn(np.linspace(0, 1, top_n))

plt.rc('font', family='Arial', size=10)

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(feature_names[top_indices], feature_importances[top_indices], color=colors)
ax.set_ylabel('Relative Importance [%]', fontsize=16)
ax.set_xticklabels([])
ax.set_title('Feature Importance', fontsize=18)

top_n_inset = 20  
axins = inset_axes(ax, width='60%', height='80%', loc='upper right')
bars = axins.bar(feature_names[top_indices[:top_n_inset]], feature_importances[top_indices[:top_n_inset]], color=colors[:top_n_inset])
axins.set_xticklabels(feature_names[top_indices[:top_n_inset]], rotation=45, fontsize=12)

y_ticks = np.arange(0, 41, 5)
ax.set_yticks(y_ticks)
axins.set_yticks(y_ticks)

ax.set_ylim(0, 40)
axins.set_ylim(0, 40)

plt.tight_layout()

plt.show()
