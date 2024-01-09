# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:54:40 2023

@author: Administrator
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Model': ['2', '3', '4', '5', '6'],
    'R2': [0.962, 0.962, 0.965, 0.962, 0.958],
    'MAE': [0.087, 0.087, 0.079, 0.082, 0.089],
    'RMSE': [0.173, 0.172, 0.164, 0.173, 0.180]
})

plt.rc('font', family='Arial')

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(data['Model'], data['R2'], label='R2', marker='o', linestyle='-', linewidth=2)

ax.plot(data['Model'], data['MAE'], label='MAE', marker='s', linestyle='-', linewidth=2)

ax.plot(data['Model'], data['RMSE'], label='RMSE', marker='^', linestyle='-', linewidth=2)


ax.set_xlabel('Number of hidden layers', fontsize=18)
ax.set_ylabel('Performance Metric Value', fontsize=18)

ax.tick_params(axis='both', which='major', labelsize=16)

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.legend(fontsize=14)
plt.show()
