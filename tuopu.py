# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:53:22 2023

@author: Administrator
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

data = pd.read_excel('C:/Users/Administrator/Desktop/daima/daima/data.xlsx')

smiles = data['rsmi']

adj_matrices = []
topological_indices = []

for smi in smiles:
    mol = Chem.MolFromSmiles(smi)

    adj_matrix = Chem.GetAdjacencyMatrix(mol)

    topological_index_k1 = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    topological_index_k2 = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    topological_index_k3 = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    topological_index_k4 = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    topological_index_k5 = Chem.GetSSSR(mol)  # 使用GetSSSR函数计算环数
    topological_index_kinf = rdMolDescriptors.CalcNumRotatableBonds(mol)

    adj_matrices.append(adj_matrix)
    topological_indices.append([topological_index_k1, topological_index_k2, topological_index_k3,
                                topological_index_k4, topological_index_k5, topological_index_kinf])

result_df = pd.DataFrame({'SMILES': smiles, 'Adjacency Matrix': adj_matrices,
                          'Kappa 1': [indices[0] for indices in topological_indices],
                          'Kappa 2': [indices[1] for indices in topological_indices],
                          'Kappa 3': [indices[2] for indices in topological_indices],
                          'Kappa 4': [indices[3] for indices in topological_indices],
                          'Kappa 5': [indices[4] for indices in topological_indices],
                          'Kappa Inf': [indices[5] for indices in topological_indices]})

result_df.to_excel('C:/Users/Administrator/Desktop/daima/daima/1.xlsx', index=False)
