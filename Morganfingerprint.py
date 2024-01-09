# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:50:41 2023

@author: Administrator
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

data = pd.read_excel('C:/Users/Administrator/Desktop/daima/daima/data.xlsx')

reactant_smiles = data['rsmi']
product_smiles = data['psmi']

output_data = pd.DataFrame(columns=['Reactant Fingerprint', 'Product Fingerprint'])

for reactant, product in zip(reactant_smiles, product_smiles):
    reactant_mol = Chem.MolFromSmiles(reactant)
    product_mol = Chem.MolFromSmiles(product)
    
    reactant_fingerprint = AllChem.GetMorganFingerprintAsBitVect(reactant_mol, radius=2, nBits=25)
    product_fingerprint = AllChem.GetMorganFingerprintAsBitVect(product_mol, radius=2, nBits=25)
    
    reactant_fingerprint_string = reactant_fingerprint.ToBitString()
    product_fingerprint_string = product_fingerprint.ToBitString()
    
    output_data = pd.concat([output_data, pd.DataFrame({'Reactant SMILES': [reactant], 'Product SMILES': [product], 'Reactant Fingerprint': [reactant_fingerprint_string], 'Product Fingerprint': [product_fingerprint_string]})], ignore_index=True)

output_data.to_excel('C:/Users/Administrator/Desktop/daima/daima/3.xlsx', index=False)
