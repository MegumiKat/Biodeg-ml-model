import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read Excel file
df = pd.read_excel("Final version-Biodeg data summary sheet.xlsx")

# Rename columns
df.rename(columns={
    'Composition (%)': 'CHDA', 'Unnamed: 2': 'EG', 'Unnamed: 3': 'SSIA', 'Unnamed: 4': 'SSIT',
    'Unnamed: 5': 'PDA', 'Unnamed: 6': 'Quat-PDA', 'Unnamed: 7': 'PET', 'Unnamed: 8': 'LA', 'Unnamed: 9': 'TMA',
    'Unnamed: 10': 'CHDM', 'Unnamed: 11': 'FUGLA', 'Unnamed: 12': 'CHGLA', 'Unnamed: 13': 'BHET',
    'Unnamed: 14': 'NPG', 'Unnamed: 15': 'TA', 'Unnamed: 16': 'FDCA', 'Unnamed: 17': 'FDME', 'Unnamed: 18': 'Ad'
}, inplace=True)

# Drop unnecessary rows
df = df.drop(index=range(0,12))

# Fill missing values with 0
df.fillna({
    "CHDA": 0, "EG": 0, "SSIA":0, "SSIT":0, "PDA":0, "Quat-PDA":0, "PET": 0, "LA": 0, "TMA": 0,
    "CHDM": 0, "FUGLA": 0, "CHGLA": 0, "BHET": 0, "NPG": 0, "TA": 0, "FDCA": 0, "FDME": 0, "Ad": 0
}, inplace=True)

ingredients = ["CHDA","EG","SSIA","SSIT","PDA","Quat-PDA","PET","LA","TMA","CHDM","FUGLA","CHGLA","BHET","NPG","TA","FDCA","FDME","Ad"]

# Replace 0 with NaN to ignore zeros in min/max calculation
df_nozero = df[ingredients].replace(0, np.nan)

# Calculate min, max, mean (ignoring zeros)
min_values = df_nozero.min()
max_values = df_nozero.max()
mean_values = df_nozero.mean()

plt.figure(figsize=(12, 8))
y_pos = np.arange(len(ingredients))

# Draw interval bars between min and max (ignoring zeros)
plt.barh(y_pos, max_values - min_values, left=min_values, color='skyblue', alpha=0.7, label='Min-Max Interval')

# Optionally, plot mean as red dots
plt.plot(mean_values, y_pos, 'ro', label='Mean')

for i, ingredient in enumerate(ingredients):
    min_val = min_values[ingredient]
    max_val = max_values[ingredient]
    if not np.isnan(min_val):
        plt.text(min_val, i, f'{min_val:.2f}', va='center', ha='right', color='blue', fontsize=9)
    if not np.isnan(max_val):
        plt.text(max_val, i, f'{max_val:.2f}', va='center', ha='left', color='blue', fontsize=9)


plt.yticks(y_pos, ingredients)
plt.xlabel('Percentage (%)')
plt.title('Non-zero Min-Max Interval for Each Ingredient')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
