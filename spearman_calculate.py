
import pandas as pd
from scipy.stats import spearmanr

file_path = 'C:/Users/fulvi/DataspellProjects/tesi/results/hypersearch_result_knn.csv'
df = pd.read_csv(file_path)

numeric_columns = df.drop(columns=['Unnamed: 0', 'layer_type', 'writer'])

spearman_results = {}

# correlazione di Spearman tra ciascuna colonna numerica e 'err'
for column in numeric_columns.columns:
    if column != 'err':
        corr, p_value = spearmanr(numeric_columns[column], numeric_columns['err'], nan_policy='omit')
        spearman_results[column] = (corr, p_value)

for feature, (corr, p_value) in spearman_results.items():
    print(f"Feature: {feature}, Spearman Correlation: {corr}, p-value: {p_value}")
