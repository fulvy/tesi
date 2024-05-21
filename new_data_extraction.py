import os
import pandas as pd

features_folder = "C:\\Users\\fulvi\\DataspellProjects\\tesi\\extraction_matlab\\Features"

files = [f for f in os.listdir(features_folder) if f.endswith('.txt')]
dataframes = []

#colonne attese nei file
columns = [
    "X", "Y", "Average Score", "Direction", "Area", "Perimeter", "Eccentricity",
    "Extent", "MajorAxisLength", "MinorAxisLength", "Orientation", "Solidity",
    "Circularity", "MaxFeretDiameter", "MaxFeretAngle", "MinFeretDiameter", "MinFeretAngle"
]

#%%
#leggo e carico i file in un DataFrame
for file in files:
    file_path = os.path.join(features_folder, file)
    df = pd.read_csv(file_path, delimiter=" ", header=None, names=columns, skiprows=1)
    dataframes.append(df)

    #print(f"Data from file: {file}")
    #print(df.head())  # Mostra le prime 5 righe

combined_df = pd.concat(dataframes, ignore_index=True)

#show df
print("Combined DataFrame:")
print(combined_df.head())

combined_df.to_csv("combined_new_features.csv", index=False)
