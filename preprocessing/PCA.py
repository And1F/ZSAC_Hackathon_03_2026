# %%
import sklearn.decomposition as skd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
import glob
import os

# %%
def train_pca(path_to_csv):
    data_matrix = pd.read_csv(path_to_csv).values
    pca = skd.PCA(n_components = 'mle', svd_solver = 'full') # Minka’s MLE is used to guess the dimension.
    pca.fit(data_matrix)
    
    print("componets:", pca.n_components_)
    print("Explained variance ratio per pc:", pca.explained_variance_ratio_)

    return pca

def apply_pca(pca, path_to_csv, output_filename):
    # Load the raw data
    data_matrix = pd.read_csv(path_to_csv).values
    
    # Transform the data using your fitted PCA object
    pca_data = pca.transform(data_matrix)
    
    # Create column names PC1, PC2, ..., PCn
    column_names = [f"PC{i+1}" for i in range(pca_data.shape[1])]
    
    # Create a DataFrame and save to CSV
    df_pca = pd.DataFrame(pca_data, columns=column_names)
    df_pca.to_csv(output_filename, index=False)
    
    print(f"Successfully saved {pca_data.shape[1]} components to {output_filename}")
    
def merge_csvs_for_pca(folder_path, output_filename="data/merged_dataset.csv"):
    # 1. Get a list of all CSV files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print("No CSV files found in the specified folder.")
        return None

    # 2. Use a list comprehension to read all CSVs (efficient for memory)
    # Note: This assumes all CSVs have the same header/columns
    data_list = [pd.read_csv(f) for f in all_files]
    
    # 3. Concatenate them into one large DataFrame
    merged_df = pd.concat(data_list, axis=0, ignore_index=True)
    
    # 4. Save to a single file for future use
    merged_df.to_csv(output_filename, index=False)
    
    print(f"Merged {len(all_files)} files into {output_filename}")
    print(f"Final shape: {merged_df.shape}")
    
    return merged_df
    
merge_csvs_for_pca("PATH_TO_FOLDER_WITH_CSV_FILES")

pca = train_pca("data/merged_dataset.csv")
apply_pca(pca, "data/merged_dataset.csv", "data/pca_transformed.csv")


