import pandas as pd
from sklearn.decomposition import PCA

def task_func(df):
    """
    Perform Principal Component Analysis (PCA) on the DataFrame and record the first two main components.

    Parameters:
    - df (DataFrame): The pandas DataFrame.

    Returns:
    - df_pca (DataFrame): The DataFrame with the first two principal components named 'PC1' and 'PC2' as columns.
    """
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])
    return df_pca

# Test with the example
df = pd.DataFrame([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3]], columns = ['x', 'y', 'z'])
df_pca = task_func(df)
print("Actual output:")
print(df_pca)
print("\nExpected output:")
print("        PC1       PC2")
print("0  0.334781 -0.011992")
print("1 -0.187649 -0.142630")
print("2 -0.147132  0.154622")
