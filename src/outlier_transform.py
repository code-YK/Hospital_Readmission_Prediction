import numpy as np

def log_transform_columns(df, cols):
    """Apply log1p transformation to numeric columns"""
    for col in cols:
        df[col] = np.log1p(df[col])
    return df
