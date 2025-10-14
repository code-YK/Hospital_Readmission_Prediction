from sklearn.preprocessing import LabelEncoder
import pandas as pd


def encode_readmitted(df):

    """Binary mapping for target"""
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 0, '<30': 1})
    return df

def encode_medication_columns(df, cols):

    """Ordered Mapping for medication columns"""
    for col in cols:
        df[col] = df[col].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})
    return df


def label_encode_columns(df, cols):

    """Label encode ordinal/nominal columns"""
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def encode_diag(df):

    """One Hot Encode Categorical columns (Diag_cat)"""
    diag_cols = ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']
    df_dummies = pd.get_dummies(df[diag_cols], drop_first=True)
    df_dummies = df_dummies.astype(int) # True -> 1, False -> 0
    df = pd.concat([df, df_dummies], axis=1)
    return df
