import pandas as pd
import numpy as np

def replace_question_mark(df):

    """Replace '?' with NaN"""
    return df.replace('?', np.nan)

def drop_irrelevant_columns(df):
    
    """Drop ID columns and high-null columns"""
    df = df.drop(columns=['encounter_id', 'patient_nbr'])
    if 'weight' in df.columns:
        df = df.drop(columns=['weight'])
    if 'examide' in df.columns:
        df = df.drop(columns=['examide'])
    if 'citoglipton' in df.columns:
        df = df.drop(columns=['citoglipton'])
    return df

def fill_missing_values(df):

    """Impute missing values based on column types"""
    # Low-null categorical
    df['race'] = df['race'].fillna(df['race'].mode()[0])

    # High-null meaningful categorical
    df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
    df['A1Cresult'] = df['A1Cresult'].fillna('None')

    # Nominal categorical
    df['payer_code'] = df['payer_code'].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')

    # Diagnosis columns
    for col in ['diag_1','diag_2','diag_3']:
        df[col] = df[col].fillna('Unknown')

    return df
