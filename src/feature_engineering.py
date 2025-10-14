import pandas as pd
import numpy as np

def create_medication_features(df, medication_columns):
    """Derived features for medications"""
    # Total medications (non-No)
    df['total_medications'] = df[medication_columns].apply(lambda x: x.ne('No').sum(), axis=1)

    # Number of medication changes
    df['num_med_changes'] = df[medication_columns].apply(lambda x: x.isin(['Up','Down','Steady']).sum(), axis=1)

    return df


def diag_mapping(diag):
    if pd.isna(diag) or diag == '?':
        return 'Unknown'
    diag = str(diag)
    if diag.startswith('250'):
        return 'Diabetes'
    elif diag.startswith('414'):
        return 'Heart Disease'
    elif diag.startswith('428'):
        return 'Heart Failure'
    elif diag.startswith('401'):
        return 'Hypertension'
    else:
        return 'Other'


def diag_categorization(df):
    """Categorize diagnosis codes"""
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col + '_cat'] = df[col].apply(diag_mapping)
    return df


def bin_age(df, age_col):
    """Convert age range to median age"""
    age_dict = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    df['numeric_age'] = df[age_col].map(age_dict)
    return df