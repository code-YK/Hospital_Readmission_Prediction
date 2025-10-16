from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_X_y(df):
    # Drop non-feature columns

    X = df.drop(columns=['readmitted', 'age', 'diag_1', 'diag_2', 'diag_3', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat'], errors='ignore')
    y = df['readmitted']
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
