from src.data_cleaning import replace_question_mark, drop_irrelevant_columns, fill_missing_values
from src.encoding import encode_readmitted, encode_medication_columns, label_encode_columns, encode_diag
from src.feature_engineering import create_medication_features, diag_categorization, bin_age
from src.outlier_transform import log_transform_columns

def preprocess(df, medication_cols, label_encode_cols, log_cols):
    # Step 1: Replace '?' and drop irrelevant columns
    df = replace_question_mark(df)
    df = drop_irrelevant_columns(df)
    
    # Step 2: Fill missing values
    df = fill_missing_values(df)
    
    # Step 3: Encode target and categorical features
    df = encode_readmitted(df)
    df = encode_medication_columns(df, medication_cols)
    df = encode_diag(diag_categorization(df))
    df = label_encode_columns(df, label_encode_cols)

    # Step 4: Create derived medication features
    df = create_medication_features(df, medication_cols)
    df = bin_age(df, 'age')

    # Step 5: Log transform numeric columns
    df = log_transform_columns(df, log_cols)
    
    return df
