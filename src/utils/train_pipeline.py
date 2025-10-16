from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline  # to handle SMOTE + model
from imblearn.over_sampling import SMOTE

# Custom imports
from src.utils.data_loader import load_data, split_X_y, train_test_split_data
from src.utils.evaluater import evaluate_model


def build_pipeline(model, model_type="non-tree", use_smote=True):
    # Choose scaler
    scaler = RobustScaler() if model_type == "tree" else StandardScaler()

    # Create pipeline with or without SMOTE
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ('scaler', scaler),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('scaler', scaler),
            ('model', model)
        ])
    return pipeline


def train_pipeline(data_path, model, model_type="non-tree", use_smote=True):
    """Train and evaluate model using the modular pipeline"""

    # Step 1: Load data
    df = load_data(data_path)

    # Step 2: Split features and target
    X, y = split_X_y(df)

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Step 4: Build pipeline
    pipeline = build_pipeline(model, model_type=model_type, use_smote=use_smote)

    # Step 5: Fit model
    pipeline.fit(X_train, y_train)

    # Step 6: Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline.named_steps['model'], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Step 7: Evaluate
    results = evaluate_model(y_test, y_pred, y_proba, model_name=model.__class__.__name__)
    print(results)

    return pipeline, results, (y_test, y_pred, y_proba)