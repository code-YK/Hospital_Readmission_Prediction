import joblib
import os

os.makedirs("E:\Hospital_Readmission_Prediction\models", exist_ok=True)

def save_model(model, model_name="model.pkl"):
    joblib.dump(model, f"E:\Hospital_Readmission_Prediction\models\{model_name}")
    print(f"Model saved to E:\Hospital_Readmission_Prediction\models\{model_name}")