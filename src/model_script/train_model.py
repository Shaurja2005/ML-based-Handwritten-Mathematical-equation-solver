#This model script isnt being used anymore due to improper performance while detecting the digits and overfiiting issues  , please use the enhanced labelled script only 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
# import os
def train_svm_model(csv_path):
    print(" Loading dataset...")
    df = pd.read_csv(csv_path)
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.\n")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(" Initializing Support Vector Machine...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True, random_state=42)
    
    print("\n Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Average CV Accuracy: {cv_scores.mean() * 100:.2f}%\n")
    
    print(" Training final model on the full training set...")
    svm_model.fit(X_train_scaled, y_train)
    
    print("\n Final Test Set Evaluation:")
    y_pred = svm_model.predict(X_test_scaled)
    print(f"Holdout Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # os.makedirs("models", exist_ok=True)
    joblib.dump(svm_model, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print(" Saved 'svm_model.pkl' and 'scaler.pkl' successfully.")

if __name__ == "__main__":
    CSV_FILE_PATH = r"data\dataset_3x3.csv"
    train_svm_model(CSV_FILE_PATH)