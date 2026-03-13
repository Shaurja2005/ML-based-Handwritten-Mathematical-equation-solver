"""
Enhanced Model Training with Hyperparameter Tuning.

Improvements:
1. GridSearchCV for optimal C, gamma
2. Multiple classifier comparison (SVM, RF, GradientBoosting)
3. Confusion matrix analysis
4. Feature importance (for tree-based models)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time


def print_confusion_pairs(y_test, y_pred, classes):
    """Print the most confused class pairs."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Find off-diagonal pairs with highest confusion
    confusions = []
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], true_class, pred_class))
    
    confusions.sort(reverse=True)
    
    print("\nTop Confusion Pairs (True → Predicted):")
    print("-" * 40)
    for count, true_c, pred_c in confusions[:10]:
        print(f"  {true_c:>8} → {pred_c:<8} : {count} times")


def train_with_grid_search(X_train, y_train, X_test, y_test):
    """
    Train SVM with GridSearchCV for hyperparameter tuning.
    """
    print("\n" + "=" * 60)
    print("SVM with GridSearchCV")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 50.0],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf']
    }
    
    print(f"Searching parameters: {param_grid}")
    print("This may take a few minutes...\n")
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train_scaled, y_train)
    elapsed = time.time() - start
    
    print(f"\nGrid search completed in {elapsed:.1f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print_confusion_pairs(y_test, y_pred, best_model.classes_)
    
    return best_model, scaler, test_acc


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest (no scaling needed).
    """
    print("\n" + "=" * 60)
    print("Random Forest Classifier")
    print("=" * 60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    print(f"Searching parameters: {param_grid}")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"\nGrid search completed in {elapsed:.1f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_names = X_train.columns.tolist()
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    print("\nTop 15 Important Features:")
    for i in indices:
        print(f"  {feature_names[i]:20s}: {importances[i]:.4f}")
    
    print_confusion_pairs(y_test, y_pred, best_model.classes_)
    
    return best_model, None, test_acc


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train Gradient Boosting Classifier.
    """
    print("\n" + "=" * 60)
    print("Gradient Boosting Classifier")
    print("=" * 60)
    
    # Lighter grid for GB (it's slower)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    print(f"Searching parameters: {param_grid}")
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"\nGrid search completed in {elapsed:.1f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print_confusion_pairs(y_test, y_pred, best_model.classes_)
    
    return best_model, None, test_acc


def main():
    CSV_PATH = r"data\dataset_enhanced.csv"
    
    print("Loading enhanced dataset...")
    df = pd.read_csv(CSV_PATH)
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    print(f"Samples: {len(df)}, Features: {X.shape[1]}, Classes: {y.nunique()}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # --- Train all models ---
    svm_model, svm_scaler, svm_acc = train_with_grid_search(X_train, y_train, X_test, y_test)
    results['SVM'] = (svm_model, svm_scaler, svm_acc)
    
    rf_model, _, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    results['RF'] = (rf_model, None, rf_acc)
    
    gb_model, _, gb_acc = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['GB'] = (gb_model, None, gb_acc)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for name, (_, _, acc) in results.items():
        print(f"  {name:20s}: {acc * 100:.2f}%")
    
    # --- Save best model ---
    best_name = max(results, key=lambda k: results[k][2])
    best_model, best_scaler, best_acc = results[best_name]
    
    print(f"\nBest model: {best_name} ({best_acc * 100:.2f}%)")
    
    joblib.dump(best_model, "models/best_model.pkl")
    print("Saved: models/best_model.pkl")
    
    if best_scaler is not None:
        joblib.dump(best_scaler, "models/best_scaler.pkl")
        print("Saved: models/best_scaler.pkl")
    
    # Also save SVM specifically (since backend uses it)
    joblib.dump(svm_model, "models/svm_model_enhanced.pkl")
    joblib.dump(svm_scaler, "models/scaler_enhanced.pkl")
    print("Saved: models/svm_model_enhanced.pkl, models/scaler_enhanced.pkl")


if __name__ == "__main__":
    main()
