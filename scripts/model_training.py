import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_processed_data():
    """Load processed data"""
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("Training XGBoost model...")
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def train_svm(X_train, y_train):
    """Train SVM model"""
    print("Training SVM model...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, model_name):
    """Save trained model"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    filename = f'models/{model_name}.joblib'
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print("\nTraining models...")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    
    print("\nEvaluating models...")
    rf_acc, rf_report = evaluate_model(rf_model, X_test, y_test)
    xgb_acc, xgb_report = evaluate_model(xgb_model, X_test, y_test)
    svm_acc, svm_report = evaluate_model(svm_model, X_test, y_test)
    
    print("\nModel Performance:")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    
    # ... (after all model training and evaluation code)

    print("\nSaving models...")
    save_model(rf_model, "random_forest")
    save_model(xgb_model, "xgboost")
    save_model(svm_model, "svm")
    
    # Add visualization code here
    import matplotlib.pyplot as plt
    
    models = ['Random Forest', 'XGBoost', 'SVM']
    accuracies = [rf_acc, xgb_acc, svm_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nModel training complete!")