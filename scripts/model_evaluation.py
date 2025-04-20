import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_model(model_name):
    """Load trained model"""
    return joblib.load(f'models/{model_name}.joblib')

def load_processed_data():
    """Load processed data"""
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'models/{model_name}_confusion_matrix.png')
    plt.close()

def feature_importance_plot(model, feature_names, model_name):
    """Plot feature importance"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print(f"\nDebug - {model_name}:")
            print(f"Number of features in model: {len(importances)}")
            print(f"Number of feature names: {len(feature_names)}")
            
            # Get matching subset of features and names
            n_features = min(len(importances), len(feature_names))
            indices = np.argsort(importances)[::-1][:n_features]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(n_features), importances[indices], align='center')
            plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/{model_name}_feature_importance.png')
            plt.close()
            print(f"Successfully generated feature importance plot for {model_name}")
    except Exception as e:
        print(f"Error in feature_importance_plot for {model_name}: {str(e)}")

if __name__ == "__main__":
    print("Loading data and models...")
    _, X_test, _, y_test = load_processed_data()
    
    # Debug: Print shape of test data
    print(f"\nDebug - X_test shape: {X_test.shape}")
    
    # Get actual feature names from preprocessing
    # Update this list to exactly match your preprocessing output
    feature_names = [
        'Age', 'Gender', 'Daily_Steps', 'Heart_Rate', 'Sleep_Hours', 
        'BMI', 'Activity_Level', 'Diet_Quality', 'Water_Intake',
        'Steps_per_Hour', 'Sleep_Efficiency'
    ]
    
    models = {
        'Random Forest': load_model('random_forest'),
        'XGBoost': load_model('xgboost'),
        'SVM': load_model('svm')
    }
    
    print("\nGenerating evaluation plots...")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)
        
        if name != 'SVM':  # SVM doesn't have feature importance
            feature_importance_plot(model, feature_names, name)
    
    print("\nEvaluation complete! Check the 'models' directory for plots.")