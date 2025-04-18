import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from lifestyle.daily_life_college_student_project.scripts.model_training import load_processed_data

def load_model(model_name):
    """Load trained model"""
    return joblib.load(f'models/{model_name}.joblib')

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
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_feature_importance.png')
        plt.close()

if __name__ == "__main__":
    print("Loading data and models...")
    _, X_test, _, y_test = load_processed_data()
    
    models = {
        'Random Forest': load_model('random_forest'),
        'XGBoost': load_model('xgboost'),
        'SVM': load_model('svm')
    }
    
    feature_names = [
        'Age', 'Gender', 'Daily_Steps', 'Heart_Rate', 'Sleep_Hours', 
        'BMI', 'Activity_Level', 'Diet_Quality', 'Water_Intake',
        'Steps_per_Hour', 'Sleep_Efficiency'
    ]
    
    print("\nGenerating evaluation plots...")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)
        
        if name != 'SVM':  # SVM doesn't have feature importance
            feature_importance_plot(model, feature_names, name)
    
    print("Evaluation complete! Check the 'models' directory for plots.")