import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    print("Starting data preprocessing...")
    
    # 1. Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    print("Verified directory structure exists")

    # 2. Load data with validation
    try:
        raw_path = 'data/raw/fitness_data.xlsx'
        print(f"Loading data from {raw_path}")
        df = pd.read_excel(raw_path)
        print(f"Successfully loaded data with shape {df.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. Data preprocessing
    try:
        # Handle categorical data
        df['Gender'] = df['Gender'].map({'Male':0, 'Female':1, 'Other':2})
        df['Activity_Level'] = df['Activity_Level'].map({
            'Sedentary':0, 
            'Moderate':1, 
            'Active':2, 
            'Athlete':3
        })
        
        # Feature engineering
        df['Steps_per_Hour'] = df['Daily_Steps'] / 24
        df['Sleep_Efficiency'] = df['Sleep_Hours'] / df['Stress_Level']
        df['Stress_Level'] = df['Stress_Level'] - 1  # Convert 1-5 range to 0-4
        
        # Split data
        X = df.drop('Stress_Level', axis=1)
        y = df['Stress_Level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 4. Save processed data
        np.save('/Users/ayush_agrawal2004/Documents/student_life_generation/lifestyle/daily_life_college_student_project/data/processed/X_train.npy', X_train)
        np.save('/Users/ayush_agrawal2004/Documents/student_life_generation/lifestyle/daily_life_college_student_project/data/processed/X_test.npy', X_test)
        np.save('/Users/ayush_agrawal2004/Documents/student_life_generation/lifestyle/daily_life_college_student_project/data/processed/y_train.npy', y_train)
        np.save('/Users/ayush_agrawal2004/Documents/student_life_generation/lifestyle/daily_life_college_student_project/data/processed/y_test.npy', y_test)
        joblib.dump(scaler, '/Users/ayush_agrawal2004/Documents/student_life_generation/lifestyle/daily_life_college_student_project/data/processed/scaler.joblib')
        
        print("Successfully saved processed data:")
        print(os.listdir('data/processed'))
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

if __name__ == "__main__":
    main()