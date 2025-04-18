import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_fitness_data():
    """Generate synthetic fitness data with 1000 records and 10 attributes"""
    np.random.seed(42)
    
    data = {
        'Age': np.random.randint(18, 70, 1000),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], 1000, p=[0.48, 0.48, 0.04]),
        'Daily_Steps': np.random.randint(1000, 25000, 1000),
        'Heart_Rate': np.random.randint(50, 100, 1000),
        'Sleep_Hours': np.round(np.random.uniform(3, 12, 1000), 1),
        'BMI': np.round(np.random.uniform(15, 40, 1000), 1),
        'Activity_Level': np.random.choice(['Sedentary', 'Moderate', 'Active', 'Athlete'], 1000),
        'Stress_Level': np.random.randint(1, 6, 1000),
        'Diet_Quality': np.random.randint(1, 6, 1000),
        'Water_Intake': np.round(np.random.uniform(1, 5, 1000), 1),
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(1000)]
    }
    
    return pd.DataFrame(data)

def save_data(df, filename):
    """Save data to Excel file"""
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    
    filepath = os.path.join('data/raw', filename)
    df.to_excel(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    print("Generating fitness dataset...")
    fitness_df = generate_fitness_data()
    save_data(fitness_df, "fitness_data.xlsx")
    print("Dataset generation complete!")