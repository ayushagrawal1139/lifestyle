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
        'Activity_Level': np.random.choice(
            ['Sedentary', 'Moderate', 'Active', 'Athlete'], 
            1000, 
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        'Stress_Level': np.random.randint(1, 6, 1000),
        'Diet_Quality': np.random.randint(1, 6, 1000),
        'Water_Intake': np.round(np.random.uniform(1, 5, 1000), 1),
        'Fitness_Score': np.round(np.random.uniform(1, 10, 1000), 1)  # Target variable
    }
    
    return pd.DataFrame(data)

def save_data(df, filename):
    """Save dataframe to Excel file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_excel(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Generate and save data
    df = generate_fitness_data()
    save_data(df, "data/raw/fitness_data.xlsx")