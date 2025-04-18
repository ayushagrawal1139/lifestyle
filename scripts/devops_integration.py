import os
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='devops.log'
)

def git_operations():
    """Perform Git operations for version control"""
    try:
        # Initialize Git if not already
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            logging.info("Git repository initialized")
        
        # Add all files
        subprocess.run(['git', 'add', '.'], check=True)
        logging.info("Files added to Git staging")
        
        # Commit with timestamp
        commit_message = f"Auto commit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        logging.info(f"Git commit performed: {commit_message}")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {e}")
        return False

def setup_ci_cd():
    """Set up basic CI/CD pipeline using GitHub Actions"""
    workflows_dir = '.github/workflows'
    if not os.path.exists(workflows_dir):
        os.makedirs(workflows_dir)
    
    # Create a simple GitHub Actions workflow file
    workflow_content = f"""name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.8
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run data collection
      run: python scripts/data_collection.py
    - name: Run preprocessing
      run: python scripts/data_preprocessing.py
    - name: Run model training
      run: python scripts/model_training.py
    - name: Run evaluation
      run: python scripts/model_evaluation.py
"""
    
    workflow_file = os.path.join(workflows_dir, 'python_ci.yml')
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    logging.info("GitHub Actions workflow created")

def main():
    print("Starting DevOps integration...")
    
    # Perform Git operations
    print("Performing Git operations...")
    if git_operations():
        print("Git operations completed successfully")
    else:
        print("Git operations encountered issues - check devops.log")
    
    # Setup CI/CD
    print("Setting up CI/CD pipeline...")
    setup_ci_cd()
    print("Basic CI/CD pipeline configured using GitHub Actions")
    
    print("\nNext steps:")
    print("1. Create a GitHub repository")
    print("2. Add the remote origin: git remote add origin <repository_url>")
    print("3. Push your code: git push -u origin main")
    print("4. Create a Kaggle account and upload your dataset")
    print("5. Submit your research paper to IEEE")

if __name__ == "__main__":
    main()