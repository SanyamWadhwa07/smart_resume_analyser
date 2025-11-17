"""
Setup and install script for Smart Resume Analyzer
Installs all required packages and downloads necessary data
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print('='*70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       SMART RESUME ANALYZER - SETUP SCRIPT                   ║
    ║       Installing dependencies and downloading models          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Upgrade pip
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Step 1/5: Upgrading pip"
    )
    
    # Step 2: Install requirements
    run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Step 2/5: Installing Python packages from requirements.txt"
    )
    
    # Step 3: Download spaCy model
    run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Step 3/5: Downloading spaCy English model"
    )
    
    # Step 4: Download NLTK data
    print(f"\n{'='*70}")
    print("  Step 4/5: Downloading NLTK data")
    print('='*70)
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✓ Step 4/5: Downloading NLTK data - SUCCESS")
    except Exception as e:
        print(f"✗ Step 4/5: Downloading NLTK data - FAILED")
        print(f"Error: {e}")
    
    # Step 5: Create necessary directories
    print(f"\n{'='*70}")
    print("  Step 5/5: Creating project directories")
    print('='*70)
    
    dirs = ['models', 'data/processed', 'data/raw']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    print(f"✓ Step 5/5: Creating project directories - SUCCESS")
    
    # Final summary
    print(f"""
    \n{'='*70}
    SETUP COMPLETE!
    {'='*70}
    
    Next steps:
    
    1. Train the models:
       python train_models.py
    
    2. Test the system:
       python test_analyzer.py
    
    3. Launch the dashboard:
       streamlit run dashboard/streamlit_app.py
    
    {'='*70}
    """)

if __name__ == "__main__":
    main()
