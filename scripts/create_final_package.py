"""
Create Final Deployment Package
Organizes all important files into a clean 'final' directory
"""

import os
import shutil
from pathlib import Path

def create_final_package():
    print("=" * 80)
    print("CREATING FINAL DEPLOYMENT PACKAGE")
    print("=" * 80)
    
    # Create final directory structure
    final_dir = Path("final")
    final_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "src",
        "dashboard", 
        "models",
        "data/sample",
        "docs",
        "scripts"
    ]
    
    for subdir in subdirs:
        (final_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("\n[1/7] Creating directory structure...")
    print("✓ Created final/ directory")
    
    # Copy source files
    print("\n[2/7] Copying source code...")
    src_files = [
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/utils.py"
    ]
    
    for file in src_files:
        if os.path.exists(file):
            shutil.copy2(file, final_dir / "src")
            print(f"  ✓ Copied {file}")
    
    # Copy utility scripts
    print("\n[3/7] Copying utility scripts...")
    script_files = [
        "train_models.py",
        "test_analyzer.py",
        "analyze_data.py",
        "setup.py"
    ]
    
    for file in script_files:
        if os.path.exists(file):
            shutil.copy2(file, final_dir / "scripts")
            print(f"  ✓ Copied {file}")
    
    # Copy dashboard
    print("\n[4/7] Copying dashboard...")
    if os.path.exists("dashboard/streamlit_app.py"):
        shutil.copy2("dashboard/streamlit_app.py", final_dir / "dashboard")
        print("  ✓ Copied dashboard/streamlit_app.py")
    
    # Copy models
    print("\n[5/7] Copying trained models...")
    model_files = [
        "models/tfidf_vectorizer.pkl",
        "models/doc2vec_model.model",
        "models/random_forest_model.pkl"
    ]
    
    for file in model_files:
        if os.path.exists(file):
            shutil.copy2(file, final_dir / "models")
            print(f"  ✓ Copied {file}")
    
    # Copy sample data
    print("\n[6/7] Copying sample data...")
    if os.path.exists("data/raw/UpdatedResumeDataSet.csv"):
        # Copy first 100 rows as sample
        import pandas as pd
        df = pd.read_csv("data/raw/UpdatedResumeDataSet.csv", nrows=100)
        df.to_csv(final_dir / "data/sample/sample_resumes.csv", index=False)
        print("  ✓ Created sample_resumes.csv (100 rows)")
    
    # Create requirements.txt
    print("\n[7/7] Creating deployment files...")
    
    requirements = """# Core Dependencies
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0

# NLP Libraries
nltk>=3.8
spacy>=3.5.0
gensim>=4.3.0

# PDF Processing
PyPDF2>=3.0.0

# Web Framework
streamlit>=1.28.0
plotly>=5.17.0

# Utilities
joblib>=1.3.0
"""
    
    with open(final_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    print("  ✓ Created requirements.txt")
    
    # Create README
    readme = """# Smart Resume Analyzer - Final Package

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

3. Run the application:
```bash
streamlit run dashboard/streamlit_app.py
```

## What's Included

- `src/` - Core Python modules (preprocessing, feature engineering, ML)
- `dashboard/` - Streamlit web interface
- `models/` - Pre-trained AI models (TF-IDF, Doc2Vec, Random Forest)
- `scripts/` - Utility scripts (train, test, analyze)
- `data/sample/` - Sample resume data for testing
- `docs/` - Documentation

## Utility Scripts

### scripts/test_analyzer.py
Quick test to verify the system is working:
```bash
python scripts/test_analyzer.py
```

### scripts/train_models.py
Retrain all models with new data:
```bash
python scripts/train_models.py
```

### scripts/analyze_data.py
Analyze your dataset structure:
```bash
python scripts/analyze_data.py
```

## System Features

- PDF resume upload and parsing
- AI-powered job fit scoring (0-100%)
- Skill gap analysis
- Batch candidate ranking
- Visual analytics and recommendations

## Support

For detailed instructions, see the documentation files in `docs/` folder.
"""
    
    with open(final_dir / "README.md", "w") as f:
        f.write(readme)
    print("  ✓ Created README.md")
    
    # Create run script
    run_script = """# Run Smart Resume Analyzer

Write-Host "Starting Smart Resume Analyzer..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run dashboard/streamlit_app.py
"""
    
    with open(final_dir / "run.ps1", "w") as f:
        f.write(run_script)
    print("  ✓ Created run.ps1 (Windows PowerShell)")
    
    # Create setup script
    setup_script = """# Setup Script for Smart Resume Analyzer

Write-Host "Installing Python packages..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host ""
Write-Host "Downloading spaCy model..." -ForegroundColor Green
python -m spacy download en_core_web_sm

Write-Host ""
Write-Host "Downloading NLTK data..." -ForegroundColor Green
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

Write-Host ""
Write-Host "Setup complete! Run 'run.ps1' to start the application." -ForegroundColor Green
"""
    
    with open(final_dir / "setup.ps1", "w") as f:
        f.write(setup_script)
    print("  ✓ Created setup.ps1 (Installation script)")
    
    # Copy documentation
    print("\n[8/8] Copying documentation...")
    doc_files = [
        "HOW_TO_RUN.md",
        "SYSTEM_LIVE.md",
        "PROJECT_COMPLETE.md"
    ]
    
    for file in doc_files:
        if os.path.exists(file):
            shutil.copy2(file, final_dir / "docs")
            print(f"  ✓ Copied {file}")
    
    # Create deployment info
    deployment_info = f"""# Deployment Package Info

**Created**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Package**: Smart Resume Analyzer v1.0

## Contents

### Source Code (src/)
- data_preprocessing.py - PDF parsing, text cleaning, skill extraction
- feature_engineering.py - TF-IDF, Doc2Vec, NLP features
- model_training.py - Random Forest classifier
- utils.py - Helper functions

### Utility Scripts (scripts/)
- train_models.py - Retrain all models with new data
- test_analyzer.py - Quick system health check
- analyze_data.py - Analyze dataset structure
- setup.py - Dependency installer

### Models (models/)
- tfidf_vectorizer.pkl - 5000-feature TF-IDF model
- doc2vec_model.model - 200D semantic embeddings
- random_forest_model.pkl - ML classifier

### Dashboard (dashboard/)
- streamlit_app.py - Interactive web interface

### Data (data/sample/)
- sample_resumes.csv - 100 sample resumes for testing

### Scripts
- setup.ps1 - One-click installation
- run.ps1 - Start the application
- requirements.txt - Python dependencies

### Documentation (docs/)
- HOW_TO_RUN.md - Usage instructions
- SYSTEM_LIVE.md - System overview
- PROJECT_COMPLETE.md - Project details

## Package Size
Total files: ~30
Models: ~50MB
Ready for deployment!

## Next Steps

1. Run setup.ps1 to install dependencies
2. Run run.ps1 to start the application
3. Open http://localhost:8501 in browser
4. Upload resume PDF and job description
5. Get AI-powered analysis!
"""
    
    with open(final_dir / "PACKAGE_INFO.md", "w") as f:
        f.write(deployment_info)
    print("  ✓ Created PACKAGE_INFO.md")
    
    # Create package summary
    print("\n" + "=" * 80)
    print("FINAL PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\nPackage Location: ./final/")
    print("\nContents:")
    print("  📁 src/ - Core Python modules")
    print("  📁 scripts/ - Utility scripts (train, test, analyze)")
    print("  📁 dashboard/ - Web interface")
    print("  📁 models/ - Pre-trained AI models")
    print("  📁 data/sample/ - Sample data")
    print("  📁 docs/ - Documentation")
    print("  📄 requirements.txt - Dependencies")
    print("  📄 README.md - Quick start guide")
    print("  📄 setup.ps1 - Installation script")
    print("  📄 run.ps1 - Run script")
    print("  📄 PACKAGE_INFO.md - Package details")
    
    print("\n" + "=" * 80)
    print("TO DEPLOY:")
    print("=" * 80)
    print("\n1. Copy the 'final' folder to your deployment location")
    print("2. Run: .\\final\\setup.ps1")
    print("3. Run: .\\final\\run.ps1")
    print("4. Access: http://localhost:8501")
    print("\n" + "=" * 80)
    
    # Get package size
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(final_dir):
        for file in files:
            fp = os.path.join(root, file)
            total_size += os.path.getsize(fp)
            file_count += 1
    
    print(f"\nPackage Stats:")
    print(f"  Files: {file_count}")
    print(f"  Size: {total_size / (1024*1024):.1f} MB")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import pandas as pd
    create_final_package()
