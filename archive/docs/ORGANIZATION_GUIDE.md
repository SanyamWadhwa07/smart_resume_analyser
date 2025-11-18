# 📁 Directory Organization Guide

## Clean Structure Overview

```
smart_resume_analyzer/
├── 📂 src/                    # Core application code
├── 📂 dashboard/              # Web interface
├── 📂 models/                 # Trained ML models
├── 📂 data/                   # Datasets
├── 📂 scripts/                # Utility scripts
├── 📂 tests/                  # Test files
├── 📂 docs/                   # Documentation
├── 📂 archive/                # Old/deprecated files
├── 📂 final/                  # Production package
├── 📂 m/                      # Virtual environment
├── 📂 nltk_data/              # NLTK data files
├── 📄 README.md               # Main documentation
├── 📄 requirements.txt        # Dependencies
└── 📄 .gitignore             # Git ignore rules
```

## Folder Descriptions

### Core Application (`src/`)
**Contains**: Main application source code
- `data_preprocessing.py` - Text extraction, cleaning, skill extraction
- `feature_engineering.py` - TF-IDF, Doc2Vec, feature generation
- `model_training.py` - Random Forest classifier
- `utils.py` - Helper functions and utilities

### Web Interface (`dashboard/`)
**Contains**: Streamlit web application
- `streamlit_app.py` - Interactive dashboard with visualizations

### Models (`models/`)
**Contains**: Trained machine learning models
- `tfidf_vectorizer.pkl` - TF-IDF model (5000 features)
- `doc2vec_model.model` - Doc2Vec embeddings (200D)
- `random_forest_model.pkl` - Random Forest classifier

### Data (`data/`)
**Contains**: Training and test datasets
- `raw/` - Original datasets (resumes, job postings)
- `processed/` - Processed data for analysis

### Scripts (`scripts/`)
**Contains**: Utility and maintenance scripts
- `train_models.py` - Complete model training pipeline
- `create_final_package.py` - Production package creator
- `system_improvements.py` - System analysis tool
- `setup.py` - Setup automation

### Tests (`tests/`)
**Contains**: Test and validation scripts
- `test_analyzer.py` - System health check and validation
- `test_improvements.py` - Feature testing suite

### Documentation (`docs/`)
**Contains**: Project documentation
- `HOW_TO_RUN.md` - Setup and running instructions
- `SETUP_GUIDE.md` - Detailed configuration guide
- `PROJECT_COMPLETE.md` - Development history
- `FINAL_PACKAGE_SUMMARY.md` - Package details
- `SYSTEM_LIVE.md` - System status
- `DIRECTORY_STRUCTURE.txt` - Complete file tree

### Archive (`archive/`)
**Contains**: Deprecated/old files (kept for reference)
- `analyze_data.py` - Old analysis script
- `fix_nltk.py` - NLTK setup fix script

### Production (`final/`)
**Contains**: Complete production-ready package
- Self-contained deployment package with all dependencies
- Includes: code, models, sample data, documentation

## File Organization Rules

### ✅ Keep in Root
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusion rules

### ✅ Keep in `src/`
- All core application Python modules
- Only production-ready code

### ✅ Keep in `scripts/`
- Training pipelines
- Setup automation
- Data generation
- System utilities

### ✅ Keep in `tests/`
- Unit tests
- Integration tests
- System validation scripts

### ✅ Keep in `docs/`
- All markdown documentation
- Guides and tutorials
- API documentation

### ❌ Move to `archive/`
- Old/deprecated code
- Temporary fix scripts
- Experimental code
- One-time use scripts

## Quick Commands

### Run Application
```bash
streamlit run dashboard/streamlit_app.py
```

### Train Models
```bash
python scripts/train_models.py
```

### Run Tests
```bash
python tests/test_analyzer.py
python tests/test_improvements.py
```

### Create Package
```bash
python scripts/create_final_package.py
```

### Analyze System
```bash
python scripts/system_improvements.py
```

## Maintenance

### Weekly
- Clean temporary files
- Update documentation
- Run test suite

### Monthly
- Review archive folder
- Update dependencies
- Retrain models with new data

### When Adding New Code
1. Place in appropriate folder
2. Update README if needed
3. Add tests to `tests/`
4. Update documentation

## Clean-Up Checklist

- [x] Moved test files to `tests/`
- [x] Moved scripts to `scripts/`
- [x] Moved documentation to `docs/`
- [x] Moved old files to `archive/`
- [x] Created `.gitignore`
- [x] Updated `README.md`
- [x] Kept only essentials in root
- [x] Organized by function
- [x] Clear naming conventions
- [x] Documented structure

## Best Practices

1. **Keep Root Clean**: Only essential config files
2. **Organize by Purpose**: Group related files together
3. **Clear Names**: Descriptive folder and file names
4. **Document Changes**: Update docs when reorganizing
5. **Regular Clean-up**: Monthly review of file structure
6. **Archive Don't Delete**: Move old files to archive
7. **Test Organization**: Ensure all paths still work

---

**Last Organized**: November 15, 2025
**Status**: ✅ Clean and Organized
