# 🚀 Quick Command Reference

**Last Updated**: November 15, 2025 (After Directory Cleanup)

## Common Tasks

### 🎯 Running the Application

#### Start Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```
Opens at: `http://localhost:8501`

---

## 🔧 Development & Training

#### Train Models from Scratch
```bash
python scripts/train_models.py
```
- Loads 962 resumes + 1000+ jobs
- Creates 10,000 training pairs
- Trains TF-IDF, Doc2Vec, Random Forest
- Saves models to `models/` folder
- Takes ~5-10 minutes

#### Analyze System for Improvements
```bash
python scripts/system_improvements.py
```
- Scans codebase
- Identifies 30+ improvement opportunities
- Categorizes by priority
- Estimates implementation time

#### Create Production Package
```bash
python scripts/create_final_package.py
```
- Creates `final/` folder
- Copies all essential files
- Includes models, data, docs
- ~50MB package size

---

## 🧪 Testing & Validation

#### Run System Health Check
```bash
python tests/test_analyzer.py
```
- Tests all components
- Validates models loaded
- Checks feature extraction
- Reports accuracy metrics

#### Test New Features
```bash
python tests/test_improvements.py
```
- Tests DOCX support
- Tests experience extraction
- Tests education detection
- Tests keyword highlighting
- Tests comparison generation

---

## 📦 Environment Setup

#### Create Virtual Environment
```powershell
python -m venv m
```

#### Activate Environment (Windows PowerShell)
```powershell
.\m\Scripts\Activate.ps1
```

#### Activate Environment (Windows CMD)
```cmd
.\m\Scripts\activate.bat
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

---

## 📊 Data Management

#### View Resume Dataset
```bash
python
>>> import pandas as pd
>>> df = pd.read_csv('data/raw/UpdatedResumeDataSet.csv')
>>> df.head()
>>> df['Category'].value_counts()
```

#### View Job Postings
```bash
python
>>> import pandas as pd
>>> jobs = pd.read_csv('data/raw/jobPostings/postings.csv')
>>> jobs.head()
```

---

## 🔍 File Locations (Updated)

### Source Code
```
src/data_preprocessing.py    # Text extraction & cleaning
src/feature_engineering.py   # TF-IDF, Doc2Vec, skills
src/model_training.py         # Random Forest classifier
src/utils.py                  # Helper functions
```

### Scripts (Utilities)
```
scripts/train_models.py           # Training pipeline
scripts/create_final_package.py   # Package creator
scripts/system_improvements.py    # System analyzer
scripts/setup.py                  # Setup automation
```

### Tests
```
tests/test_analyzer.py        # System health check
tests/test_improvements.py    # Feature tests
```

### Documentation
```
docs/HOW_TO_RUN.md               # Running instructions
docs/SETUP_GUIDE.md              # Configuration guide
docs/PROJECT_COMPLETE.md         # Development history
docs/ORGANIZATION_GUIDE.md       # Directory guide
docs/CLEANUP_SUMMARY.md          # Cleanup details
docs/FINAL_PACKAGE_SUMMARY.md    # Package info
docs/SYSTEM_LIVE.md              # System status
```

### Models
```
models/tfidf_vectorizer.pkl      # TF-IDF model
models/doc2vec_model.model       # Doc2Vec model
models/random_forest_model.pkl   # RF classifier
```

---

## 💡 Usage Examples

### Analyze Single Resume (Python)
```python
import sys
sys.path.append('src')

from data_preprocessing import TextPreprocessor, extract_skills
from feature_engineering import FeatureEngineer
from model_training import ResumeJobMatcher

# Initialize
preprocessor = TextPreprocessor()
fe = FeatureEngineer()
matcher = ResumeJobMatcher()

# Load models
fe.load_models('models')
matcher.load_model('models/random_forest_model.pkl')

# Analyze
resume_text = preprocessor.extract_text_from_pdf('resume.pdf')
job_text = "Python developer with ML experience needed..."

features = fe.generate_features(resume_text, job_text)
prediction = matcher.predict_job_fit(features)

print(f"Job Fit: {prediction['fit_probability']:.1%}")
print(f"Label: {prediction['fit_label']}")
```

### Extract Skills from Text
```python
from src.data_preprocessing import extract_skills

text = "Expert in Python, Java, Docker, AWS, and machine learning"
skills = extract_skills(text)
print(skills)  # ['python', 'java', 'docker', 'aws', 'machine learning']
```

### Check Experience & Education
```python
from src.data_preprocessing import TextPreprocessor

prep = TextPreprocessor()
resume = "PhD in CS with 5 years of experience in AI"

years = prep.extract_years_of_experience(resume)
education = prep.extract_education_level(resume)

print(f"Experience: {years} years")  # 5 years
print(f"Education: {education}")     # phd
```

---

## 🐛 Troubleshooting

### Dashboard Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :8501

# Kill process if needed
taskkill /PID <PID> /F

# Restart dashboard
streamlit run dashboard/streamlit_app.py
```

### Models Not Found
```bash
# Check models exist
dir models

# If missing, train them
python scripts/train_models.py
```

### Import Errors
```python
# Ensure src is in path
import sys
sys.path.append('src')
```

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('all')"
```

---

## 📝 Git Commands

### Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit - Smart Resume Analyzer"
```

### Create .gitignore (Already Created)
Already exists with proper exclusions for:
- `__pycache__/`
- `m/` (virtual environment)
- `models/*.pkl` (large files)
- `nltk_data/`

### Push to GitHub
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

---

## 🎯 Production Deployment

### Using the Production Package
```bash
cd final
streamlit run dashboard/streamlit_app.py
```

### Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Select `dashboard/streamlit_app.py`
5. Deploy

---

## 📊 Performance Monitoring

### Check Model Performance
```bash
python tests/test_analyzer.py
```

Expected Output:
- TF-IDF: ~34% similarity
- Doc2Vec: ~24% similarity
- Skills: ~63% coverage

### Run All Tests
```bash
python tests/test_analyzer.py
python tests/test_improvements.py
```

---

## 🔄 Maintenance Tasks

### Weekly
- Run tests to verify system health
- Check for updates to dependencies

### Monthly
- Retrain models with new data
- Review and archive old files
- Update documentation

### When Needed
- Add new skills to database
- Update feature engineering
- Improve recommendations

---

**Quick Reference Status**: ✅ Updated
**Last Tested**: November 15, 2025
**All Commands Verified**: ✅ Working
