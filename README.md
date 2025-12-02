# Smart Resume-Job Fit Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent ML-powered system that automatically matches resumes to job descriptions using NLP and machine learning, achieving **78.81% accuracy**.

## Features

- **Automated Resume-Job Matching**: TF-IDF, Doc2Vec, and skill-based matching
- **720+ Skills Database**: 520+ technical + 200+ non-technical skills
- **Interactive Dashboard**: Real-time predictions with Streamlit
- **Batch Processing**: Rank multiple candidates simultaneously
- **Actionable Insights**: Matched/missing skills and personalized recommendations
- **Multiple Formats**: Supports PDF and TXT files
- **Fast Inference**: <100ms per prediction

## Project Structure

```
smart_resume_analyser/
├── src/                          # Core modules
│   ├── data_loader.py           # Resume/job data loading
│   ├── data_preprocessing.py    # Text cleaning and preprocessing
│   ├── skill_extractor.py       # 720+ skill extraction
│   ├── feature_engineering.py   # TF-IDF, Doc2Vec, Jaccard features
│   ├── model_training.py        # Random Forest classifier
│   └── utils.py                 # Helper functions
├── scripts/                      # Training pipeline
│   ├── prepare_training_data.py # Creates 4,457 balanced pairs
│   └── train_model.py           # Trains all models
├── dashboard/
│   └── streamlit_app.py         # Interactive web interface
├── data/
│   ├── raw/                     # 1,000 resumes + 10,000 jobs
│   └── processed/               # Training data (4,457 pairs)
├── models/                       # Trained models (~22 MB)
│   ├── tfidf_vectorizer.pkl
│   ├── doc2vec_model.model
│   └── random_forest_model.pkl
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SanyamWadhwa07/smart_resume_analyser.git
cd smart_resume_analyser

# Create virtual environment
python -m venv m

# Activate virtual environment
# Windows PowerShell:
.\m\Scripts\Activate.ps1
# Windows CMD:
.\m\Scripts\activate.bat
# Linux/Mac:
source m/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Train Models (Optional)

Pre-trained models are included. To retrain:

```bash
# Step 1: Prepare training data (creates 4,457 balanced pairs)
python scripts/prepare_training_data.py

# Step 2: Train all models (TF-IDF, Doc2Vec, Random Forest)
python scripts/train_model.py
# Training time: ~3-5 minutes
# Achieves: 78.81% accuracy
```

### 3. Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Access at `http://localhost:8501`

## How It Works

### Analysis Pipeline

1. **Data Loading**: 1,000 resumes + 10,000 LinkedIn job postings
2. **Preprocessing**: Clean text (remove URLs, emails) → tokenize → lemmatize → remove stopwords
3. **Feature Engineering**:
   - **TF-IDF Similarity**: Keyword matching (3,000 features, cosine similarity)
   - **Doc2Vec Similarity**: Semantic embeddings (50-dim vectors, PV-DM algorithm)
   - **Skill Jaccard Index**: Skill overlap using 720+ skill database
4. **ML Prediction**: Random Forest classifier (100 trees, balanced weights)
5. **Output**: Fit score (0-100%), matched/missing skills, recommendations

### Model Architecture

**Input Features (3)**:
- TF-IDF Similarity (0-1)
- Doc2Vec Similarity (0-1)  
- Skill Jaccard Index (0-1)

**Model**: Random Forest Classifier
- n_estimators: 100
- class_weight: balanced
- Training: 3,565 samples (80%)
- Testing: 892 samples (20%)

**Output**: Binary classification (Good Fit / Not a Fit) + probability score

### Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 78.81% |
| **Precision (Good Fit)** | 72% |
| **Recall (Good Fit)** | 65% |
| **F1-Score** | 0.68 |
| **Inference Time** | <100ms |
| **Model Size** | 22 MB |

**Feature Importance**:
- Skill Jaccard: 45%
- TF-IDF: 30%
- Doc2Vec: 25%

## Dataset

- **Resumes**: 1,000 (JSONL format, avg 1,227 chars, 10.2 skills/resume)
- **Job Postings**: 10,000 LinkedIn jobs (9,937 after filtering)
- **Training Pairs**: 4,457 balanced pairs (40% positive, 60% negative)
- **Skills Database**: 720+ skills (520+ technical + 200+ non-technical)

### Skills Categories

**Technical**: Programming (Python, Java, C++), Web (React, Django, Angular), Cloud (AWS, Docker, Kubernetes), Databases (MySQL, MongoDB), ML (TensorFlow, PyTorch), Mobile (iOS, Android), Testing (Selenium, Jest)

**Non-Technical**: Business Management, Sales/Marketing, Finance, HR, Customer Service, Healthcare, Legal, Education, Design

## Usage Examples

### Single Analysis

```python
from src.data_preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ResumeJobMatcher

# Initialize
preprocessor = TextPreprocessor()
feature_engineer = FeatureEngineer()
matcher = ResumeJobMatcher()

# Preprocess
resume_clean = preprocessor.preprocess(resume_text)
job_clean = preprocessor.preprocess(job_text)

# Generate features
features = feature_engineer.generate_features(resume_clean, job_clean)

# Predict
result = matcher.predict_job_fit(features)
print(f"Fit Score: {result['fit_score']}%")
print(f"Matched Skills: {result['matched_skills']}")
print(f"Missing Skills: {result['missing_skills']}")
```

### Batch Processing

```python
# Rank multiple candidates for one job
results = matcher.predict_batch(resumes, job_description)
sorted_candidates = sorted(results, key=lambda x: x['fit_score'], reverse=True)
```

## Dashboard Features

- **Single Analysis**: Upload resume + job → instant predictions
- **Sample Examples**: 6 pre-loaded examples (3 good fits, 3 bad fits)
- **Batch Analysis**: Upload multiple resumes → ranked results table
- **Visualizations**: Gauge charts, radar charts, skill badges
- **Export**: Download results as CSV

## Technologies

- **NLP**: NLTK, spaCy, Gensim
- **ML**: scikit-learn (Random Forest, TF-IDF)
- **Web**: Streamlit
- **Data**: Pandas, NumPy
- **PDF**: PyPDF2

## Documentation

- **ML_LAB_EVALUATION_REPORT.md**: Complete evaluation report (10 pages)
- **PROJECT_DOCUMENTATION.md**: Full technical documentation (29 pages)
- **GitHub**: https://github.com/SanyamWadhwa07/smart_resume_analyser

## Future Enhancements

**Short-term**:
- Education/experience/location matching
- Fuzzy skill matching
- Hyperparameter tuning (GridSearchCV)

**Long-term**:
- BERT/Transformer models
- REST API
- 50K+ training pairs
- Multi-language support

## Troubleshooting

**Issue**: spaCy model not found  
**Solution**: `python -m spacy download en_core_web_sm`

**Issue**: NLTK data not found  
**Solution**: `python -c "import nltk; nltk.download('all')"`

**Issue**: Models not trained  
**Solution**: Run `python scripts/train_model.py`

## License

MIT License

## Contact

**Repository**: https://github.com/SanyamWadhwa07/smart_resume_analyser  
**Author**: Sanyam Wadhwa

---

**Built with Python and Machine Learning**  
**Version**: 1.0 | **Last Updated**: December 2025 | **Accuracy**: 78.81%
