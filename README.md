# 🎯 Smart Resume Analyzer# Smart Resume - Job Fit Analyzer

## Complete Code Documentation

An AI-powered resume analysis system that matches resumes to job descriptions and provides actionable insights for improvement.

### File Overview

## ✨ Features

#### 1. data_preprocessing.py

- **🔍 Resume-Job Matching**: Uses TF-IDF, Doc2Vec, and skill extraction for comprehensive matching**Purpose**: Text extraction, cleaning, and preprocessing

- **📊 Detailed Scoring**: Provides multiple similarity metrics and confidence scores**Key Classes**: TextPreprocessor

- **🎯 Skill Analysis**: Identifies matched and missing skills from job requirements**Features**:

- **💡 Smart Recommendations**: Generates actionable improvement suggestions- PDF text extraction (PyPDF2)

- **📈 Interactive Dashboard**: Beautiful Streamlit interface with visualizations- Text cleaning (URLs, emails, special chars removal)

- **📦 Batch Processing**: Analyze multiple resumes against one job description- Tokenization (NLTK)

- **📄 Multiple Formats**: Supports both PDF and DOCX resume formats- Lemmatization

- **🎓 Additional Insights**: Extracts years of experience and education level- Stop word removal

- **🔎 Keyword Highlighting**: Visual highlighting of matched and missing skills

**Usage Example**:

## 🏗️ Project Structure```python

from data_preprocessing import TextPreprocessor

```

smart_resume_analyzer/preprocessor = TextPreprocessor()

├── src/                          # Core source code

│   ├── data_preprocessing.py     # Text extraction and cleaning# From PDF

│   ├── feature_engineering.py    # TF-IDF, Doc2Vec, skill matchingtext = preprocessor.extract_text_from_pdf('resume.pdf')

│   ├── model_training.py         # Random Forest classifier

│   └── utils.py                  # Helper functions# Clean and preprocess

├── dashboard/                    # Web interfacecleaned = preprocessor.preprocess(text)

│   └── streamlit_app.py         # Streamlit dashboard```

├── models/                       # Trained ML models

│   ├── tfidf_vectorizer.pkl     # TF-IDF model (5000 features)#### 2. feature_engineering.py

│   ├── doc2vec_model.model      # Doc2Vec model (200 dimensions)**Purpose**: Feature extraction using NLP techniques

│   └── random_forest_model.pkl  # Random Forest classifier**Key Classes**: FeatureEngineer

├── data/                        # Dataset storage**Features**:

│   ├── raw/                     # Original datasets- TF-IDF vectorization (scikit-learn)

│   └── processed/               # Processed data- Doc2Vec embeddings (gensim)

├── scripts/                     # Utility scripts- Named Entity Recognition for skills (spaCy)

│   ├── train_models.py          # Training pipeline- Cosine similarity calculation

│   ├── create_final_package.py  # Package creator- Jaccard similarity for skills

│   └── system_improvements.py   # System analyzer- 200+ technical skills database

├── tests/                       # Test files

│   ├── test_analyzer.py         # System health check**Usage Example**:

│   └── test_improvements.py     # Feature tests```python

├── docs/                        # Documentationfrom feature_engineering import FeatureEngineer

├── archive/                     # Old/deprecated files

├── final/                       # Production-ready packagefe = FeatureEngineer()

├── requirements.txt             # Python dependencies

└── README.md                    # This file# Train on documents

```fe.fit_tfidf(documents)

fe.train_doc2vec(documents)

## 🚀 Quick Start

# Generate features

### 1. Installationfeatures = fe.generate_features(resume_text, job_text)

# Returns: tfidf_similarity, doc2vec_similarity, skill_jaccard, skill_coverage

```bash```

# Create virtual environment

python -m venv m#### 3. model_training.py

**Purpose**: Machine learning model training and prediction

# Activate virtual environment**Key Classes**: ResumeJobMatcher

.\m\Scripts\Activate.ps1  # Windows PowerShell**Features**:

- Random Forest classifier (scikit-learn)

# Install dependencies- Hyperparameter tuning (RandomizedSearchCV)

pip install -r requirements.txt- Cross-validation

- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)

# Download NLTK data- Model persistence (joblib)

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

**Usage Example**:

# Download spaCy model```python

python -m spacy download en_core_web_smfrom model_training import ResumeJobMatcher

```

matcher = ResumeJobMatcher()

### 2. Run the Dashboard

# Train

```bashX, y = matcher.prepare_training_data(features_df)

streamlit run dashboard/streamlit_app.pymatcher.train_simple(X, y, n_estimators=200)

```

# Predict

The dashboard will open at `http://localhost:8501`result = matcher.predict_job_fit(features)

# Returns: fit_probability, fit_label, confidence

### 3. Train Models (Optional)```



If you want to retrain the models:#### 4. utils.py

**Purpose**: Helper functions and utilities

```bash**Key Functions**:

python scripts/train_models.py- load_resume_dataset()

```- load_job_dataset()

- create_labeled_pairs()

## 📊 How It Works- validate_input_text()

- generate_recommendations()

### Analysis Pipeline- calculate_match_quality()



1. **Text Extraction**: Extracts text from PDF/DOCX resumes#### 5. streamlit_app.py

2. **Preprocessing**: Cleans, tokenizes, and lemmatizes text**Purpose**: Interactive web dashboard

3. **Feature Engineering**:**Features**:

   - TF-IDF vectorization (5000 features)- Single resume analysis

   - Doc2Vec embeddings (200 dimensions)- Batch candidate ranking

   - Skill extraction (100+ technical skills)- Interactive visualizations (Plotly)

   - Skill matching (Jaccard similarity & coverage)- File upload (PDF/TXT)

4. **ML Prediction**: Random Forest classifier predicts job fit- Real-time predictions

5. **Recommendations**: Generates actionable improvement suggestions- Downloadable results



### Scoring Metrics**Run**:

```bash

- **TF-IDF Similarity**: Keyword-based matching (0-100%)streamlit run streamlit_app.py

- **Doc2Vec Similarity**: Semantic understanding (0-100%)```

- **Skill Jaccard**: Skill overlap ratio (0-100%)

- **Skill Coverage**: Percentage of job skills in resume (0-100%)#### 6. train_model.py

- **Job Fit Score**: Overall prediction probability (0-100%)**Purpose**: Main training pipeline script

**Steps**:

## 🎓 Dataset1. Load datasets

2. Create labeled pairs

- **Resumes**: 962 resumes across 25 job categories3. Preprocess texts

- **Job Postings**: 1000+ real job descriptions4. Train TF-IDF and Doc2Vec

- **Training Data**: 10,000 resume-job pairs5. Generate features

- **Skills Database**: 100+ technical skills6. Train Random Forest

7. Save models

## 🧪 Testing

**Run**:

```bash```bash

# Run system health checkpython train_model.py

python tests/test_analyzer.py```



# Run feature tests### Data Flow

python tests/test_improvements.py

```1. **Input**: Resume + Job Description (PDF/TXT)

2. **Preprocessing**: Clean, tokenize, lemmatize

## 📦 Production Package3. **Feature Engineering**:

   - TF-IDF vectors → Cosine similarity

The `final/` folder contains a complete production-ready package:   - Doc2Vec embeddings → Semantic similarity

   - NER → Skill extraction → Jaccard similarity

```bash4. **Model Prediction**: Random Forest → Fit probability

cd final5. **Output**: Fit score, matched/missing skills, recommendations

streamlit run dashboard/streamlit_app.py

```### Model Architecture



## 🛠️ Technologies**Input Features (4)**:

1. TF-IDF Cosine Similarity (0-1)

- **NLP**: NLTK, spaCy, Gensim2. Doc2Vec Semantic Similarity (0-1)

- **ML**: Scikit-learn (Random Forest, TF-IDF)3. Skill Jaccard Similarity (0-1)

- **Web**: Streamlit, Plotly4. Skill Coverage Percentage (0-1)

- **Data**: Pandas, NumPy

- **PDF**: PyPDF2, python-docx**Model**: Random Forest Classifier

- n_estimators: 200-300

## 📈 Performance- max_depth: 15-25

- class_weight: balanced

- **TF-IDF Accuracy**: ~34% average similarity

- **Doc2Vec Accuracy**: ~24% average similarity**Output**:

- **Skill Coverage**: ~63% average coverage- Binary classification (Good Fit / Not Fit)

- **Processing Time**: ~2-3 seconds per resume- Probability score (0-100%)

- **Model Size**: ~50MB total

### Training Data Requirements

## 🎯 Future Improvements

**Minimum**:

See `scripts/system_improvements.py` for analysis of 30+ potential enhancements- 500+ labeled resume-job pairs

- Balanced classes (50% fit, 50% not fit)

## 📝 Documentation

**Recommended**:

Check the `docs/` folder for detailed guides:- 1000+ labeled pairs

- `HOW_TO_RUN.md` - Detailed setup instructions- Diverse job categories

- `SETUP_GUIDE.md` - Configuration guide- Mix of positive and negative examples

- `PROJECT_COMPLETE.md` - Development history

**Label Creation**:

---- Positive (1): Resume category matches job category

- Negative (0): Resume category different from job category

**Built with ❤️ using Python and AI**

### Configuration

**TF-IDF Parameters**:
- max_features: 7000
- ngram_range: (1, 2)
- min_df: 2
- max_df: 0.8

**Doc2Vec Parameters**:
- vector_size: 200
- window: 10
- epochs: 50
- dm: 1 (PV-DM)

**Random Forest Parameters**:
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- class_weight: balanced

### Performance Optimization

**Speed**:
- Cache models (Streamlit session state)
- Use sparse matrices (TF-IDF)
- Batch processing
- Multiprocessing for large datasets

**Accuracy**:
- Increase n_estimators (200→400)
- Hyperparameter tuning
- More training data
- Better skill taxonomy

### Deployment

**Local**:
```bash
streamlit run streamlit_app.py
```

**Streamlit Cloud**:
1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy

**Docker**:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run streamlit_app.py
```

### Troubleshooting

**Issue**: "spaCy model not found"
**Solution**: `python -m spacy download en_core_web_sm`

**Issue**: "NLTK data not found"
**Solution**: `python -c "import nltk; nltk.download('all')"`

**Issue**: "Model not trained"
**Solution**: Run `python train_model.py` first

**Issue**: Low accuracy
**Solution**: 
- Add more training data
- Tune hyperparameters
- Improve skills database
- Check data quality

### Best Practices

1. **Data Quality**: Clean, well-labeled data is crucial
2. **Feature Engineering**: Most important for performance
3. **Model Selection**: Random Forest works well for this task
4. **Validation**: Always use cross-validation
5. **Testing**: Test with real resumes before deployment
6. **Monitoring**: Track prediction quality over time
7. **Updates**: Regularly update skills database

### Future Enhancements

- BERT/Transformer embeddings
- Multi-language support
- Experience level matching
- Education matching
- Location filtering
- Salary prediction
- Interview question generation
- ATS integration

### Contact & Support

For questions or issues, please refer to the README.md or contact the developer.

---
**Version**: 1.0
**Last Updated**: October 2025
**Python**: 3.9+
**License**: MIT
'''

# Save code guide
with open('/tmp/smart_resume_analyzer/CODE_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(code_guide)

print("✓ Saved: CODE_GUIDE.md")

# Create .gitignore
gitignore = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Models
models/*.pkl
models/*.model

# Data
data/raw/*.csv
data/raw/*.pdf
data/processed/*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
predictions.log

# Streamlit
.streamlit/