# 📄 Smart Resume Analyzer - Complete Setup Guide

An AI-powered resume screening system that analyzes how well a resume matches a job description.

## 🎯 What This Does

Takes a **PDF resume** and a **job description** as input, then provides:
- **Job Fit Score** (0-100%)
- **Matched Skills** (what you have)
- **Missing Skills** (what you need)
- **Recommendations** (how to improve)

## 🚀 Quick Start (5 Minutes)

### Step 1: Activate Virtual Environment
```powershell
.\m\Scripts\Activate.ps1
```

### Step 2: Install Dependencies (if not done)
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3: Train Models
```powershell
python train_models.py
```
This takes 2-5 minutes and creates the AI models.

### Step 4: Test It Works
```powershell
python test_analyzer.py
```
This runs a quick test with sample data.

### Step 5: Launch Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```
Opens in your browser at http://localhost:8501

## 📊 Using the Dashboard

### Single Analysis (One Resume vs One Job)

1. **Upload Resume**:
   - Click "Upload Resume (PDF/TXT)"
   - Or paste text directly

2. **Upload Job Description**:
   - Click "Upload Job Description (PDF/TXT)"
   - Or paste text directly

3. **Click "Analyze Job Fit"**

4. **View Results**:
   - 📊 Job Fit Score gauge (0-100%)
   - ✅ Matched skills (green badges)
   - ❌ Missing skills (red badges)
   - 💡 Recommendations for improvement

### Batch Analysis (Multiple Resumes)

1. **Upload ONE job description**
2. **Upload MULTIPLE resumes** (select many files)
3. **Click "Rank Candidates"**
4. **Download rankings as CSV**

## 🛠️ Project Structure

```
smart_resume_analyzer/
│
├── dashboard/
│   └── streamlit_app.py       # Web interface (Streamlit)
│
├── src/
│   ├── data_preprocessing.py  # Text cleaning & PDF reading
│   ├── feature_engineering.py # TF-IDF, Doc2Vec, skill matching
│   ├── model_training.py      # Random Forest ML model
│   └── utils.py               # Helper functions
│
├── data/
│   └── raw/                   # Your resume & job data
│
├── models/                    # Saved AI models (auto-created)
│   ├── tfidf_vectorizer.pkl
│   ├── doc2vec_model.model
│   └── random_forest_model.pkl
│
├── train_models.py            # Main training script
├── test_analyzer.py           # Quick test script
└── requirements.txt           # Python packages
```

## 🧠 How It Works

### 1. Text Processing
- Reads PDF files
- Removes emails, phone numbers, URLs
- Cleans and normalizes text
- Tokenizes and lemmatizes

### 2. Feature Extraction
- **TF-IDF**: Finds keyword similarity
- **Doc2Vec**: Understands semantic meaning
- **Skill Matching**: Finds 100+ technical skills
- **Coverage**: Calculates % of required skills present

### 3. AI Prediction
- **Random Forest** machine learning model
- Predicts job fit based on multiple features
- Outputs probability (0-100%) and confidence

### 4. Recommendations
- Identifies priority skills to learn
- Suggests improvements
- Highlights strengths

## 🔧 Customization

### Adding Your Own Skills

Edit `src/data_preprocessing.py`, find the `extract_skills()` function, and add to the list:

```python
comprehensive_skills = [
    'python', 'java', 'javascript',
    'your_new_skill_here',  # Add here
    # ... more skills
]
```

### Changing Model Settings

Edit `train_models.py`:

```python
# More features = slower but more accurate
fe.fit_tfidf(all_documents, max_features=5000)  # Default: 5000

# More epochs = slower but better semantic understanding
fe.train_doc2vec(all_documents, epochs=40)  # Default: 40

# More trees = slower but more accurate predictions
matcher.train_simple(X, y, n_estimators=200)  # Default: 200
```

## 🐛 Troubleshooting

### Error: "Models not found"
**Solution**: Run `python train_models.py` first

### Error: "spaCy model not found"
**Solution**: 
```powershell
python -m spacy download en_core_web_sm
```

### Error: "NLTK data not found"
**Solution**:
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Low Similarity Scores (TF-IDF: 0.000)
**Causes**:
1. Not enough text in resume/job description
2. Models not trained on enough data
3. No common keywords between documents

**Solutions**:
1. Ensure resume/job have detailed content
2. Run `python train_models.py` with more data
3. Check that `data/raw/` contains resume dataset

### Skills Not Being Found
**Solution**: Check that skills are spelled correctly and exist in the skills list in `src/data_preprocessing.py`

## 📝 Training on Your Own Data

### Resume Data Format
Place CSV file in `data/raw/UpdatedResumeDataSet.csv`:

```csv
Category,Resume
Data Science,"John Doe... Python, Machine Learning..."
Java Developer,"Jane Smith... Java, Spring Boot..."
```

### Job Postings Format
Place CSV file in `data/raw/jobPostings/postings.csv`:

```csv
job_id,description
1,"Seeking Python developer with ML experience..."
2,"Java engineer needed for backend development..."
```

Then run:
```powershell
python train_models.py
```

## 🎓 Understanding the Scores

### TF-IDF Similarity (0-100%)
- Measures keyword overlap
- High = many matching words
- Low = different vocabulary

### Doc2Vec Similarity (0-100%)
- Measures semantic meaning
- High = similar topics/concepts
- Can be high even with different words

### Skill Coverage (0-100%)
- % of required skills found in resume
- 80%+ = Excellent match
- 60-80% = Good match
- <60% = Skill gaps exist

### Fit Score (0-100%)
- Overall job fit prediction
- 70%+ = Good Fit
- <70% = Not a Fit

## 🚀 Next Steps

1. **Collect More Data**: Add more resumes and job descriptions to `data/raw/`
2. **Retrain Models**: Run `python train_models.py` periodically
3. **Customize Skills**: Add industry-specific skills to the database
4. **Share Dashboard**: Deploy to Streamlit Cloud for team access

## 📞 Need Help?

- Check the full code documentation in the original `README.md`
- Review error messages carefully
- Ensure virtual environment is activated
- Verify all dependencies are installed

---

**Happy Analyzing! 🎉**
