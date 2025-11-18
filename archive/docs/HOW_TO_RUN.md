# 🎯 Smart Resume Analyzer - How to Run

## Current Status
✅ Virtual environment created (`m/`)
✅ Source code complete
⚠️ Dependencies need to be installed
⚠️ Models need to be trained

## Steps to Run (In Order)

### Step 1: Install All Dependencies
```powershell
python setup.py
```

This will:
- Install all Python packages (pandas, scikit-learn, nltk, spacy, etc.)
- Download spaCy English model
- Download NLTK data
- Create necessary directories

**Expected time**: 2-5 minutes

### Step 2: Train the AI Models
```powershell
python train_models.py
```

This will:
- Load your resume dataset from `data/raw/UpdatedResumeDataSet.csv`
- Preprocess all text data
- Train TF-IDF vectorizer (keyword matching)
- Train Doc2Vec model (semantic understanding)
- Train Random Forest classifier (prediction)
- Save all models to `models/` folder

**Expected time**: 3-10 minutes (depending on data size)

**What you'll see**:
```
[Step 1/5] Loading data...
✓ Loaded 399 resumes
✓ Loaded X job postings

[Step 2/5] Preprocessing data...
Processing 200 resumes...
✓ Total documents for training: 250

[Step 3/5] Training feature engineering models...
TF-IDF vectorizer fitted with 4521 features
Doc2Vec model trained with 200-dimensional vectors

[Step 4/5] Generating training features...
✓ Generated 1000 training samples

[Step 5/5] Training Random Forest...
✓ Accuracy: 0.85
```

### Step 3: Test the System (Optional but Recommended)
```powershell
python test_analyzer.py
```

This will:
- Run a quick test with sample resume and job description
- Show you how the analyzer works
- Verify everything is working correctly

**What you'll see**:
```
📊 SIMILARITY SCORES:
  TF-IDF Similarity:    65.3%
  Doc2Vec Similarity:   58.2%
  Skill Coverage:       80.0%

✅ MATCHED SKILLS:
  • python
  • machine learning
  • tensorflow
  ...

❌ MISSING SKILLS:
  • spark
  • hadoop
```

### Step 4: Launch the Web Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```

This will:
- Start the web server
- Open your browser automatically
- Show the interactive dashboard at http://localhost:8501

**What you can do**:
1. Upload PDF resume
2. Upload or paste job description
3. Click "Analyze Job Fit"
4. View results:
   - Job fit score (0-100%)
   - Matched skills
   - Missing skills
   - Recommendations

## 🎨 Using the Dashboard

### Single Analysis Mode

1. **Select "Single Analysis"** in the sidebar

2. **Upload Resume**:
   - Method 1: Click "Upload File" → Select PDF or TXT
   - Method 2: Click "Paste Text" → Copy-paste resume

3. **Upload Job Description**:
   - Same as above

4. **Click "🔍 Analyze Job Fit"**

5. **View Results**:
   - 📊 **Gauge Chart**: Overall fit score
   - ✅ **Matched Skills**: Green badges showing what matches
   - ❌ **Missing Skills**: Red badges showing gaps
   - 💡 **Recommendations**: Specific advice to improve
   - 📈 **Radar Chart**: Visual breakdown of scores

### Batch Analysis Mode

1. **Select "Batch Analysis"** in the sidebar

2. **Upload ONE job description**

3. **Upload MULTIPLE resumes** (Ctrl+Click or Cmd+Click to select many)

4. **Click "📈 Rank Candidates"**

5. **View Rankings**:
   - Table sorted by fit score
   - Bar chart of top 10 candidates
   - Download button for CSV export

## 📊 Understanding Your Scores

### TF-IDF Similarity (Keyword Match)
- **High (>60%)**: Many matching keywords between resume and job
- **Medium (30-60%)**: Some overlap in vocabulary
- **Low (<30%)**: Different terminology used

### Doc2Vec Similarity (Semantic Match)
- **High (>50%)**: Similar topics and concepts
- **Medium (20-50%)**: Related but different focus
- **Low (<20%)**: Very different domains

### Skill Coverage
- **80-100%**: Excellent! You have most required skills
- **60-80%**: Good match, minor gaps
- **40-60%**: Moderate gaps, consider upskilling
- **<40%**: Significant gaps, may need training

### Overall Fit Score
- **70-100%**: **Good Fit** ✅ - Strong candidate
- **0-70%**: **Not a Fit** ❌ - Consider upskilling

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nltk'"
**Fix**: Run `python setup.py` to install all dependencies

### Issue: "Models not found" when running dashboard
**Fix**: Run `python train_models.py` first

### Issue: Low similarity scores (TF-IDF: 0.0, Doc2Vec: -0.002)
**Causes**:
1. Models trained on too little data
2. Resume/job description too short
3. No keyword overlap

**Fixes**:
1. Ensure `data/raw/UpdatedResumeDataSet.csv` has data
2. Use detailed resume and job description (at least 200 words each)
3. Retrain models: `python train_models.py`

### Issue: "TF-IDF vectorizer fitted with 1 features"
**Cause**: Not enough documents during training

**Fix**: 
1. Check that `data/raw/UpdatedResumeDataSet.csv` exists and has content
2. Check that `data/raw/jobPostings/postings.csv` exists
3. Retrain: `python train_models.py`

### Issue: Skills not being detected
**Fix**: 
1. Check spelling of skills
2. Add custom skills to `src/data_preprocessing.py` in the `extract_skills()` function
3. Retrain models

## 📁 File Structure After Setup

```
smart_resume_analyzer/
│
├── m/                          # Virtual environment
│   └── ...
│
├── models/                     # ✨ Created after training
│   ├── tfidf_vectorizer.pkl    # Keyword matching model
│   ├── doc2vec_model.model     # Semantic model
│   └── random_forest_model.pkl # ML prediction model
│
├── data/
│   ├── raw/
│   │   ├── UpdatedResumeDataSet.csv    # Your resume data
│   │   └── jobPostings/
│   │       └── postings.csv             # Job postings data
│   └── processed/               # ✨ Created by pipeline
│
├── src/                        # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
│
├── dashboard/
│   └── streamlit_app.py        # Web interface
│
├── setup.py                    # 👈 Run this FIRST
├── train_models.py             # 👈 Run this SECOND
├── test_analyzer.py            # 👈 Run this THIRD (optional)
├── requirements.txt
├── SETUP_GUIDE.md             # This file
└── README.md                  # Full documentation
```

## ⚡ Quick Commands Reference

```powershell
# 1. First time setup
python setup.py

# 2. Train models
python train_models.py

# 3. Test system
python test_analyzer.py

# 4. Run dashboard
streamlit run dashboard/streamlit_app.py

# If you need to reinstall packages
pip install -r requirements.txt

# If spaCy model missing
python -m spacy download en_core_web_sm

# If NLTK data missing
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🎯 Expected Results

After proper training, you should see:

**Good Results**:
- TF-IDF Similarity: 30-80%
- Doc2Vec Similarity: 20-70%
- Skill Coverage: Depends on resume vs job
- TF-IDF features: 3000-7000

**Poor Results (Need Retraining)**:
- TF-IDF Similarity: 0%
- Doc2Vec Similarity: <0.01
- TF-IDF features: <100

## 🚀 Next Steps After Setup

1. **Customize Skills Database**:
   - Edit `src/data_preprocessing.py`
   - Add industry-specific skills
   - Retrain models

2. **Improve Accuracy**:
   - Add more resume data to `data/raw/`
   - Increase training epochs in `train_models.py`
   - Tune Random Forest parameters

3. **Deploy**:
   - Share locally: `streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0`
   - Deploy to cloud: Streamlit Cloud, Heroku, AWS

## 📞 Still Need Help?

1. Check error messages carefully
2. Ensure virtual environment is activated: `.\m\Scripts\Activate.ps1`
3. Verify Python version: `python --version` (should be 3.8+)
4. Check logs in terminal for detailed errors

---

**You're all set! Start with `python setup.py` and follow the steps above. Good luck! 🎉**
