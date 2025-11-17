# 🚀 Complete Training Guide - From Scratch

## Prerequisites Checklist

Before training, ensure you have:

- [x] Virtual environment activated (`.\m\Scripts\Activate.ps1`)
- [x] All dependencies installed (`pip install -r requirements.txt`)
- [x] NLTK data downloaded
- [x] spaCy model downloaded
- [x] Resume dataset in `data/raw/UpdatedResumeDataSet.csv`
- [x] Job postings in `data/raw/jobPostings/postings.csv`

---

## 🎯 Complete Training Process

### Step 1: Activate Environment
```powershell
# Navigate to project
cd d:\Projects\smart_resume_analyzer

# Activate virtual environment
.\m\Scripts\Activate.ps1
```

### Step 2: Verify Data Exists
```powershell
# Check resume dataset
Test-Path "data\raw\UpdatedResumeDataSet.csv"
# Should return: True

# Check job postings
Test-Path "data\raw\jobPostings\postings.csv"
# Should return: True
```

### Step 3: Run Training Pipeline
```powershell
python scripts/train_models.py
```

**What this does:**
1. ✅ Loads 962 resumes from dataset
2. ✅ Loads 1000+ job postings
3. ✅ Creates 10,000 resume-job training pairs
4. ✅ Preprocesses all text (clean, tokenize, lemmatize)
5. ✅ Trains TF-IDF vectorizer (5000 features)
6. ✅ Trains Doc2Vec model (200 dimensions, 40 epochs)
7. ✅ Extracts skills from all documents
8. ✅ Generates 4 features for each pair
9. ✅ Trains Random Forest classifier (200 trees)
10. ✅ Saves all models to `models/` folder

**Expected Duration:** 5-10 minutes

**Expected Output:**
```
Starting Smart Resume Analyzer Training Pipeline...

Step 1/6: Loading datasets...
✓ Loaded 962 resumes
✓ Loaded 1061 job postings

Step 2/6: Creating training pairs...
✓ Created 10000 pairs (5000 matched, 5000 unmatched)

Step 3/6: Preprocessing texts...
✓ Preprocessed 20000 documents

Step 4/6: Training TF-IDF model...
✓ TF-IDF trained with 5000 features

Step 5/6: Training Doc2Vec model...
Epoch 1/40...
...
✓ Doc2Vec trained (200 dimensions)

Step 6/6: Training Random Forest...
✓ Random Forest trained (Accuracy: ~50%)

✓ All models saved to models/
✓ Training complete!
```

---

## 📁 Models Created

After training, you'll have these files in `models/`:

```
models/
├── tfidf_vectorizer.pkl        (~15 MB)
├── doc2vec_model.model         (~30 MB)
└── random_forest_model.pkl     (~5 MB)
```

---

## 🧪 Verify Training Success

### Test 1: Quick System Check
```powershell
python tests/test_analyzer.py
```

**Expected Output:**
```
✓ Models loaded successfully
✓ TF-IDF Similarity: 34.1%
✓ Doc2Vec Similarity: 24.2%
✓ Skills Coverage: 63.2%
✓ Prediction: GOOD FIT (85.6%)
```

### Test 2: Run Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```

**Verify:**
- Dashboard opens at http://localhost:8501
- Upload a sample PDF resume
- Paste a job description
- Click "Analyze Resume"
- Results appear with scores and recommendations

---

## 🔧 Training Parameters (Advanced)

If you want to customize training, edit `scripts/train_models.py`:

### TF-IDF Settings (Line ~150)
```python
vectorizer = TfidfVectorizer(
    max_features=5000,      # Change to 7000 for more features
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,               # Min document frequency
    max_df=0.8              # Max document frequency
)
```

### Doc2Vec Settings (Line ~180)
```python
model = Doc2Vec(
    vector_size=200,        # Change to 300 for more dimensions
    window=10,              # Context window size
    min_count=2,            # Min word frequency
    workers=4,              # CPU cores to use
    epochs=40,              # Change to 100 for better accuracy
    dm=1                    # PV-DM algorithm
)
```

### Random Forest Settings (Line ~240)
```python
model = RandomForestClassifier(
    n_estimators=200,       # Change to 500 for more trees
    max_depth=20,           # Max tree depth
    min_samples_split=5,    # Min samples to split
    class_weight='balanced', # Handle imbalanced classes
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)
```

---

## 📊 Training with Custom Data

### Option 1: Replace Existing Data
1. Replace `data/raw/UpdatedResumeDataSet.csv` with your resume CSV
2. Replace `data/raw/jobPostings/postings.csv` with your job CSV
3. Run `python scripts/train_models.py`

**Required CSV Format:**

**Resumes CSV:**
```csv
Category,Resume
"Data Science","Python expert with 5 years experience..."
"Java Developer","Senior Java developer with Spring..."
```

**Jobs CSV:**
```csv
job_id,title,description
1,"Data Scientist","Looking for Python and ML expert..."
2,"Java Developer","Need experienced Java developer..."
```

### Option 2: Modify Training Script
Edit `scripts/train_models.py`:

```python
# Around line 30-40
def load_resume_data():
    # Change this path
    df = pd.read_csv('data/raw/your_custom_resumes.csv')
    return df

def load_job_data():
    # Change this path
    df = pd.read_csv('data/raw/your_custom_jobs.csv')
    return df
```

---

## 🎯 Training Data Requirements

### Minimum Requirements
- **Resumes**: 500+ diverse resumes
- **Jobs**: 500+ job descriptions
- **Categories**: 10+ different job categories
- **Training Pairs**: 5,000+ labeled pairs

### Recommended
- **Resumes**: 1,000+ resumes
- **Jobs**: 1,000+ job postings
- **Categories**: 20+ categories
- **Training Pairs**: 10,000+ pairs
- **Balance**: 50% matched, 50% unmatched

### Data Quality Tips
✅ Clean, real-world resumes and jobs
✅ Diverse job categories
✅ Various experience levels
✅ Different industries
✅ Recent data (last 2-3 years)

❌ Avoid duplicate resumes
❌ Avoid template-generated content
❌ Avoid very short documents (<100 words)

---

## 🐛 Troubleshooting Training Issues

### Issue: "File not found"
```powershell
# Check file paths
Test-Path "data\raw\UpdatedResumeDataSet.csv"
Test-Path "data\raw\jobPostings\postings.csv"

# If False, download or move your data files
```

### Issue: "Memory Error"
Reduce training size in `scripts/train_models.py`:

```python
# Line ~70
num_pairs = 5000  # Instead of 10000
```

### Issue: "NLTK data not found"
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Issue: "spaCy model not found"
```powershell
python -m spacy download en_core_web_sm
```

### Issue: Training takes too long
- Reduce `num_pairs` to 5000
- Reduce `epochs` to 20
- Reduce `n_estimators` to 100
- Use fewer CPU cores

### Issue: Low accuracy
- Increase training data (>10,000 pairs)
- Increase `epochs` to 100
- Increase `n_estimators` to 500
- Check data quality
- Ensure balanced classes

---

## 📈 Monitoring Training Progress

The training script shows progress for each step:

```
Step 4/6: Training TF-IDF model...
  → Vectorizing 20000 documents...
  ✓ TF-IDF trained with 5000 features

Step 5/6: Training Doc2Vec model...
  → Tagging documents...
  → Building vocabulary...
  → Training (40 epochs)...
  Epoch 1/40... (2s)
  Epoch 2/40... (2s)
  ...
  ✓ Doc2Vec trained (200 dimensions)
```

---

## 🎓 Understanding the Training Pipeline

### What Gets Trained?

1. **TF-IDF Vectorizer**
   - Learns vocabulary from all documents
   - Assigns importance weights to words
   - Creates 5000-dimensional vectors

2. **Doc2Vec Model**
   - Learns semantic representations
   - Creates 200-dimensional embeddings
   - Captures context and meaning

3. **Random Forest Classifier**
   - Learns to predict job fit
   - Uses 4 features as input
   - Outputs probability (0-100%)

### Training Data Flow

```
Resumes (962) + Jobs (1061)
         ↓
Create Pairs (10,000)
    ↓ 50% matched
    ↓ 50% unmatched
         ↓
Preprocess Text
    ↓ Clean
    ↓ Tokenize
    ↓ Lemmatize
         ↓
Train TF-IDF (on all 20K docs)
         ↓
Train Doc2Vec (on all 20K docs)
         ↓
Extract Features (4 per pair)
    ↓ TF-IDF similarity
    ↓ Doc2Vec similarity
    ↓ Skill Jaccard
    ↓ Skill coverage
         ↓
Train Random Forest
    ↓ Input: 4 features
    ↓ Output: Fit probability
         ↓
Save Models (3 files)
```

---

## ✅ Post-Training Checklist

After successful training:

- [ ] Verify 3 model files in `models/` folder
- [ ] Run `python tests/test_analyzer.py` (all pass)
- [ ] Run `streamlit run dashboard/streamlit_app.py`
- [ ] Test with sample resume + job description
- [ ] Check scores are reasonable (20-80% range)
- [ ] Verify recommendations appear
- [ ] Models load without errors

---

## 🔄 Retraining Schedule

### When to Retrain?

**Immediately:**
- Adding new job categories
- Significant data quality improvements
- Changing feature engineering

**Monthly:**
- Adding 500+ new resumes/jobs
- Updating skills database
- Improving preprocessing

**Quarterly:**
- Regular maintenance
- Keeping models fresh
- Incorporating feedback

### Quick Retrain
```powershell
# Backup old models
Copy-Item -Path "models\*" -Destination "models\backup_$(Get-Date -Format 'yyyy-MM-dd')\"

# Retrain
python scripts/train_models.py

# Test
python tests/test_analyzer.py
```

---

## 💡 Pro Tips

1. **Before Training:**
   - Backup existing models
   - Clean your data thoroughly
   - Check for duplicates

2. **During Training:**
   - Don't interrupt the process
   - Monitor memory usage
   - Watch for errors

3. **After Training:**
   - Always run tests
   - Compare with previous models
   - Document changes

4. **Performance:**
   - More data = better accuracy
   - More epochs = better Doc2Vec
   - More trees = better predictions

---

**Training Guide Status**: ✅ Complete
**Last Updated**: November 15, 2025
**Tested**: ✅ Working
