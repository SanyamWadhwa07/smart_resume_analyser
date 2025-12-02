# Dataset Preparation Documentation

## Overview
This document explains how the training dataset is prepared for the Smart Resume Analyzer machine learning model. The preparation process transforms raw resume and job posting data into balanced, feature-rich training pairs.

---

## Data Sources

### 1. Resume Data
- **File**: `data/raw/master_resumes.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Contents**: Structured resume information including:
  - Resume text
  - Extracted skills
  - Experience years
  - Education details
  - Contact information

### 2. Job Posting Data
Multiple related CSV files in `data/raw/jobPostings/`:

#### Primary Data
- **`postings.csv`**: Main job postings with descriptions, titles, and requirements
- **`companies/companies.csv`**: Company information (name, size, location, description)

#### Metadata Mappings
- **`jobs/job_industries.csv`**: Links jobs to industries
- **`mappings/industries.csv`**: Industry type definitions

---

## Preparation Pipeline

The dataset preparation follows a **4-step pipeline** implemented in `scripts/prepare_training_data.py`:

### Step 1: Load Resume Data
```
Input:  master_resumes.jsonl
Output: DataFrame with ~1,000 quality resumes
```

**Process**:
1. Load structured resume data from JSONL file
2. Filter resumes with sufficient content (>100 characters)
3. Extract resume features:
   - Full resume text
   - Skill lists
   - Years of experience
   - Education level

**Quality Metrics**:
- Average skills per resume: ~10-15
- Average experience: varies by dataset
- Minimum text length: 100 characters

---

### Step 2: Load & Enrich Job Postings
```
Input:  postings.csv, companies.csv, job_industries.csv, industries.csv
Output: Enriched job DataFrame with ~10,000 quality jobs
```

**Process**:
1. Load up to 10,000 job postings from `postings.csv`
2. Filter quality jobs:
   - Must have job description (not null)
   - Must have job title (not null)
   - Description length > 200 characters
3. Join company metadata:
   - Company name
   - Company size
   - Location (city, state, country)
   - Company description
4. Join industry information:
   - Multiple industries per job (aggregated)
   - Industry names from lookup table
5. Create full job description:
   - Combines: title + description + skills_desc

**Enrichment Statistics**:
- Jobs with salary data: ~40-60%
- Jobs with industry data: ~70-90%
- Jobs with company data: ~95-99%
- Average description length: 800-1200 characters

---

### Step 3: Create Balanced Training Pairs
```
Input:  Resumes DataFrame + Jobs DataFrame
Output: Balanced pairs with 40% positive, 60% negative
```

**Pairing Strategy**:

For each resume:
1. **Sample** 100 random jobs for diversity
2. **Extract skills** from both resume and job descriptions using SkillExtractor
3. **Calculate skill coverage**:
   ```
   skill_coverage = matched_skills / total_job_skills
   ```
4. **Rank jobs** by skill coverage
5. **Select pairs**:
   - **Top 3 jobs** → Likely positive matches (if coverage ≥ 30%)
   - **Bottom 2 jobs** → Guaranteed negative matches

**Labeling Logic**:
- **Label = 1 (Positive)**: Skill coverage ≥ 30%
- **Label = 0 (Negative)**: Skill coverage < 30%

**Balancing**:
- Target ratio: 40% positive, 60% negative
- If too many negatives, randomly sample to maintain balance
- Final shuffle for random distribution

**Pair Features**:
Each training pair includes:
- `resume_text`: Full resume content
- `job_text`: Full job description
- `job_title`: Job position title
- `company`: Company name
- `location`: City, State
- `industries`: Industry categories
- `salary_min`, `salary_max`: Salary range (if available)
- `resume_skills`: Set of extracted resume skills
- `job_skills`: Set of extracted job skills
- `skill_coverage`: Percentage of job skills matched
- `matched_count`: Number of matched skills
- `label`: Binary label (1=match, 0=no match)

---

### Step 4: Text Preprocessing
```
Input:  Raw training pairs
Output: Cleaned text ready for model training
```

**Process**:
1. Apply `TextPreprocessor.clean_text()` to resume texts
2. Apply `TextPreprocessor.clean_text()` to job texts
3. Create new columns:
   - `resume_clean`: Preprocessed resume text
   - `job_clean`: Preprocessed job text

**Text Cleaning Steps** (from TextPreprocessor):
- Lowercase conversion
- Remove special characters
- Remove extra whitespace
- Tokenization
- Remove stop words (optional)
- Lemmatization (optional)

---

## Output Files

### 1. CSV Format: `data/processed/training_data.csv`
**Purpose**: Human-readable inspection

**Modifications for CSV**:
- Skill sets converted to pipe-delimited strings: `"python|java|sql"`
- All columns flattened for spreadsheet compatibility

**Use Cases**:
- Open in Excel for manual review
- Data quality inspection
- Statistical analysis
- Share with non-technical stakeholders

### 2. Pickle Format: `data/processed/training_data.pkl`
**Purpose**: Fast loading for model training

**Advantages**:
- Preserves Python data types (sets, lists, etc.)
- Much faster to load than CSV
- Maintains full data integrity
- Directly used by `train_model.py`

**File Sizes**: Typically 5-20 MB depending on dataset size

---

## Dataset Statistics

### Typical Output
```
Total pairs:     ~5,000 - 8,000
Positive pairs:  ~2,000 - 3,200 (40%)
Negative pairs:  ~3,000 - 4,800 (60%)

Average skill coverage:  25-35%
Average matched skills:  3-5 skills per pair
```

### Quality Metrics
- **Skill coverage distribution**: Right-skewed (most pairs have low coverage)
- **Matched skills**: Ranges from 0 to 15+ depending on seniority
- **Text lengths**: 
  - Resumes: 500-2000 characters
  - Jobs: 800-1500 characters

---

## SkillExtractor Component

The `SkillExtractor` class is crucial for intelligent pairing:

**Features**:
- Maintains database of ~200-500 technical skills
- Extracts skills from unstructured text
- Case-insensitive matching
- Handles variations (e.g., "JavaScript" = "JS")

**Usage in Pipeline**:
```python
skill_extractor = SkillExtractor()
resume_skills = skill_extractor.extract_skills(resume_text)
job_skills = skill_extractor.extract_skills(job_description)
```

---

## Running the Pipeline

### Prerequisites
```bash
# Activate virtual environment
& D:\Projects\smart_resume_analyser\m\Scripts\Activate.ps1

# Ensure required packages are installed
pip install pandas numpy
```

### Execute
```bash
python scripts/prepare_training_data.py
```

### Expected Runtime
- Small dataset (1K resumes, 10K jobs): 2-5 minutes
- Large dataset (5K resumes, 50K jobs): 10-20 minutes

### Progress Output
The script provides detailed progress logging:
```
[Step 1/4] Loading structured resume data...
[Step 2/4] Loading REAL job postings with metadata...
[Step 3/4] Creating BALANCED training pairs...
[Step 4/4] Preprocessing text data...
SAVING PREPROCESSED DATA...
```

---

## Next Steps

After dataset preparation:

1. **Verify Data Quality**:
   ```bash
   python scripts/check_training_data.py
   ```

2. **Train Model**:
   ```bash
   python scripts/train_model.py
   ```

3. **Validate Results**:
   ```bash
   python scripts/validate_logic.py
   ```

---

## Troubleshooting

### Issue: Low Positive Pair Count
**Solution**: Lower the skill coverage threshold (currently 30%)

### Issue: Unbalanced Dataset
**Solution**: Adjust `target_negative` ratio in `create_balanced_training_pairs()`

### Issue: Missing Job Metadata
**Solution**: Check that all CSV files are present in `data/raw/jobPostings/`

### Issue: Memory Errors
**Solution**: Reduce `limit` parameter when loading jobs/resumes

---

## Configuration Options

Key parameters in `prepare_training_data.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resume_limit` | 1000 | Max resumes to load |
| `job_limit` | 10000 | Max jobs to load |
| `pairs_per_resume` | 5 | Total pairs per resume |
| `skill_coverage_threshold` | 0.30 | Minimum for positive label |
| `target_ratio` | 1.5 | Negative/positive ratio |

---

## Data Privacy & Ethics

⚠️ **Important Considerations**:
- Ensure resume data is anonymized (no personal identifiable information)
- Job posting data should be publicly available or properly licensed
- Follow data protection regulations (GDPR, CCPA, etc.)
- Do not use for discriminatory purposes

---

## Version History

- **v1.0** (Nov 2025): Initial pipeline with balanced pairing strategy
- Enriched job metadata integration
- Dual output format (CSV + PKL)

---

## Contact & Support

For questions or issues with dataset preparation:
1. Check the terminal output for detailed error messages
2. Review logs in the console
3. Verify input data files exist and are formatted correctly
4. Consult the main `README.md` for project setup

---

*Last Updated: November 19, 2025*
