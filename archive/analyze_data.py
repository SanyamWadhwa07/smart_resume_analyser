"""
Quick data analysis script to understand the dataset structure
"""
import pandas as pd
import os

print("=" * 80)
print("DATA ANALYSIS FOR SMART RESUME ANALYZER")
print("=" * 80)

# 1. Analyze Resume Dataset
print("\n[1] RESUME DATASET ANALYSIS")
print("-" * 80)
try:
    resume_df = pd.read_csv('data/raw/UpdatedResumeDataSet.csv')
    print(f"✓ Rows: {len(resume_df):,}")
    print(f"✓ Columns: {list(resume_df.columns)}")
    print(f"\n✓ Categories Distribution:")
    print(resume_df['Category'].value_counts())
    print(f"\n✓ Sample Resume Length: {len(resume_df['Resume'].iloc[0])} characters")
    print(f"✓ Average Resume Length: {resume_df['Resume'].str.len().mean():.0f} characters")
    print(f"\n✓ First 200 chars of sample resume:")
    print(resume_df['Resume'].iloc[0][:200])
except Exception as e:
    print(f"✗ Error: {e}")

# 2. Analyze Job Postings Dataset
print("\n\n[2] JOB POSTINGS DATASET ANALYSIS")
print("-" * 80)
try:
    # Check file size first
    file_path = 'data/raw/jobPostings/postings.csv'
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    
    # Read first 1000 rows to analyze structure
    job_df = pd.read_csv(file_path, nrows=1000)
    print(f"✓ Columns: {list(job_df.columns)}")
    print(f"\n✓ Sample column values:")
    for col in job_df.columns[:10]:  # First 10 columns
        print(f"  - {col}: {job_df[col].iloc[0][:100] if isinstance(job_df[col].iloc[0], str) else job_df[col].iloc[0]}")
    
    # Count total rows
    total_rows = sum(1 for _ in open(file_path)) - 1  # -1 for header
    print(f"\n✓ Total job postings: {total_rows:,}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# 3. Analyze Skills Mapping
print("\n\n[3] SKILLS MAPPING ANALYSIS")
print("-" * 80)
try:
    skills_df = pd.read_csv('data/raw/jobPostings/mappings/skills.csv')
    print(f"✓ Rows: {len(skills_df)}")
    print(f"✓ Columns: {list(skills_df.columns)}")
    print(f"\n✓ Skills:")
    print(skills_df)
except Exception as e:
    print(f"✗ Error: {e}")

# 4. Analyze Job Skills
print("\n\n[4] JOB SKILLS DATASET ANALYSIS")
print("-" * 80)
try:
    job_skills_df = pd.read_csv('data/raw/jobPostings/jobs/job_skills.csv', nrows=100)
    print(f"✓ Columns: {list(job_skills_df.columns)}")
    print(f"\n✓ Sample data:")
    print(job_skills_df.head())
except Exception as e:
    print(f"✗ Error: {e}")

# 5. Recommendations
print("\n\n[5] RECOMMENDATIONS FOR TRAINING")
print("-" * 80)
print("""
Based on data analysis:

1. Resume Dataset: 
   - Use 'Category' column to create job categories
   - Use 'Resume' column for text analysis
   
2. Job Postings Dataset:
   - Large file - sample for training efficiency
   - Need to identify the description column
   
3. Skills:
   - Limited skills in mapping (only job categories)
   - Use comprehensive technical skills list instead
   
4. Training Strategy:
   - Match resumes to jobs based on Category
   - Extract skills using NLP from resume text
   - Create positive/negative training pairs
""")

print("=" * 80)
