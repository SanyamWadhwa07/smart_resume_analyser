"""
DATA PREPARATION PIPELINE
Steps 1-4: Load data, create balanced pairs, preprocess text
Saves output to avoid reprocessing every time
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from src.data_loader import ResumeDataLoader
from src.data_preprocessing import TextPreprocessor
from src.skill_extractor import SkillExtractor

def load_enriched_job_data(limit=10000):
    """Load job postings with ALL metadata"""

    print("="*80)
    
    # Load main job postings
    print("  Loading postings.csv...")
    jobs_df = pd.read_csv('data/raw/jobPostings/postings.csv', nrows=limit)
    print(f"    ✓ Loaded {len(jobs_df):,} job postings")
    
    # Filter quality jobs
    jobs_df = jobs_df[jobs_df['description'].notna()]
    jobs_df = jobs_df[jobs_df['title'].notna()]
    jobs_df = jobs_df[jobs_df['description'].str.len() > 200]
    print(f"    ✓ After filtering: {len(jobs_df):,} quality jobs")
    
    # Load companies
    print("\n  Loading company data...")
    companies = pd.read_csv('data/raw/jobPostings/companies/companies.csv')
    print(f"    Loaded {len(companies):,} companies")
    
    # Load industries
    print("  Loading industry mappings...")
    job_industries = pd.read_csv('data/raw/jobPostings/jobs/job_industries.csv')
    industries_lookup = pd.read_csv('data/raw/jobPostings/mappings/industries.csv')
    print(f"    Loaded {len(job_industries):,} job-industry mappings")
    print(f"    Loaded {len(industries_lookup):,} industry types")
    
    # Join job industries
    job_industries_merged = job_industries.merge(
        industries_lookup, 
        on='industry_id', 
        how='left'
    )
    
    # Aggregate industries per job
    job_industries_agg = job_industries_merged.groupby('job_id')['industry_name'].apply(
        lambda x: ', '.join(x.dropna())
    ).reset_index()
    job_industries_agg.columns = ['job_id', 'industries']
    
    # Join all metadata
    print("\n  Enriching jobs with metadata...")
    jobs_enriched = jobs_df.copy()
    
    # Add company info
    jobs_enriched = jobs_enriched.merge(
        companies[['company_id', 'name', 'description', 'company_size', 'city', 'state', 'country']],
        on='company_id',
        how='left',
        suffixes=('', '_company')
    )
    
    # Add industries
    jobs_enriched = jobs_enriched.merge(
        job_industries_agg,
        on='job_id',
        how='left'
    )
    
    # Create full description
    jobs_enriched['full_description'] = jobs_enriched.apply(
        lambda row: f"{row['title']}. {row['description']}. " + 
                   (f"Skills: {row['skills_desc']}" if pd.notna(row['skills_desc']) else ""),
        axis=1
    )
    
    print(f"\n✓ Created {len(jobs_enriched):,} enriched job postings")
    print(f"  Average description length: {jobs_enriched['full_description'].str.len().mean():.0f} chars")
    
    if 'med_salary' in jobs_enriched.columns:
        print(f"  Jobs with salary data: {jobs_enriched['med_salary'].notna().sum():,} ({jobs_enriched['med_salary'].notna().mean()*100:.1f}%)")
    if 'industries' in jobs_enriched.columns:
        print(f"  Jobs with industry data: {jobs_enriched['industries'].notna().sum():,} ({jobs_enriched['industries'].notna().mean()*100:.1f}%)")
    if 'name' in jobs_enriched.columns:
        print(f"  Jobs with company data: {jobs_enriched['name'].notna().sum():,} ({jobs_enriched['name'].notna().mean()*100:.1f}%)")
    
    print(f"\n  Top 5 job titles:")
    for i, (title, count) in enumerate(jobs_enriched['title'].value_counts().head(5).items(), 1):
        print(f"    {i}. {title} ({count} postings)")
    
    return jobs_enriched

def create_balanced_training_pairs(resumes_df, jobs_df, skill_extractor, pairs_per_resume=5):
    """Create BALANCED training pairs"""
    print("\n[Step 3/4] Creating BALANCED training pairs...")
    print("="*80)
    print(f"Strategy: Smart pairing to achieve 40% positive, 60% negative balance")
    print(f"Pairing {len(resumes_df):,} resumes with {len(jobs_df):,} jobs\n")
    
    positive_pairs = []
    negative_pairs = []
    
    for idx, resume_row in resumes_df.iterrows():
        resume_text = resume_row['resume_text']
        resume_skills = set([s.lower() for s in resume_row['skills']])
        
        # Extract additional skills from resume text
        extracted_resume_skills = skill_extractor.extract_skills(resume_text)
        all_resume_skills = resume_skills.union(set(extracted_resume_skills))
        
        # Sample different jobs per resume for diversity
        sampled_jobs = jobs_df.sample(min(100, len(jobs_df)), random_state=idx)
        
        job_scores = []
        for job_idx, job_row in sampled_jobs.iterrows():
            job_text = job_row['full_description']
            
            # Extract technical skills from job description
            job_skills = set(skill_extractor.extract_skills(job_text))
            
            # Calculate overlap
            if len(job_skills) > 0:
                matched = len(all_resume_skills.intersection(job_skills))
                skill_coverage = matched / len(job_skills)
            else:
                skill_coverage = 0.0
            
            job_scores.append({
                'job_idx': job_idx,
                'job_row': job_row,
                'job_skills': job_skills,
                'skill_coverage': skill_coverage,
                'matched_count': len(all_resume_skills.intersection(job_skills)) if job_skills else 0
            })
        
        # Sort by skill coverage
        job_scores.sort(key=lambda x: x['skill_coverage'], reverse=True)
        
        # TOP 3 jobs → likely positive
        for job_score in job_scores[:3]:
            skill_coverage = job_score['skill_coverage']
            label = 1 if skill_coverage >= 0.30 else 0
            
            pair = {
                'resume_text': resume_text,
                'job_text': job_score['job_row']['full_description'],
                'job_title': job_score['job_row']['title'],
                'company': job_score['job_row']['name'],
                'location': f"{job_score['job_row']['city']}, {job_score['job_row']['state']}" if pd.notna(job_score['job_row']['city']) else 'Unknown',
                'industries': job_score['job_row']['industries'] if pd.notna(job_score['job_row']['industries']) else '',
                'salary_min': job_score['job_row'].get('min_salary', 0) if 'min_salary' in job_score['job_row'] and pd.notna(job_score['job_row'].get('min_salary')) else 0,
                'salary_max': job_score['job_row'].get('max_salary', 0) if 'max_salary' in job_score['job_row'] and pd.notna(job_score['job_row'].get('max_salary')) else 0,
                'resume_skills': all_resume_skills,
                'job_skills': job_score['job_skills'],
                'skill_coverage': skill_coverage,
                'matched_count': job_score['matched_count'],
                'label': label
            }
            
            if label == 1:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)
        
        # BOTTOM 2 jobs → guaranteed negative
        for job_score in job_scores[-2:]:
            pair = {
                'resume_text': resume_text,
                'job_text': job_score['job_row']['full_description'],
                'job_title': job_score['job_row']['title'],
                'company': job_score['job_row']['name'],
                'location': f"{job_score['job_row']['city']}, {job_score['job_row']['state']}" if pd.notna(job_score['job_row']['city']) else 'Unknown',
                'industries': job_score['job_row']['industries'] if pd.notna(job_score['job_row']['industries']) else '',
                'resume_skills': all_resume_skills,
                'job_skills': job_score['job_skills'],
                'skill_coverage': job_score['skill_coverage'],
                'matched_count': job_score['matched_count'],
                'label': 0
            }
            negative_pairs.append(pair)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1:,}/{len(resumes_df):,} resumes...")
    
    # BALANCE the dataset
    print(f"\n  Before balancing:")
    print(f"    Positive pairs: {len(positive_pairs):,}")
    print(f"    Negative pairs: {len(negative_pairs):,}")
    
    if len(positive_pairs) > 0:
        target_negative = int(len(positive_pairs) * 1.5)
        
        if len(negative_pairs) > target_negative:
            negative_pairs = list(np.random.choice(
                negative_pairs, 
                size=target_negative, 
                replace=False
            ))
    
    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    
    df_balanced = pd.DataFrame(all_pairs)
    
    pos_count = (df_balanced['label'] == 1).sum()
    neg_count = (df_balanced['label'] == 0).sum()
    
    print(f"\n  After balancing:")
    print(f"    Positive pairs: {pos_count:,} ({pos_count/len(df_balanced)*100:.1f}%)")
    print(f"    Negative pairs: {neg_count:,} ({neg_count/len(df_balanced)*100:.1f}%)")
    
    print(f"\n✓ Created {len(df_balanced):,} BALANCED training pairs")
    print(f"  Average skill coverage: {df_balanced['skill_coverage'].mean()*100:.2f}%")
    print(f"  Average matched skills: {df_balanced['matched_count'].mean():.1f}")
    
    return df_balanced

def main():
    print("="*80)
    print("DATA PREPARATION PIPELINE (Steps 1-4)")
    print("Saves preprocessed data to: data/processed/training_data.pkl")
    print("="*80)
    
    # Initialize skill extractor
    skill_extractor = SkillExtractor()
    print(f"\n✓ Initialized SkillExtractor with {len(skill_extractor.skills_set)} technical skills")
    
    # Step 1: Load Resumes
    print("\n[Step 1/4] Loading structured resume data...")
    print("="*80)
    loader = ResumeDataLoader('data/raw/master_resumes.jsonl')
    resumes = loader.load_resumes(limit=1000)
    resumes_df = loader.create_dataframe()
    resumes_df = resumes_df[resumes_df['resume_text'].str.len() > 100]
    
    print(f"✓ Loaded {len(resumes_df):,} resumes")
    print(f"  Average skills: {resumes_df['skills'].apply(len).mean():.1f}")
    print(f"  Average experience: {resumes_df['experience_years'].mean():.1f} years")
    
    # Step 2: Load Enriched Jobs
    jobs_df = load_enriched_job_data(limit=10000)
    
    # Step 3: Create Balanced Pairs
    training_data = create_balanced_training_pairs(
        resumes_df, 
        jobs_df, 
        skill_extractor,
        pairs_per_resume=5
    )
    
    # Step 4: Preprocess
    print("\n[Step 4/4] Preprocessing text data...")
    print("="*80)
    preprocessor = TextPreprocessor()
    
    print("  Cleaning resume texts...")
    training_data['resume_clean'] = training_data['resume_text'].apply(preprocessor.clean_text)
    print("  Cleaning job texts...")
    training_data['job_clean'] = training_data['job_text'].apply(preprocessor.clean_text)
    
    print(f"\n✓ Preprocessed {len(training_data)*2:,} documents")
    
    # SAVE preprocessed data in BOTH formats
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING PREPROCESSED DATA...")
    print("="*80)
    
    # 1. Save as CSV for human inspection
    csv_file = 'data/processed/training_data.csv'
    
    # Create a simplified version for CSV (sets can't be saved directly)
    csv_data = training_data.copy()
    csv_data['resume_skills'] = csv_data['resume_skills'].apply(lambda x: '|'.join(sorted(x)))
    csv_data['job_skills'] = csv_data['job_skills'].apply(lambda x: '|'.join(sorted(x)))
    
    csv_data.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    print(f"  File size: {os.path.getsize(csv_file) / (1024*1024):.2f} MB")
    print(f"  Columns: {len(csv_data.columns)}")
    print(f"  Preview: Open in Excel/Notepad to inspect")
    
    # 2. Save as PKL for fast loading (preserves sets and all data types)
    pkl_file = 'data/processed/training_data.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n✓ Saved PKL: {pkl_file}")
    print(f"  File size: {os.path.getsize(pkl_file) / (1024*1024):.2f} MB")
    print(f"  Use: Fast loading with preserved data types")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total pairs: {len(training_data):,}")
    print(f"  Positive: {(training_data['label']==1).sum():,} ({(training_data['label']==1).mean()*100:.1f}%)")
    print(f"  Negative: {(training_data['label']==0).sum():,} ({(training_data['label']==0).mean()*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output files:")
    print(f"  1. {csv_file} (human-readable)")
    print(f"  2. {pkl_file} (for train_model.py)")
    print(f"\nNext step: Run train_model.py to train the classifier")
    print("="*80)

if __name__ == "__main__":
    main()
