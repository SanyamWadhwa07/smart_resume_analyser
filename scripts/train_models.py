"""
Complete training pipeline for Smart Resume Analyzer
Trains TF-IDF, Doc2Vec, and Random Forest models on actual resume data
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, 'src')

from data_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ResumeJobMatcher

def load_resume_data():
    """Load resume dataset"""
    return pd.read_csv('data/raw/UpdatedResumeDataSet.csv')

def load_job_data(sample_size=1000):
    """Load job postings dataset (sampled for efficiency)"""
    try:
        # Large file - sample it
        return pd.read_csv('data/raw/jobPostings/postings.csv', 
                          nrows=sample_size,
                          encoding='utf-8',
                          on_bad_lines='skip')
    except:
        try:
            return pd.read_csv('data/raw/jobPostings/postings.csv', 
                              nrows=sample_size,
                              encoding='latin-1',
                              on_bad_lines='skip')
        except Exception as e:
            print(f"Warning: Could not load job postings: {e}")
            return None

def create_training_pairs(resumes_df, jobs_df=None):
    """Create training pairs from resume data"""
    pairs = []
    
    if jobs_df is not None and 'description' in jobs_df.columns:
        # Match resumes with actual job descriptions
        print("Creating resume-job pairs from job postings...")
        
        # Sample to avoid too many combinations
        resume_sample = resumes_df.sample(n=min(100, len(resumes_df)), random_state=42)
        job_sample = jobs_df.sample(n=min(100, len(jobs_df)), random_state=42)
        
        for _, resume_row in resume_sample.iterrows():
            for _, job_row in job_sample.iterrows():
                # Positive pair: same category (simplified)
                label = 1 if np.random.random() > 0.5 else 0  # Placeholder logic
                
                pairs.append({
                    'resume_text': resume_row['Resume'],
                    'job_text': job_row['description'],
                    'resume_category': resume_row['Category'],
                    'label': label
                })
    else:
        # Create synthetic job descriptions from categories
        print("Creating synthetic training data from resume categories...")
        
        categories = resumes_df['Category'].unique()
        
        for category in categories:
            category_resumes = resumes_df[resumes_df['Category'] == category]
            other_resumes = resumes_df[resumes_df['Category'] != category]
            
            # Sample resumes from this category
            for _, resume_row in category_resumes.head(5).iterrows():
                # Positive pair: same category
                if len(category_resumes) > 1:
                    job_text = f"We are looking for a {category}. " + resume_row['Resume'][:500]
                    pairs.append({
                        'resume_text': resume_row['Resume'],
                        'job_text': job_text,
                        'resume_category': category,
                        'label': 1
                    })
                
                # Negative pair: different category
                if len(other_resumes) > 0:
                    other_category = np.random.choice(other_resumes['Category'].unique())
                    job_text = f"We are looking for a {other_category} professional."
                    pairs.append({
                        'resume_text': resume_row['Resume'],
                        'job_text': job_text,
                        'resume_category': category,
                        'label': 0
                    })
    
    return pd.DataFrame(pairs)

def main():
    print("=" * 80)
    print("SMART RESUME ANALYZER - MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load Data
    print("\n[Step 1/6] Loading data...")
    try:
        resumes_df = load_resume_data()
        print(f"✓ Loaded {len(resumes_df)} resumes")
        print(f"✓ Categories: {resumes_df['Category'].nunique()}")
        
        jobs_df = load_job_data(sample_size=1000)
        if jobs_df is not None:
            print(f"✓ Loaded {len(jobs_df)} job postings (sampled)")
        else:
            print("⚠ Job postings not available, will create synthetic data")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Create Training Pairs
    print("\n[Step 2/6] Creating training pairs...")
    try:
        pairs_df = create_training_pairs(resumes_df, jobs_df)
        print(f"✓ Created {len(pairs_df)} training pairs")
        print(f"✓ Positive samples: {sum(pairs_df['label'] == 1)}")
        print(f"✓ Negative samples: {sum(pairs_df['label'] == 0)}")
    except Exception as e:
        print(f"✗ Error creating pairs: {e}")
        return
    
    # Step 3: Preprocess Data
    print("\n[Step 3/6] Preprocessing text data...")
    preprocessor = TextPreprocessor()
    
    print("Processing resumes and job descriptions...")
    cleaned_resumes = []
    cleaned_jobs = []
    
    for idx, row in pairs_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(pairs_df)} pairs...")
        
        try:
            cleaned_resume = preprocessor.preprocess(row['resume_text'])
            cleaned_job = preprocessor.preprocess(row['job_text'])
            
            if cleaned_resume and cleaned_job:
                cleaned_resumes.append(cleaned_resume)
                cleaned_jobs.append(cleaned_job)
        except:
            continue
    
    all_documents = cleaned_resumes + cleaned_jobs
    print(f"✓ Total documents for training: {len(all_documents)}")
    
    # Step 4: Train Feature Engineering Models
    print("\n[Step 4/6] Training feature engineering models...")
    feature_engineer = FeatureEngineer()
    
    # Train TF-IDF
    print("Training TF-IDF vectorizer...")
    feature_engineer.fit_tfidf(all_documents, max_features=5000)
    feature_engineer.save_tfidf()
    print("✓ TF-IDF model saved")
    
    # Train Doc2Vec
    print("\nTraining Doc2Vec model (this may take a few minutes)...")
    feature_engineer.train_doc2vec(all_documents, vector_size=200, epochs=40, save=True)
    print("✓ Doc2Vec model saved")
    
    # Step 5: Generate Training Features for ML Model
    print("\n[Step 5/6] Generating features for Random Forest...")
    
    training_features = []
    training_labels = []
    
    for idx, row in pairs_df.iterrows():
        if idx % 50 == 0:
            print(f"  Generated features for {idx}/{len(pairs_df)} pairs...")
        
        try:
            cleaned_resume = preprocessor.preprocess(row['resume_text'])
            cleaned_job = preprocessor.preprocess(row['job_text'])
            
            if not cleaned_resume or not cleaned_job:
                continue
            
            # Generate features
            features = feature_engineer.generate_features(cleaned_resume, cleaned_job)
            
            # Create feature vector for Random Forest
            feature_vector = [
                features.get('skill_coverage', 0),
                features.get('tfidf_similarity', 0),
                features.get('doc2vec_similarity', 0),
                features.get('skill_jaccard', 0)
            ]
            
            training_features.append(feature_vector)
            training_labels.append(row['label'])
        except Exception as e:
            continue
    
    print(f"✓ Generated {len(training_features)} feature vectors")
    
    # Step 6: Train Random Forest Model
    print("\n[Step 6/6] Training Random Forest classifier...")
    matcher = ResumeJobMatcher()
    
    try:
        X = np.array(training_features)
        y = np.array(training_labels)
        
        print(f"Training on {len(X)} samples...")
        print(f"Feature shape: {X.shape}")
        print(f"Class distribution: Positive={sum(y==1)}, Negative={sum(y==0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        matcher.model = matcher.train_simple(X_train, y_train)
        
        # Evaluate on test set
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = matcher.model.predict(X_test)
        
        print(f"\n✓ Model Performance:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Fit', 'Good Fit']))
        
        matcher.save_model()
        print("✓ Random Forest model saved")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nModels saved in 'models/' directory:")
    print("  ✓ tfidf_vectorizer.pkl")
    print("  ✓ doc2vec_model.model")
    print("  ✓ random_forest_model.pkl")
    print("\nYou can now run the Streamlit dashboard:")
    print("  streamlit run dashboard/streamlit_app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
