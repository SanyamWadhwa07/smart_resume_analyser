"""
MODEL TRAINING PIPELINE (Steps 5-7)
Loads preprocessed data from prepare_training_data.py
Trains feature models and Random Forest classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from src.feature_engineering import FeatureEngineer
from src.model_training import ResumeJobMatcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("="*80)
    print("MODEL TRAINING PIPELINE (Steps 5-7)")
    print("Loads from: data/processed/training_data.pkl")
    print("="*80)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed training data...")
    input_file = 'data/processed/training_data.pkl'
    
    if not os.path.exists(input_file):
        print(f"\n❌ ERROR: {input_file} not found!")
        print("Please run prepare_training_data.py first.")
        return
    
    with open(input_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"✓ Loaded {len(training_data):,} training pairs")
    print(f"  Positive: {(training_data['label']==1).sum():,}")
    print(f"  Negative: {(training_data['label']==0).sum():,}")
    
    # Step 5: Train Feature Models
    print("\n[Step 5/7] Training feature extraction models...")
    print("="*80)
    feature_engineer = FeatureEngineer()
    
    all_texts = list(training_data['resume_clean']) + list(training_data['job_clean'])
    
    print(f"  Training TF-IDF (3000 features)...")
    feature_engineer.fit_tfidf(all_texts, max_features=3000)
    feature_engineer.save_tfidf('tfidf_vectorizer.pkl')
    print(f"    ✓ TF-IDF model saved")
    
    print(f"\n  Training Doc2Vec (50 dimensions, 15 epochs)...")
    feature_engineer.train_doc2vec(all_texts, vector_size=50, epochs=15, save=True)
    print(f"    ✓ Doc2Vec model saved")
    
    # Step 6: Generate Features
    print("\n[Step 6/7] Generating feature vectors...")
    print("="*80)
    features = []
    
    for idx, row in training_data.iterrows():
        # Text similarity features
        tfidf_sim = feature_engineer.calculate_tfidf_similarity(
            row['resume_clean'], 
            row['job_clean']
        )
        doc2vec_sim = feature_engineer.calculate_doc2vec_similarity(
            row['resume_clean'], 
            row['job_clean']
        )
        
        # Skill matching features
        matched = len(row['resume_skills'].intersection(row['job_skills']))
        total_union = len(row['resume_skills'].union(row['job_skills']))
        
        skill_jaccard = matched / total_union if total_union > 0 else 0
        
        # Create feature vector (AVOID DATA LEAKAGE)
        features.append([
            tfidf_sim,
            doc2vec_sim,
            skill_jaccard,
        ])
        
        if (idx + 1) % 500 == 0:
            print(f"  Generated features for {idx + 1:,}/{len(training_data):,} pairs...")
    
    X = np.array(features)
    y = training_data['label'].values
    
    print(f"\n✓ Generated {len(X):,} feature vectors with {X.shape[1]} features each")
    print(f"  Features: TF-IDF similarity, Doc2Vec similarity, Skill Jaccard")
    
    # Step 7: Train Random Forest
    print("\n[Step 7/7] Training Random Forest classifier...")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Class distribution (train): Positive={sum(y_train==1):,}, Negative={sum(y_train==0):,}")
    
    matcher = ResumeJobMatcher()
    matcher.train_simple(X_train, y_train, test_size=None)
    
    # Evaluate on test set
    y_pred = matcher.model.predict(X_test)
    y_pred_proba = matcher.model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ON TEST SET:")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fit', 'Good Fit']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    matcher.save_model('random_forest_model.pkl')
    
    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS ON REAL JOBS:")
    print("="*80)
    
    # Get test indices
    train_indices, test_indices = train_test_split(
        training_data.index, test_size=0.2, random_state=42, 
        stratify=training_data['label']
    )
    
    for i in range(min(5, len(X_test))):
        pred_proba = matcher.model.predict_proba(X_test[i].reshape(1, -1))[0]
        pred_label = 1 if pred_proba[1] >= 0.5 else 0
        actual = y_test[i]
        
        test_idx = test_indices[i]
        job_title = training_data.loc[test_idx, 'job_title']
        company = training_data.loc[test_idx, 'company']
        location = training_data.loc[test_idx, 'location']
        matched = training_data.loc[test_idx, 'matched_count']
        coverage = training_data.loc[test_idx, 'skill_coverage']
        
        print(f"\n{i+1}. {job_title}")
        print(f"   Company: {company} | Location: {location}")
        print(f"   Skills: {matched} matched ({coverage*100:.0f}% coverage)")
        print(f"   Predicted: {'✓ GOOD FIT' if pred_label == 1 else '✗ NOT FIT'} ({pred_proba[pred_label]*100:.1f}% confidence)")
        print(f"   Actual: {'✓ GOOD FIT' if actual == 1 else '✗ NOT FIT'}")
        print(f"   Result: {'✅ CORRECT' if pred_label == actual else '❌ WRONG'}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Model Accuracy: {accuracy:.2%}")
    print(f"   Training Dataset: {len(training_data):,} balanced pairs")
    
    print(f"\n💾 Models saved:")
    print(f"   → models/tfidf_vectorizer.pkl")
    print(f"   → models/doc2vec_model.model")
    print(f"   → models/random_forest_model.pkl")
    
    print(f"\n🚀 Run the dashboard:")
    print(f"   streamlit run dashboard/streamlit_app.py")
    print("="*80)

if __name__ == "__main__":
    main()
