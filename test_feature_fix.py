"""
Test script to verify the feature mismatch fix
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ResumeJobMatcher

def test_feature_compatibility():
    """Test if the model can handle features correctly"""
    print("="*60)
    print("Testing Feature Compatibility Fix")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer(model_dir='models')
    matcher = ResumeJobMatcher(model_dir='models')
    
    # Load models
    print("\n2. Loading models...")
    try:
        feature_engineer.load_tfidf()
        feature_engineer.load_doc2vec()
        matcher.load_model()
        print(f"   ✓ Models loaded successfully")
        print(f"   ✓ Model expects {matcher.model.n_features_in_} features")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        return
    
    # Sample resume and job description
    resume_text = """
    Python Developer with 5 years of experience in machine learning and data science.
    Proficient in TensorFlow, PyTorch, scikit-learn, pandas, and numpy.
    Experience with AWS, Docker, and CI/CD pipelines.
    Strong background in NLP and computer vision.
    """
    
    job_text = """
    Looking for a Python Machine Learning Engineer with expertise in deep learning.
    Required skills: Python, TensorFlow, PyTorch, scikit-learn, pandas, numpy.
    Experience with cloud platforms (AWS/Azure) and containerization (Docker).
    Knowledge of NLP is a plus.
    """
    
    # Preprocess
    print("\n3. Preprocessing texts...")
    cleaned_resume = preprocessor.preprocess(resume_text)
    cleaned_job = preprocessor.preprocess(job_text)
    print(f"   ✓ Texts preprocessed")
    
    # Generate features
    print("\n4. Generating features...")
    features = feature_engineer.generate_features(cleaned_resume, cleaned_job)
    print(f"   ✓ Features generated:")
    print(f"     - tfidf_similarity: {features.get('tfidf_similarity', 0):.3f}")
    print(f"     - doc2vec_similarity: {features.get('doc2vec_similarity', 0):.3f}")
    print(f"     - skill_jaccard: {features.get('skill_jaccard', 0):.3f}")
    print(f"     - skill_coverage: {features.get('skill_coverage', 0):.3f}")
    print(f"     - resume_skill_count: {features.get('resume_skill_count', 0)}")
    print(f"     - job_skill_count: {features.get('job_skill_count', 0)}")
    print(f"   Total features in dict: {len([k for k in features.keys() if k not in ['matched_skills', 'missing_skills']])}")
    
    # Predict
    print("\n5. Making prediction...")
    try:
        prediction = matcher.predict_job_fit(features)
        print(f"   ✓ Prediction successful!")
        print(f"     - Fit Probability: {prediction['fit_probability']:.1f}%")
        print(f"     - Fit Label: {prediction['fit_label']}")
        print(f"     - Confidence: {prediction['confidence']:.3f}")
        
        print("\n" + "="*60)
        print("✓ TEST PASSED - Feature compatibility fix working!")
        print("="*60)
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        print("\n" + "="*60)
        print("✗ TEST FAILED - Issue persists")
        print("="*60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_compatibility()
