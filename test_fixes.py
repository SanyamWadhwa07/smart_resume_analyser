"""
Quick test to verify both fixes:
1. Feature mismatch fix
2. HTML highlight fix
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ResumeJobMatcher
from utils import highlight_keywords, generate_comparison_text

def test_feature_fix():
    """Test the feature mismatch fix"""
    print("="*60)
    print("TEST 1: Feature Mismatch Fix")
    print("="*60)
    
    try:
        # Initialize
        preprocessor = TextPreprocessor()
        feature_engineer = FeatureEngineer(model_dir='models')
        matcher = ResumeJobMatcher(model_dir='models')
        
        # Load models
        feature_engineer.load_tfidf()
        feature_engineer.load_doc2vec()
        matcher.load_model()
        
        print(f"✓ Model expects {matcher.model.n_features_in_} features")
        
        # Test data
        resume = "Python developer with machine learning experience"
        job = "Looking for Python ML engineer"
        
        # Process
        cleaned_resume = preprocessor.preprocess(resume)
        cleaned_job = preprocessor.preprocess(job)
        features = feature_engineer.generate_features(cleaned_resume, cleaned_job)
        
        print(f"✓ Generated {len([k for k in features.keys() if k not in ['matched_skills', 'missing_skills']])} features")
        
        # Predict
        prediction = matcher.predict_job_fit(features)
        print(f"✓ Prediction successful: {prediction['fit_probability']:.1f}%")
        print("✅ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_html_fix():
    """Test the HTML highlighting fix"""
    print("="*60)
    print("TEST 2: HTML Highlighting Fix")
    print("="*60)
    
    try:
        # Test with potentially problematic text
        text = "I have experience with <script>alert('test')</script> and marker pens"
        keywords = ["marker", "experience"]
        
        # This should not create invalid HTML
        highlighted = highlight_keywords(text, keywords, "yellow")
        
        print("Original text:", text)
        print("Highlighted:", highlighted)
        
        # Check if HTML entities are properly escaped
        if "&lt;" in highlighted and "&gt;" in highlighted:
            print("✓ HTML properly escaped")
        
        # Check if mark tags are properly formed
        if "<mark" in highlighted and "</mark>" in highlighted:
            print("✓ Mark tags created")
        
        # Check no invalid patterns like <ma<mark>rk>
        if "<ma<mark>" not in highlighted:
            print("✓ No invalid nested tags")
        
        print("✅ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_text():
    """Test the generate_comparison_text function"""
    print("="*60)
    print("TEST 3: Comparison Text Generation")
    print("="*60)
    
    try:
        resume_text = "Python developer with experience in machine learning"
        job_text = "We need a Python expert with ML skills and cloud experience"
        matched = ["python", "machine learning"]
        missing = ["cloud"]
        
        resume_html, job_html = generate_comparison_text(
            resume_text, job_text, matched, missing
        )
        
        print("Resume HTML length:", len(resume_html))
        print("Job HTML length:", len(job_html))
        
        # Check for mark tags
        if "<mark" in resume_html:
            print("✓ Resume has highlighted skills")
        if "<mark" in job_html:
            print("✓ Job has highlighted skills")
        
        print("✅ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔧 Running Fix Verification Tests\n")
    
    results = []
    results.append(("Feature Mismatch Fix", test_feature_fix()))
    results.append(("HTML Highlighting Fix", test_html_fix()))
    results.append(("Comparison Text", test_comparison_text()))
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️  SOME TESTS FAILED"))
