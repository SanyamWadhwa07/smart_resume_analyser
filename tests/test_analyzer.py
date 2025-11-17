"""
Quick test script for Smart Resume Analyzer
Tests the complete pipeline with sample data
"""

import sys
sys.path.insert(0, 'src')

from data_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ResumeJobMatcher

# Sample resume
SAMPLE_RESUME = """
John Smith
Senior Software Engineer
Email: john.smith@email.com | Phone: +1-555-0123

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development, machine learning,
and cloud technologies. Proven track record of building scalable applications.

TECHNICAL SKILLS
- Programming: Python, Java, JavaScript, TypeScript, SQL
- Web: React, Node.js, Django, Flask, REST API
- ML/AI: TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy
- Cloud: AWS, Docker, Kubernetes, CI/CD
- Database: PostgreSQL, MongoDB, Redis
- Tools: Git, Jira, Jenkins

EXPERIENCE
Senior Software Engineer | Tech Corp | 2020 - Present
- Built machine learning models for customer segmentation using Python and TensorFlow
- Developed REST APIs using Django and Flask, handling 10M+ requests/day
- Deployed microservices on AWS using Docker and Kubernetes
- Led team of 5 engineers in agile environment

Software Developer | StartupXYZ | 2018 - 2020
- Developed full-stack web applications using React and Node.js
- Implemented CI/CD pipelines with Jenkins and GitHub Actions
- Optimized database queries reducing response time by 40%

EDUCATION
Bachelor of Science in Computer Science
Stanford University | 2014 - 2018
GPA: 3.8/4.0

PROJECTS
- Built NLP chatbot using BERT and transformers
- Created data analytics dashboard with Tableau and Power BI
- Open source contributor to scikit-learn
"""

# Sample job description
SAMPLE_JOB = """
Machine Learning Engineer

ABOUT THE ROLE
We are seeking a talented Machine Learning Engineer to join our AI team. You will design,
develop, and deploy ML models at scale.

REQUIRED SKILLS
- Strong programming skills in Python
- Experience with TensorFlow, PyTorch, or Keras
- Knowledge of machine learning, deep learning, and NLP
- Proficiency with Pandas, NumPy, and Scikit-learn
- Experience with cloud platforms (AWS, Azure, or GCP)
- Understanding of Docker and Kubernetes

PREFERRED SKILLS
- Experience with transformers and BERT
- Knowledge of MLOps and model deployment
- Familiarity with Spark and big data technologies
- Experience with React for building dashboards

QUALIFICATIONS
- Bachelor's or Master's degree in Computer Science or related field
- 3+ years of experience in machine learning engineering
- Strong problem-solving and communication skills

WHAT WE OFFER
- Competitive salary and equity
- Remote-friendly work environment
- Learning and development budget
- Health insurance and 401k matching
"""

def main():
    print("=" * 80)
    print("SMART RESUME ANALYZER - QUICK TEST")
    print("=" * 80)
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer()
    matcher = ResumeJobMatcher()
    
    # Load models if they exist
    print("\n[2/4] Loading models...")
    try:
        feature_engineer.load_tfidf()
        feature_engineer.load_doc2vec()
        matcher.load_model()
        print("✓ All models loaded successfully")
    except Exception as e:
        print(f"⚠ Models not found. Please run train_models.py first.")
        print("  Training on sample data for demo...")
        
        # Quick training on sample data
        sample_docs = [SAMPLE_RESUME, SAMPLE_JOB]
        cleaned_docs = [preprocessor.preprocess(doc) for doc in sample_docs]
        
        feature_engineer.fit_tfidf(cleaned_docs)
        feature_engineer.train_doc2vec(cleaned_docs, epochs=20)
        print("✓ Quick models trained")
    
    # Preprocess texts
    print("\n[3/4] Preprocessing texts...")
    cleaned_resume = preprocessor.preprocess(SAMPLE_RESUME)
    cleaned_job = preprocessor.preprocess(SAMPLE_JOB)
    print(f"✓ Resume: {len(cleaned_resume)} chars")
    print(f"✓ Job: {len(cleaned_job)} chars")
    
    # Generate features
    print("\n[4/4] Analyzing resume-job match...")
    features = feature_engineer.generate_features(cleaned_resume, cleaned_job)
    
    # Display results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    print("\n📊 SIMILARITY SCORES:")
    print(f"  TF-IDF Similarity:    {features['tfidf_similarity']:.1%}")
    print(f"  Doc2Vec Similarity:   {features['doc2vec_similarity']:.1%}")
    print(f"  Skill Jaccard:        {features['skill_jaccard']:.1%}")
    print(f"  Skill Coverage:       {features['skill_coverage']:.1%}")
    
    print("\n✅ MATCHED SKILLS:")
    if features['matched_skills']:
        for skill in sorted(features['matched_skills']):
            print(f"  • {skill}")
    else:
        print("  (none)")
    
    print("\n❌ MISSING SKILLS:")
    if features['missing_skills']:
        for skill in sorted(features['missing_skills'])[:10]:
            print(f"  • {skill}")
        if len(features['missing_skills']) > 10:
            print(f"  ... and {len(features['missing_skills']) - 10} more")
    else:
        print("  (none)")
    
    print("\n📈 SKILL STATISTICS:")
    print(f"  Resume has {features['resume_skill_count']} skills")
    print(f"  Job requires {features['job_skill_count']} skills")
    print(f"  Matched {len(features['matched_skills'])} skills")
    print(f"  Missing {len(features['missing_skills'])} skills")
    
    # Try prediction if model is available
    if matcher.model is not None:
        try:
            # Create feature vector for prediction
            feature_dict = {
                'skill_match': features['skill_coverage'],
                'experience_gap': 0,  # Would need to extract from text
                'education_match': 1,
                'industry_match': 1
            }
            
            prediction = matcher.predict_job_fit(feature_dict)
            print("\n🎯 PREDICTION:")
            print(f"  Fit Score:      {prediction['fit_probability']:.1f}%")
            print(f"  Prediction:     {prediction['fit_label']}")
            print(f"  Confidence:     {prediction['confidence']:.1%}")
        except Exception as e:
            print(f"\n⚠ Prediction not available: {e}")
    
    print("\n💡 RECOMMENDATIONS:")
    
    if features['skill_coverage'] >= 0.8:
        print("  ✓ Excellent skill match! This candidate is highly qualified.")
    elif features['skill_coverage'] >= 0.6:
        print("  ⚠ Good skill match, but some gaps exist.")
        print("    Consider training or upskilling for missing skills.")
    else:
        print("  ⚠ Significant skill gaps detected.")
        print("    Recommend focusing on core technical skills first.")
    
    if features['missing_skills']:
        priority_skills = sorted(features['missing_skills'])[:5]
        print(f"\n  Priority skills to acquire:")
        for skill in priority_skills:
            print(f"    • {skill}")
    
    print("\n" + "=" * 80)
    print("Test complete! Ready to use the full dashboard.")
    print("Run: streamlit run dashboard/streamlit_app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
