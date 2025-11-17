"""
Utility Functions for Smart Resume - Job Fit Analyzer
Helper functions for data loading, validation, and common operations
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

def load_resume_dataset(filepath):
    """
    Load resume dataset from CSV
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} resumes from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_job_dataset(filepath):
    """
    Load job postings dataset from CSV
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} job postings from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_labeled_pairs(resume_df, job_df, positive_samples=500, negative_samples=500):
    """
    Create labeled resume-job pairs for training
    
    Args:
        resume_df (DataFrame): Resume dataset with 'category' column
        job_df (DataFrame): Job dataset with 'category' column
        positive_samples (int): Number of matching pairs
        negative_samples (int): Number of non-matching pairs
    
    Returns:
        DataFrame: Labeled pairs with columns [resume_text, job_text, fit]
    """
    pairs = []
    
    # Create positive examples (matching categories)
    for _ in range(positive_samples):
        # Select random category
        categories = resume_df['category'].unique()
        category = np.random.choice(categories)
        
        # Get resume and job from same category
        resume_sample = resume_df[resume_df['category'] == category].sample(1)
        job_sample = job_df[job_df['category'] == category].sample(1)
        
        pairs.append({
            'resume_text': resume_sample.iloc[0]['resume_text'],
            'job_text': job_sample.iloc[0]['job_description'],
            'fit': 1
        })
    
    # Create negative examples (different categories)
    for _ in range(negative_samples):
        # Select two different categories
        categories = resume_df['category'].unique()
        cat1, cat2 = np.random.choice(categories, 2, replace=False)
        
        # Get resume from one category, job from another
        resume_sample = resume_df[resume_df['category'] == cat1].sample(1)
        job_sample = job_df[job_df['category'] == cat2].sample(1)
        
        pairs.append({
            'resume_text': resume_sample.iloc[0]['resume_text'],
            'job_text': job_sample.iloc[0]['job_description'],
            'fit': 0
        })
    
    df_pairs = pd.DataFrame(pairs)
    print(f"Created {len(df_pairs)} labeled pairs")
    print(f"Positive samples: {(df_pairs['fit']==1).sum()}")
    print(f"Negative samples: {(df_pairs['fit']==0).sum()}")
    
    return df_pairs

def validate_input_text(text, min_length=50, max_length=50000):
    """
    Validate input text
    
    Args:
        text (str): Input text
        min_length (int): Minimum required length
        max_length (int): Maximum allowed length
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text is empty or invalid"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    return True, "Valid"

def calculate_match_quality(features):
    """
    Calculate overall match quality score
    
    Args:
        features (dict): Feature dictionary
    
    Returns:
        str: Quality rating (Excellent/Good/Fair/Poor)
    """
    # Weighted average of features
    score = (
        features.get('tfidf_similarity', 0) * 0.25 +
        features.get('doc2vec_similarity', 0) * 0.25 +
        features.get('skill_jaccard', 0) * 0.2 +
        features.get('skill_coverage', 0) * 0.3
    )
    
    if score >= 0.75:
        return "Excellent"
    elif score >= 0.60:
        return "Good"
    elif score >= 0.40:
        return "Fair"
    else:
        return "Poor"

def generate_recommendations(features, prediction):
    """
    Generate actionable recommendations
    
    Args:
        features (dict): Feature dictionary with matched/missing skills
        prediction (dict): Prediction results
    
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    fit_prob = prediction.get('fit_probability', 0)
    missing_skills = features.get('missing_skills', [])
    skill_coverage = features.get('skill_coverage', 0)
    
    # Overall assessment
    if fit_prob >= 70:
        recommendations.append("✅ Strong candidate - Highly recommended for interview")
    elif fit_prob >= 50:
        recommendations.append("⚠️  Moderate match - Review experience and achievements")
    else:
        recommendations.append("❌ Weak match - May not meet core requirements")
    
    # Skill-based recommendations
    if skill_coverage < 0.5 and missing_skills:
        top_missing = missing_skills[:5]
        recommendations.append(
            f"🎯 Priority skills to acquire: {', '.join(top_missing)}"
        )
    
    if skill_coverage >= 0.8:
        recommendations.append("💪 Excellent skill match - Most requirements satisfied")
    
    # Improvement suggestions
    if features.get('tfidf_similarity', 0) < 0.5:
        recommendations.append(
            "📝 Consider tailoring resume to better match job keywords"
        )
    
    return recommendations

def export_results_to_csv(results, filename='results.csv'):
    """
    Export analysis results to CSV
    
    Args:
        results (list): List of result dictionaries
        filename (str): Output filename
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

def log_prediction(resume_id, job_id, prediction, features):
    """
    Log prediction for tracking and analytics
    
    Args:
        resume_id (str): Resume identifier
        job_id (str): Job identifier
        prediction (dict): Prediction results
        features (dict): Features used
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'resume_id': resume_id,
        'job_id': job_id,
        'fit_probability': prediction.get('fit_probability'),
        'fit_label': prediction.get('fit_label'),
        'tfidf_sim': features.get('tfidf_similarity'),
        'doc2vec_sim': features.get('doc2vec_similarity'),
        'skill_coverage': features.get('skill_coverage')
    }
    
    # Append to log file
    log_file = 'predictions.log'
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\\n')

def get_skill_recommendations(missing_skills, priority_count=5):
    """
    Get prioritized skill recommendations
    
    Args:
        missing_skills (list): List of missing skills
        priority_count (int): Number of top skills to recommend
    
    Returns:
        dict: Categorized skill recommendations
    """
    # Skill categories
    categories = {
        'Programming': ['python', 'java', 'javascript', 'c++', 'go', 'rust'],
        'ML/AI': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp'],
        'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
        'Data': ['sql', 'mongodb', 'pandas', 'numpy', 'spark'],
        'Web': ['react', 'angular', 'vue', 'node.js', 'django', 'flask']
    }
    
    recommendations = {}
    for category, skills in categories.items():
        missing_in_category = [s for s in missing_skills if s in skills]
        if missing_in_category:
            recommendations[category] = missing_in_category[:priority_count]
    
    return recommendations

def format_percentage(value):
    """Format decimal as percentage"""
    return f"{value * 100:.1f}%"

def format_skill_list(skills, max_display=10):
    """
    Format skill list for display
    
    Args:
        skills (list): List of skills
        max_display (int): Maximum skills to display
    
    Returns:
        str: Formatted skill string
    """
    if not skills:
        return "None"
    
    if len(skills) <= max_display:
        return ", ".join(skills)
    else:
        displayed = ", ".join(skills[:max_display])
        return f"{displayed} ... (+{len(skills) - max_display} more)"

def highlight_keywords(text, keywords, color="yellow"):
    """
    Highlight keywords in text with HTML markup
    
    Args:
        text (str): Original text
        keywords (list): List of keywords to highlight
        color (str): Background color for highlight
    
    Returns:
        str: HTML text with highlighted keywords
    """
    if not keywords:
        return text
    
    # Escape HTML in the original text first
    import html
    text = html.escape(text)
    
    # Sort keywords by length (longest first) to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Escape the keyword for HTML
        escaped_keyword = html.escape(keyword)
        # Case-insensitive replacement
        pattern = re.compile(re.escape(escaped_keyword), re.IGNORECASE)
        
        # Create replacement function to preserve original case
        def replace_func(match):
            original = match.group(0)
            return f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{original}</mark>'
        
        text = pattern.sub(replace_func, text)
    
    return text

def generate_comparison_text(resume_text, job_text, matched_skills, missing_skills):
    """
    Generate side-by-side comparison with highlighted skills
    
    Args:
        resume_text (str): Resume text
        job_text (str): Job description text
        matched_skills (list): Skills found in both
        missing_skills (list): Skills missing from resume
    
    Returns:
        tuple: (highlighted_resume, highlighted_job)
    """
    # Highlight matched skills in green
    highlighted_resume = highlight_keywords(resume_text, matched_skills, "lightgreen")
    highlighted_job_matched = highlight_keywords(job_text, matched_skills, "lightgreen")
    
    # Highlight missing skills in yellow (in job description only)
    highlighted_job = highlight_keywords(highlighted_job_matched, missing_skills, "yellow")
    
    return highlighted_resume, highlighted_job

# Data validation schemas
REQUIRED_RESUME_COLUMNS = ['resume_text', 'category']
REQUIRED_JOB_COLUMNS = ['job_description', 'category']

def validate_dataset(df, required_columns):
    """
    Validate dataset structure
    
    Args:
        df (DataFrame): Dataset to validate
        required_columns (list): Required column names
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check if DataFrame is valid
    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Invalid DataFrame"]
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("Dataset is empty")
    
    # Check for null values in required columns
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values")
    
    return len(errors) == 0, errors

# Constants
DEFAULT_SKILLS_LIST = [
    'python', 'java', 'javascript', 'sql', 'machine learning',
    'data science', 'aws', 'docker', 'kubernetes', 'react'
]

MATCH_THRESHOLDS = {
    'excellent': 0.75,
    'good': 0.60,
    'fair': 0.40
}