"""
Streamlit Dashboard for Smart Resume - Job Fit Analyzer
Interactive web interface for resume-job matching analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys
import os

# Add parent directory and src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

# Import custom modules
try:
    from src.data_preprocessing import TextPreprocessor, extract_skills
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ResumeJobMatcher
    from src.utils import (validate_input_text, generate_recommendations, 
                          format_percentage, format_skill_list,
                          highlight_keywords, generate_comparison_text)
except ImportError as e:
    st.error(f"⚠️ Required modules not found. Error: {e}")
    st.error("Please ensure the 'src' directory is in the correct location.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Smart Resume - Job Fit Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .skill-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .missing-skill-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_models():
    """Load models once and cache them"""
    preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer()
    matcher = ResumeJobMatcher()
    
    # Try to load pre-trained models
    try:
        feature_engineer.load_tfidf()
        feature_engineer.load_doc2vec()
        matcher.load_model()
        st.success("✅ Pre-trained models loaded successfully!")
    except Exception as e:
        st.warning("⚠️ Pre-trained models not found. Please train models first.")
        st.info("You can still use the app - models will be trained on first use.")
    
    return preprocessor, feature_engineer, matcher

@st.cache_data(show_spinner=False)
def analyze_resume_cached(resume_text, job_text, _preprocessor, _feature_engineer, _matcher):
    """
    Cached analysis function to speed up repeat analyses
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        _preprocessor: TextPreprocessor instance (underscore prevents caching)
        _feature_engineer: FeatureEngineer instance
        _matcher: ResumeJobMatcher instance
    
    Returns:
        tuple: (features, prediction)
    """
    # Preprocess texts
    cleaned_resume = _preprocessor.preprocess(resume_text)
    cleaned_job = _preprocessor.preprocess(job_text)
    
    # Generate features
    features = _feature_engineer.generate_features(cleaned_resume, cleaned_job)
    
    # Predict job fit
    prediction = _matcher.predict_job_fit(features)
    
    return features, prediction

# Initialize
try:
    preprocessor, feature_engineer, matcher = load_models()
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

def create_gauge_chart(value, title):
    """Create a gauge chart for job fit percentage"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_skill_chart(matched_skills, missing_skills):
    """Create a bar chart for skill comparison"""
    skills_data = pd.DataFrame({
        'Category': ['Matched Skills', 'Missing Skills'],
        'Count': [len(matched_skills), len(missing_skills)]
    })
    
    fig = px.bar(
        skills_data,
        x='Count',
        y='Category',
        orientation='h',
        color='Category',
        color_discrete_map={'Matched Skills': '#28a745', 'Missing Skills': '#dc3545'},
        title='Skill Match Overview',
        text='Count'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(showlegend=False, height=250)
    return fig

def create_feature_radar(features):
    """Create radar chart for feature visualization"""
    categories = ['TF-IDF', 'Doc2Vec', 'Skill Match', 'Skill Coverage']
    values = [
        features.get('tfidf_similarity', 0) * 100,
        features.get('doc2vec_similarity', 0) * 100,
        features.get('skill_jaccard', 0) * 100,
        features.get('skill_coverage', 0) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title='Feature Analysis',
        height=350
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Resume - Job Fit Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Automated Resume Screening using NLP & Machine Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        analysis_mode = st.radio(
            "Select Mode",
            ["Single Analysis", "Batch Analysis"],
            help="Single: Analyze one resume against one job. Batch: Rank multiple resumes."
        )
        
        st.markdown("---")
        st.header("📊 About")
        st.info("""
        **Technologies Used:**
        - TF-IDF for keyword matching
        - Doc2Vec for semantic similarity
        - spaCy NER for skill extraction
        - Random Forest for prediction
        
        **Features:**
        - Resume-Job fit scoring (0-100%)
        - Skill gap analysis
        - Batch candidate ranking
        - Real-time predictions
        """)
        
        st.markdown("---")
        st.header("📈 Model Info")
        if matcher.model is not None:
            st.success("✅ Model loaded")
            if hasattr(matcher, 'training_history'):
                metrics = matcher.training_history.get('metrics', {})
                if metrics:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2f}")
        else:
            st.warning("⚠️ No model loaded")
    
    # Main Content
    if analysis_mode == "Single Analysis":
        single_analysis_mode()
    else:
        batch_analysis_mode()

def single_analysis_mode():
    """Single resume-job analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Resume Upload")
        resume_input_method = st.radio(
            "Input Method", 
            ["Upload File", "Paste Text"], 
            key="resume_method",
            horizontal=True
        )
        
        resume_text = ""
        if resume_input_method == "Upload File":
            resume_file = st.file_uploader(
                "Upload Resume (PDF/TXT)", 
                type=['pdf', 'txt'], 
                key="resume_file"
            )
            if resume_file:
                if resume_file.type == "application/pdf":
                    resume_text = preprocessor.extract_text_from_pdf(resume_file)
                else:
                    resume_text = resume_file.read().decode('utf-8')
                
                if resume_text:
                    st.success(f"✅ File loaded ({len(resume_text)} characters)")
        else:
            resume_text = st.text_area(
                "Paste Resume Text", 
                height=300, 
                key="resume_text",
                placeholder="Paste your resume text here..."
            )
    
    with col2:
        st.subheader("💼 Job Description Upload")
        job_input_method = st.radio(
            "Input Method", 
            ["Upload File", "Paste Text"], 
            key="job_method",
            horizontal=True
        )
        
        job_text = ""
        if job_input_method == "Upload File":
            job_file = st.file_uploader(
                "Upload Job Description (PDF/TXT)", 
                type=['pdf', 'txt'], 
                key="job_file"
            )
            if job_file:
                if job_file.type == "application/pdf":
                    job_text = preprocessor.extract_text_from_pdf(job_file)
                else:
                    job_text = job_file.read().decode('utf-8')
                
                if job_text:
                    st.success(f"✅ File loaded ({len(job_text)} characters)")
        else:
            job_text = st.text_area(
                "Paste Job Description Text", 
                height=300, 
                key="job_text",
                placeholder="Paste job description here..."
            )
    
    # Analyze Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Job Fit", 
            type="primary", 
            use_container_width=True
        )
    
    if analyze_button:
        if resume_text and job_text:
            # Validate inputs
            is_valid_resume, resume_msg = validate_input_text(resume_text)
            is_valid_job, job_msg = validate_input_text(job_text)
            
            if not is_valid_resume:
                st.error(f"Resume Error: {resume_msg}")
                return
            if not is_valid_job:
                st.error(f"Job Description Error: {job_msg}")
                return
            
            with st.spinner("Analyzing... This may take a few moments..."):
                analyze_and_display(resume_text, job_text)
        else:
            st.error("⚠️ Please provide both resume and job description!")

def analyze_and_display(resume_text, job_text):
    """Perform analysis and display results"""
    
    try:
        # Use cached analysis
        with st.status("Processing texts...") as status:
            features, prediction = analyze_resume_cached(
                resume_text, 
                job_text, 
                preprocessor, 
                feature_engineer, 
                matcher
            )
            status.update(label="✅ Analysis complete!", state="complete")
        
        # Display Results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Row 1: Main Metrics
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Gauge Chart
            fig_gauge = create_gauge_chart(prediction['fit_probability'], "Job Fit Score")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prediction", prediction['fit_label'])
            st.metric("Confidence", f"{prediction['confidence']:.0%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("TF-IDF Score", format_percentage(features['tfidf_similarity']))
            st.metric("Semantic Score", format_percentage(features['doc2vec_similarity']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Feature Visualization
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_radar = create_feature_radar(features)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("📈 Detailed Scores")
            score_data = pd.DataFrame({
                'Metric': ['TF-IDF Similarity', 'Doc2Vec Similarity', 'Skill Jaccard', 'Skill Coverage'],
                'Score': [
                    features['tfidf_similarity'],
                    features['doc2vec_similarity'],
                    features['skill_jaccard'],
                    features['skill_coverage']
                ]
            })
            score_data['Percentage'] = score_data['Score'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(score_data, use_container_width=True, hide_index=True)
        
        # Row 3: Skills Analysis
        st.markdown("---")
        st.subheader("🎯 Skills Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Matched Skills")
            matched_count = len(features['matched_skills'])
            st.write(f"**Total: {matched_count} skills**")
            
            if features['matched_skills']:
                for skill in features['matched_skills']:
                    st.markdown(
                        f'<span class="skill-badge">{skill}</span>', 
                        unsafe_allow_html=True
                    )
            else:
                st.info("No matched skills found")
        
        with col2:
            st.markdown("#### ❌ Missing Skills")
            missing_count = len(features['missing_skills'])
            st.write(f"**Total: {missing_count} skills**")
            
            if features['missing_skills']:
                for skill in features['missing_skills'][:15]:  # Limit display
                    st.markdown(
                        f'<span class="missing-skill-badge">{skill}</span>', 
                        unsafe_allow_html=True
                    )
                if len(features['missing_skills']) > 15:
                    st.caption(f"... and {len(features['missing_skills']) - 15} more")
            else:
                st.success("All required skills present!")
        
        # Skill Chart
        if features['matched_skills'] or features['missing_skills']:
            fig_skills = create_skill_chart(features['matched_skills'], features['missing_skills'])
            st.plotly_chart(fig_skills, use_container_width=True)
        
        # Row 4: Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        recommendations = generate_recommendations(features, prediction)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(
                f'<div class="recommendation-box">{i}. {rec}</div>', 
                unsafe_allow_html=True
            )
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

def batch_analysis_mode():
    """Batch analysis for multiple resumes"""
    st.subheader("📊 Batch Resume Analysis")
    st.info("📝 Upload multiple resumes to rank them against a job description")
    
    # Job Description
    st.markdown("### Job Description")
    job_input_method = st.radio(
        "Input Method", 
        ["Upload File", "Paste Text"], 
        key="batch_job_method",
        horizontal=True
    )
    
    job_text = ""
    if job_input_method == "Upload File":
        job_file = st.file_uploader(
            "Upload Job Description (PDF/TXT)", 
            type=['pdf', 'txt'], 
            key="batch_job_file"
        )
        if job_file:
            if job_file.type == "application/pdf":
                job_text = preprocessor.extract_text_from_pdf(job_file)
            else:
                job_text = job_file.read().decode('utf-8')
    else:
        job_text = st.text_area(
            "Paste Job Description", 
            height=200, 
            key="batch_job_text"
        )
    
    # Multiple Resume Upload
    st.markdown("### Resume Upload")
    resume_files = st.file_uploader(
        "Upload Multiple Resumes (PDF/TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        key="batch_resumes"
    )
    
    if resume_files:
        st.success(f"✅ {len(resume_files)} resumes uploaded")
    
    # Analysis Button
    if st.button("📈 Rank Candidates", type="primary", use_container_width=True):
        if resume_files and job_text:
            with st.spinner(f"Analyzing {len(resume_files)} resumes..."):
                results = []
                progress_bar = st.progress(0)
                
                # Preprocess job description once
                cleaned_job = preprocessor.preprocess(job_text)
                
                for idx, resume_file in enumerate(resume_files):
                    try:
                        # Extract resume text
                        if resume_file.type == "application/pdf":
                            resume_text = preprocessor.extract_text_from_pdf(resume_file)
                        else:
                            resume_text = resume_file.read().decode('utf-8')
                        
                        # Preprocess
                        cleaned_resume = preprocessor.preprocess(resume_text)
                        
                        # Get features
                        features = feature_engineer.generate_features(cleaned_resume, cleaned_job)
                        prediction = matcher.predict_job_fit(features)
                        
                        results.append({
                            'Candidate': resume_file.name,
                            'Fit Score (%)': prediction['fit_probability'],
                            'Prediction': prediction['fit_label'],
                            'Confidence': prediction['confidence'] * 100,
                            'Skill Coverage (%)': features['skill_coverage'] * 100,
                            'TF-IDF Score (%)': features['tfidf_similarity'] * 100,
                            'Matched Skills': len(features['matched_skills']),
                            'Missing Skills': len(features['missing_skills'])
                        })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(resume_files))
                        
                    except Exception as e:
                        st.warning(f"Error processing {resume_file.name}: {str(e)}")
                
                # Clear progress bar
                progress_bar.empty()
                
                if results:
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('Fit Score (%)', ascending=False).reset_index(drop=True)
                    results_df.index += 1  # Start ranking from 1
                    results_df.index.name = 'Rank'
                    
                    # Display results
                    st.success(f"✅ Successfully analyzed {len(results)} candidates")
                    
                    # Summary Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Candidates", len(results))
                    col2.metric("Good Fit", len(results_df[results_df['Prediction'] == 'Good Fit']))
                    col3.metric("Avg Fit Score", f"{results_df['Fit Score (%)'].mean():.1f}%")
                    col4.metric("Top Candidate", f"{results_df.iloc[0]['Fit Score (%)']:.1f}%")
                    
                    # Results table
                    st.markdown("### 📋 Candidate Rankings")
                    st.dataframe(
                        results_df.style.background_gradient(
                            subset=['Fit Score (%)'], 
                            cmap='RdYlGn'
                        ).format({
                            'Fit Score (%)': '{:.1f}',
                            'Confidence': '{:.1f}',
                            'Skill Coverage (%)': '{:.1f}',
                            'TF-IDF Score (%)': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Top candidates visualization
                    st.markdown("### 🏆 Top 10 Candidates")
                    top_candidates = results_df.head(10).reset_index()
                    
                    fig = px.bar(
                        top_candidates,
                        x='Fit Score (%)',
                        y='Candidate',
                        orientation='h',
                        title='Top 10 Candidates by Fit Score',
                        color='Fit Score (%)',
                        color_continuous_scale='RdYlGn',
                        text='Fit Score (%)'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv()
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="candidate_rankings.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No results generated. Please check your inputs.")
        else:
            st.error("⚠️ Please provide both job description and resume files!")

if __name__ == "__main__":
    main()