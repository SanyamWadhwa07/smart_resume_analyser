import numpy as np

def skill_match_score(resume_skills, job_skills):
    """Compute skill match score as Jaccard similarity."""
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    if not job_set:
        return 0.0
    return len(resume_set & job_set) / len(job_set)

def experience_gap(resume_exp, job_exp):
    """Compute experience gap (years)."""
    try:
        resume_exp = float(resume_exp)
        job_exp = float(job_exp)
        return resume_exp - job_exp
    except Exception:
        return np.nan

def education_match(resume_edu, job_edu):
    """Simple education match (1 if matches, else 0)."""
    return int(str(resume_edu).strip().lower() == str(job_edu).strip().lower())

def industry_match(resume_ind, job_ind):
    """Simple industry match (1 if matches, else 0)."""
    return int(str(resume_ind).strip().lower() == str(job_ind).strip().lower())

def compute_features(resume_row, job_row):
    """Compute all features for a resume-job pair."""
    features = {}
    features['skill_match'] = skill_match_score(resume_row.get('extracted_skills', []), job_row.get('extracted_skills', []))
    features['experience_gap'] = experience_gap(resume_row.get('Experience', 0), job_row.get('Min_Experience', 0))
    features['education_match'] = education_match(resume_row.get('Education', ''), job_row.get('Education', ''))
    features['industry_match'] = industry_match(resume_row.get('Industry', ''), job_row.get('Industry', ''))
    return features
"""
Feature Engineering Module for Smart Resume - Job Fit Analyzer
Implements TF-IDF, Doc2Vec, NER, and similarity calculations
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
import pickle
import os

class FeatureEngineer:
    """
    Feature engineering for resume-job matching
    Combines basic NLP (TF-IDF) with advanced encoding (Doc2Vec, NER)
    """
    def __init__(self, model_dir='models'):
        """Initialize feature engineer"""
        self.tfidf_vectorizer = None
        self.doc2vec_model = None
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Warning: spaCy model not loaded")
            self.nlp = None
        
        # Comprehensive skills list (200+ technical skills)
        self.skills_list = self._load_skills_database()
    
    def _load_skills_database(self):
        """Load comprehensive technical skills database"""
        return [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
            'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'fastapi', 'spring', 'asp.net', 'jquery',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
            'oracle', 'sqlite', 'dynamodb', 'elasticsearch', 'neo4j',
            
            # Machine Learning & AI
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'neural networks', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'xgboost', 'lightgbm', 'opencv', 'transformers', 'bert', 'gpt',
            
            # Data Science
            'data science', 'data analysis', 'pandas', 'numpy', 'scipy',
            'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi',
            'jupyter', 'statistics', 'analytics',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd',
            'terraform', 'ansible', 'git', 'github', 'gitlab', 'bitbucket',
            'linux', 'unix', 'bash',
            
            # Big Data
            'hadoop', 'spark', 'kafka', 'airflow', 'hive', 'pig', 'storm',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin',
            
            # Testing & Quality
            'selenium', 'junit', 'pytest', 'testing', 'quality assurance',
            'test automation', 'agile', 'scrum', 'jira',
            
            # Other Technologies
            'rest api', 'graphql', 'microservices', 'redis', 'rabbitmq',
            'blockchain', 'solidity', 'etl', 'data warehousing',
            
            # Soft Skills
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'analytical', 'critical thinking'
        ]
    
    # ===== BASIC NLP FEATURES =====
    
    def fit_tfidf(self, documents, max_features=7000):
        """
        Fit TF-IDF vectorizer on document corpus
        
        Args:
            documents (list): List of preprocessed text documents
            max_features (int): Maximum number of features
        
        Returns:
            TfidfVectorizer: Fitted vectorizer
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            norm='l2'
        )
        
        self.tfidf_vectorizer.fit(documents)
        print(f"TF-IDF vectorizer fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
        
        return self.tfidf_vectorizer
    
    def calculate_tfidf_similarity(self, resume_text, job_desc_text):
        """
        Calculate TF-IDF based cosine similarity
        
        Args:
            resume_text (str): Preprocessed resume text
            job_desc_text (str): Preprocessed job description text
        
        Returns:
            float: Cosine similarity score (0-1)
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        # Transform texts to TF-IDF vectors
        resume_vec = self.tfidf_vectorizer.transform([resume_text])
        job_vec = self.tfidf_vectorizer.transform([job_desc_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vec, job_vec)[0][0]
        
        return float(similarity)
    
    # ===== ADVANCED ENCODING FEATURES =====
    
    def train_doc2vec(self, documents, vector_size=200, epochs=50, save=True):
        """
        Train Doc2Vec model for semantic embeddings
        
        Args:
            documents (list): List of preprocessed documents
            vector_size (int): Dimensionality of embeddings
            epochs (int): Number of training epochs
            save (bool): Whether to save the model
        
        Returns:
            Doc2Vec: Trained model
        """
        # Prepare tagged documents
        tagged_docs = [
            TaggedDocument(words=doc.split(), tags=[str(i)]) 
            for i, doc in enumerate(documents)
        ]
        
        # Train Doc2Vec model
        print(f"Training Doc2Vec model on {len(documents)} documents...")
        self.doc2vec_model = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            window=10,
            min_count=2,
            workers=4,
            epochs=epochs,
            dm=1  # PV-DM algorithm
        )
        
        print(f"Doc2Vec model trained with {vector_size}-dimensional vectors")
        
        if save:
            model_path = os.path.join(self.model_dir, 'doc2vec_model.model')
            self.doc2vec_model.save(model_path)
            print(f"Model saved to {model_path}")
        
        return self.doc2vec_model
    
    def calculate_doc2vec_similarity(self, resume_text, job_desc_text):
        """
        Calculate Doc2Vec based semantic similarity
        
        Args:
            resume_text (str): Preprocessed resume text
            job_desc_text (str): Preprocessed job description text
        
        Returns:
            float: Cosine similarity score (0-1)
        """
        if self.doc2vec_model is None:
            raise ValueError("Doc2Vec model not trained. Call train_doc2vec() first.")
        
        # Infer vectors for both texts
        resume_vec = self.doc2vec_model.infer_vector(resume_text.split())
        job_vec = self.doc2vec_model.infer_vector(job_desc_text.split())
        
        # Calculate cosine similarity
        similarity = np.dot(resume_vec, job_vec) / (
            np.linalg.norm(resume_vec) * np.linalg.norm(job_vec)
        )
        
        return float(similarity)
    
    # ===== SKILL EXTRACTION & MATCHING =====
    
    def extract_skills_ner(self, text, use_custom_list=True):
        """
        Extract skills using NER and predefined skills list
        
        Args:
            text (str): Text to extract skills from
            use_custom_list (bool): Whether to use custom skills database
        
        Returns:
            list: Extracted skills
        """
        text_lower = text.lower()
        extracted_skills = set()
        
        # Extract using predefined skills list
        if use_custom_list:
            for skill in self.skills_list:
                if skill.lower() in text_lower:
                    extracted_skills.add(skill)
        
        # Extract using spaCy NER (optional enhancement)
        if self.nlp:
            try:
                doc = self.nlp(text[:1000000])  # Limit text length
                for ent in doc.ents:
                    # Extract PRODUCT, ORG entities that might be technologies
                    if ent.label_ in ['PRODUCT', 'ORG']:
                        skill = ent.text.lower()
                        if len(skill) > 2:
                            extracted_skills.add(skill)
            except:
                pass
        
        return list(extracted_skills)
    
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """
        Calculate skill match score using Jaccard similarity
        
        Args:
            resume_skills (list): Skills extracted from resume
            job_skills (list): Skills extracted from job description
        
        Returns:
            dict: Match scores and skill lists
        """
        resume_set = set([s.lower() for s in resume_skills])
        job_set = set([s.lower() for s in job_skills])
        
        if len(job_set) == 0:
            return {
                'jaccard_similarity': 0.0,
                'skill_coverage': 0.0,
                'matched_skills': [],
                'missing_skills': []
            }
        
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection = resume_set.intersection(job_set)
        union = resume_set.union(job_set)
        
        jaccard_score = len(intersection) / len(union) if len(union) > 0 else 0
        
        # Skill coverage: percentage of required skills present
        skill_coverage = len(intersection) / len(job_set) if len(job_set) > 0 else 0
        
        return {
            'jaccard_similarity': float(jaccard_score),
            'skill_coverage': float(skill_coverage),
            'matched_skills': sorted(list(intersection)),
            'missing_skills': sorted(list(job_set - resume_set))
        }
    
    # ===== COMPLETE FEATURE GENERATION =====
    
    def generate_features(self, resume_text, job_desc_text):
        """
        Generate complete feature vector for ML model
        
        Args:
            resume_text (str): Preprocessed resume text
            job_desc_text (str): Preprocessed job description text
        
        Returns:
            dict: All features including similarities and skills
        """
        features = {}
        
        # Basic NLP features
        try:
            features['tfidf_similarity'] = self.calculate_tfidf_similarity(
                resume_text, job_desc_text
            )
        except:
            features['tfidf_similarity'] = 0.0
        
        # Advanced encoding features
        try:
            features['doc2vec_similarity'] = self.calculate_doc2vec_similarity(
                resume_text, job_desc_text
            )
        except:
            features['doc2vec_similarity'] = 0.0
        
        # Skill-based features
        resume_skills = self.extract_skills_ner(resume_text)
        job_skills = self.extract_skills_ner(job_desc_text)
        skill_scores = self.calculate_skill_match_score(resume_skills, job_skills)
        
        features['skill_jaccard'] = skill_scores['jaccard_similarity']
        features['skill_coverage'] = skill_scores['skill_coverage']
        features['matched_skills'] = skill_scores['matched_skills']
        features['missing_skills'] = skill_scores['missing_skills']
        
        # Additional features (optional)
        features['resume_skill_count'] = len(resume_skills)
        features['job_skill_count'] = len(job_skills)
        
        return features
    
    # ===== MODEL PERSISTENCE =====
    
    def save_tfidf(self, filename='tfidf_vectorizer.pkl'):
        """Save TF-IDF vectorizer"""
        if self.tfidf_vectorizer:
            path = os.path.join(self.model_dir, filename)
            with open(path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"TF-IDF vectorizer saved to {path}")
    
    def load_tfidf(self, filename='tfidf_vectorizer.pkl'):
        """Load TF-IDF vectorizer"""
        path = os.path.join(self.model_dir, filename)
        with open(path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        print(f"TF-IDF vectorizer loaded from {path}")
        return self.tfidf_vectorizer
    
    def load_doc2vec(self, filename='doc2vec_model.model'):
        """Load Doc2Vec model"""
        path = os.path.join(self.model_dir, filename)
        self.doc2vec_model = Doc2Vec.load(path)
        print(f"Doc2Vec model loaded from {path}")
        return self.doc2vec_model


# Example usage
if __name__ == "__main__":
    # Initialize
    fe = FeatureEngineer()
    
    # Sample documents
    documents = [
        "python machine learning data science tensorflow pandas numpy",
        "java backend development spring boot microservices",
        "javascript react angular vue frontend development"
    ]
    
    # Train models
    fe.fit_tfidf(documents)
    fe.train_doc2vec(documents, epochs=20)
    
    # Generate features
    resume = "python data science machine learning tensorflow"
    job = "python machine learning engineer tensorflow pandas"
    
    features = fe.generate_features(resume, job)
    
    print("\\nGenerated Features:")
    print(f"TF-IDF Similarity: {features['tfidf_similarity']:.3f}")
    print(f"Doc2Vec Similarity: {features['doc2vec_similarity']:.3f}")
    print(f"Skill Coverage: {features['skill_coverage']:.3f}")
    print(f"Matched Skills: {features['matched_skills']}")
    print(f"Missing Skills: {features['missing_skills']}")