import pandas as pd
import os

import nltk


# Paths to data files
RESUME_PATH = os.path.join('data', 'raw', 'UpdatedResumeDataSet.csv')
JOB_POSTINGS_PATH = os.path.join('data', 'raw', 'jobPostings', 'postings.csv')
SKILLS_MAP_PATH = os.path.join('data', 'raw', 'jobPostings', 'mappings', 'skills.csv')
INDUSTRIES_MAP_PATH = os.path.join('data', 'raw', 'jobPostings', 'mappings', 'industries.csv')

def load_resumes():
    """Load and clean resume data."""
    df = pd.read_csv(RESUME_PATH)
    # Basic cleaning: drop duplicates, fill NaNs
    df = df.drop_duplicates().fillna("")
    return df

def load_job_postings():
    """Load and clean job postings data."""
    df = pd.read_csv(JOB_POSTINGS_PATH)
    df = df.drop_duplicates().fillna("")
    return df

def load_skills_mapping():
    """Load skills mapping file."""
    return pd.read_csv(SKILLS_MAP_PATH)

def load_industries_mapping():
    """Load industries mapping file."""
    return pd.read_csv(INDUSTRIES_MAP_PATH)

def preprocess_text(text):
    """Basic text normalization."""
    return str(text).strip().lower()

def extract_skills(text, skills_map=None):
    """Extract skills from text using comprehensive skills database."""
    text = preprocess_text(text)
    
    # Comprehensive technical skills list
    comprehensive_skills = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
        'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab',
        
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express',
        'django', 'flask', 'fastapi', 'spring', 'asp.net', 'jquery',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
        'oracle', 'sqlite', 'dynamodb', 'elasticsearch', 'neo4j',
        
        # Machine Learning & AI
        'machine learning', 'deep learning', 'nlp', 'computer vision',
        'neural networks', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
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
        'rest api', 'api', 'graphql', 'microservices', 'rabbitmq',
        'blockchain', 'solidity', 'etl', 'data warehousing',
        
        # Soft Skills
        'leadership', 'communication', 'teamwork', 'problem solving',
        'project management', 'analytical', 'critical thinking'
    ]
    
    skills = []
    for skill in comprehensive_skills:
        # Use word boundary matching for better accuracy
        if skill in text:
            skills.append(skill)
    
    return skills

def preprocess_resumes(resume_df, skills_map=None):
    """Add extracted skills column to resumes."""
    resume_df['extracted_skills'] = resume_df['Resume'].apply(lambda x: extract_skills(x))
    return resume_df

def preprocess_job_postings(job_df, skills_map=None):
    """Add extracted skills column to job postings."""
    job_df['extracted_skills'] = job_df['Job_Description'].apply(lambda x: extract_skills(x))
    return job_df
"""
Data Preprocessing Module for Smart Resume - Job Fit Analyzer
Handles PDF/text extraction, cleaning, tokenization, and lemmatization
"""

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import PyPDF2

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

class TextPreprocessor:
    """
    Comprehensive text preprocessing for resumes and job descriptions
    """
    def __init__(self):
        """Initialize preprocessor with required NLP tools"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Warning: NLTK stopwords not found. Run: nltk.download('stopwords')")
            self.stop_words = set()
        
    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from PDF file
        
        Args:
            pdf_file: File path (str) or file-like object (BytesIO)
        
        Returns:
            str: Extracted text
        """
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text() + ' '
            else:
                # Handle BytesIO or file-like objects (for Streamlit uploads)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + ' '
            
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """
        Extract text from DOCX file
        
        Args:
            docx_file: File path (str) or file-like object (BytesIO)
        
        Returns:
            str: Extracted text
        """
        if not DOCX_AVAILABLE:
            print("Warning: python-docx not installed. Install with: pip install python-docx")
            return ""
        
        try:
            doc = Document(docx_file)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    def extract_years_of_experience(self, text):
        """
        Extract years of experience from resume text
        
        Args:
            text (str): Resume text
        
        Returns:
            int: Years of experience (0 if not found)
        """
        text_lower = text.lower()
        
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:of)?\s*experience',
            r'(\d+)\+?\s*year\s+experience',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        return 0
    
    def extract_education_level(self, text):
        """
        Extract highest education level from resume text
        
        Args:
            text (str): Resume text
        
        Returns:
            str: Education level (PhD, Masters, Bachelors, etc.)
        """
        text_lower = text.lower()
        
        education_levels = {
            'phd': ['ph.d', 'phd', 'doctorate', 'doctoral'],
            'masters': ['master', 'mba', 'm.s', 'ms ', 'm.a', 'ma ', 'msc'],
            'bachelors': ['bachelor', 'b.s', 'bs ', 'b.a', 'ba ', 'b.tech', 'btech', 'undergraduate'],
            'associate': ['associate', 'a.s', 'a.a'],
            'diploma': ['diploma', 'certificate'],
            'high_school': ['high school', 'secondary']
        }
        
        for level, keywords in education_levels.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return level
        
        return 'unknown'
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
        
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\\S+@\\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\\+?\\d[\\d\\s\\-\\(\\)]{8,}\\d', '', text)
        
        # Remove dates (various formats)
        text = re.sub(r'\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}', '', text)
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text (str): Cleaned text
        
        Returns:
            str: Processed text with lemmatized tokens
        """
        try:
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stop words and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
            
            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Error in tokenization: {e}")
            # Fallback to simple processing
            words = text.split()
            return ' '.join([w for w in words if len(w) > 2])
    
    def extract_sections(self, text):
        """
        Extract common resume sections (optional enhancement)
        
        Args:
            text (str): Resume text
        
        Returns:
            dict: Sections with their content
        """
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'other': text
        }
        
        # Simple section detection using keywords
        text_lower = text.lower()
        
        # Skills section
        skills_start = max(
            text_lower.find('skills'),
            text_lower.find('technical skills'),
            text_lower.find('core competencies')
        )
        
        # Experience section
        exp_start = max(
            text_lower.find('experience'),
            text_lower.find('work history'),
            text_lower.find('employment')
        )
        
        # Education section
        edu_start = max(
            text_lower.find('education'),
            text_lower.find('academic'),
            text_lower.find('qualification')
        )
        
        return sections
    
    def preprocess(self, text, is_pdf=False, pdf_path=None):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text or None if processing PDF
            is_pdf (bool): Whether input is PDF
            pdf_path (str or file-like): Path to PDF or file object
        
        Returns:
            str: Fully preprocessed text
        """
        # Extract from PDF if needed
        if is_pdf and pdf_path:
            text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return ""
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize and lemmatize
        text = self.tokenize_and_lemmatize(text)
        
        return text
    
    def preprocess_batch(self, texts, is_pdf_list=None):
        """
        Preprocess multiple texts in batch
        
        Args:
            texts (list): List of text strings
            is_pdf_list (list): List of booleans indicating if each is PDF
        
        Returns:
            list: List of preprocessed texts
        """
        if is_pdf_list is None:
            is_pdf_list = [False] * len(texts)
        
        processed = []
        for text, is_pdf in zip(texts, is_pdf_list):
            processed.append(self.preprocess(text, is_pdf=is_pdf))
        
        return processed


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with sample text
    sample_resume = """
    John Doe
    Email: john.doe@email.com | Phone: +1-234-567-8900
    
    SKILLS:
    - Python, Java, JavaScript, SQL
    - Machine Learning, Deep Learning, NLP
    - AWS, Docker, Kubernetes
    
    EXPERIENCE:
    Software Engineer at Tech Corp (2020-2023)
    - Developed ML models for customer segmentation
    - Built REST APIs using Django and Flask
    - Deployed applications on AWS
    """
    
    # Preprocess
    cleaned = preprocessor.preprocess(sample_resume)
    print("Original length:", len(sample_resume))
    print("Processed length:", len(cleaned))
    print("\\nProcessed text (first 200 chars):")
    print(cleaned[:200])
    
    # Test with batch processing
    texts = [sample_resume, "Another resume text here..."]
    batch_processed = preprocessor.preprocess_batch(texts)
    print(f"\\nProcessed {len(batch_processed)} documents")