"""
Data Loader Module for Smart Resume Analyzer
Handles loading and processing of JSONL resume dataset
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

class ResumeDataLoader:
    """
    Loader for structured JSONL resume data
    """
    def __init__(self, jsonl_path='data/raw/master_resumes.jsonl'):
        """Initialize data loader"""
        self.jsonl_path = jsonl_path
        self.resumes = []
        
    def load_resumes(self, limit=None):
        """
        Load resumes from JSONL file
        
        Args:
            limit (int): Maximum number of resumes to load
        
        Returns:
            list: List of resume dictionaries
        """
        self.resumes = []
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        resume = json.loads(line.strip())
                        self.resumes.append(resume)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue
            print(f"✓ Loaded {len(self.resumes)} resumes from {self.jsonl_path}")
        except Exception as e:
            print(f"✗ Error loading JSONL file: {e}")
        return self.resumes
    
    def extract_text_from_resume(self, resume_json):
        """
        Extract comprehensive text from structured JSON resume
        
        Args:
            resume_json (dict): Structured resume data
        
        Returns:
            str: Combined text from all sections
        """
        text_parts = []
        
        # Personal info & summary
        if 'personal_info' in resume_json:
            pi = resume_json['personal_info']
            summary = pi.get('summary', '')
            if summary and summary not in ['Unknown', 'Not Provided', '']:
                text_parts.append(summary)
            
            if pi.get('location'):
                loc = pi['location']
                city = loc.get('city', '')
                country = loc.get('country', '')
                if city and city not in ['Unknown', 'Not Provided']:
                    text_parts.append(city)
                if country and country not in ['Unknown', 'Not Provided']:
                    text_parts.append(country)
        
        # Experience
        if 'experience' in resume_json:
            for exp in resume_json['experience']:
                # Company and title
                company = exp.get('company', '')
                if company and company not in ['Unknown', 'Not Provided', '']:
                    text_parts.append(company)
                
                title = exp.get('title', '')
                if title and title not in ['Unknown', 'Not Provided', '']:
                    text_parts.append(title)
                
                # Responsibilities
                responsibilities = exp.get('responsibilities', [])
                for resp in responsibilities:
                    if resp and resp not in ['Unknown', 'Not Provided']:
                        text_parts.append(resp)
                
                # Technical environment
                if 'technical_environment' in exp:
                    tech_env = exp['technical_environment']
                    
                    # Technologies
                    for tech in tech_env.get('technologies', []):
                        if tech and tech not in ['Unknown', 'Not Provided']:
                            text_parts.append(tech)
                    
                    # Methodologies
                    for meth in tech_env.get('methodologies', []):
                        if meth and meth not in ['Unknown', 'Not Provided']:
                            text_parts.append(meth)
                    
                    # Tools
                    for tool in tech_env.get('tools', []):
                        if tool and tool not in ['Unknown', 'Not Provided']:
                            text_parts.append(tool)
        
        # Education
        if 'education' in resume_json:
            for edu in resume_json['education']:
                if 'degree' in edu:
                    degree = edu['degree']
                    level = degree.get('level', '')
                    field = degree.get('field', '')
                    if level and level not in ['Unknown', 'Not Provided']:
                        text_parts.append(level)
                    if field and field not in ['Unknown', 'Not Provided']:
                        text_parts.append(field)
                
                if 'institution' in edu:
                    inst_name = edu['institution'].get('name', '')
                    if inst_name and inst_name not in ['Unknown', 'Not Provided']:
                        text_parts.append(inst_name)
        
        # Skills
        if 'skills' in resume_json and 'technical' in resume_json['skills']:
            tech_skills = resume_json['skills']['technical']
            
            for category in ['programming_languages', 'frameworks', 'databases', 'cloud',
                           'project_management', 'automation', 'software_tools']:
                if category in tech_skills:
                    skills_list = tech_skills[category]
                    if isinstance(skills_list, list):
                        for skill in skills_list:
                            if isinstance(skill, dict):
                                skill_name = skill.get('name', '')
                                skill_level = skill.get('level', '')
                                if skill_name and skill_name not in ['Unknown', 'Not Provided']:
                                    text_parts.append(skill_name)
                                if skill_level and skill_level not in ['Unknown', 'Not Provided']:
                                    text_parts.append(skill_level)
                            elif skill and skill not in ['Unknown', 'Not Provided']:
                                text_parts.append(str(skill))
        
        # Projects
        if 'projects' in resume_json:
            for proj in resume_json['projects']:
                proj_name = proj.get('name', '')
                proj_desc = proj.get('description', '')
                proj_role = proj.get('role', '')
                
                if proj_name and proj_name not in ['Unknown', 'Not Provided']:
                    text_parts.append(proj_name)
                if proj_desc and proj_desc not in ['Unknown', 'Not Provided']:
                    text_parts.append(proj_desc)
                if proj_role and proj_role not in ['Unknown', 'Not Provided']:
                    text_parts.append(proj_role)
                
                # Project technologies
                for tech in proj.get('technologies', []):
                    if tech and tech not in ['Unknown', 'Not Provided']:
                        text_parts.append(tech)
        
        # Certifications
        if 'certifications' in resume_json:
            cert = resume_json['certifications']
            if isinstance(cert, str) and cert and cert not in ['Unknown', 'Not Provided', '']:
                text_parts.append(cert)
        
        # Join all parts
        combined_text = ' '.join([str(part) for part in text_parts if part])
        return combined_text
    
    def extract_skills_from_resume(self, resume_json):
        """
        Extract structured skills list from resume
        
        Args:
            resume_json (dict): Structured resume data
        
        Returns:
            list: List of skills
        """
        skills = set()
        
        if 'skills' in resume_json and 'technical' in resume_json['skills']:
            tech_skills = resume_json['skills']['technical']
            
            for category in ['programming_languages', 'frameworks', 'databases', 'cloud',
                           'project_management', 'automation', 'software_tools']:
                if category in tech_skills:
                    skills_list = tech_skills[category]
                    if isinstance(skills_list, list):
                        for skill in skills_list:
                            if isinstance(skill, dict):
                                skill_name = skill.get('name', '')
                                if skill_name and skill_name not in ['Unknown', 'Not Provided', '']:
                                    skills.add(skill_name.lower())
                            elif skill and skill not in ['Unknown', 'Not Provided', '']:
                                skills.add(str(skill).lower())
        
        # Also extract from technical environment in experience
        if 'experience' in resume_json:
            for exp in resume_json['experience']:
                if 'technical_environment' in exp:
                    tech_env = exp['technical_environment']
                    for tech in tech_env.get('technologies', []):
                        if tech and tech not in ['Unknown', 'Not Provided', '']:
                            skills.add(tech.lower())
        
        return list(skills)
    
    def get_experience_years(self, resume_json):
        """
        Calculate total years of experience from resume
        
        Args:
            resume_json (dict): Structured resume data
        
        Returns:
            float: Total years of experience
        """
        total_years = 0
        
        if 'experience' in resume_json:
            for exp in resume_json['experience']:
                if 'dates' in exp:
                    duration = exp['dates'].get('duration', '')
                    if duration and duration != 'Unknown':
                        # Parse duration string (e.g., "2 years", "15 months", "1 year 1 month")
                        years = 0
                        months = 0
                        
                        import re
                        year_match = re.search(r'(\d+)\s*years?', duration, re.IGNORECASE)
                        month_match = re.search(r'(\d+)\s*months?', duration, re.IGNORECASE)
                        
                        if year_match:
                            years = int(year_match.group(1))
                        if month_match:
                            months = int(month_match.group(1))
                        
                        total_years += years + (months / 12.0)
        
        return round(total_years, 1)
    
    def create_dataframe(self, resumes=None):
        """
        Create DataFrame from resume data
        
        Args:
            resumes (list): List of resume dictionaries (uses self.resumes if None)
        
        Returns:
            DataFrame: Processed resume data
        """
        if resumes is None:
            resumes = self.resumes
        
        data = []
        for resume in resumes:
            data.append({
                'resume_text': self.extract_text_from_resume(resume),
                'skills': self.extract_skills_from_resume(resume),
                'experience_years': self.get_experience_years(resume),
                'raw_json': resume
            })
        
        df = pd.DataFrame(data)
        print(f"✓ Created DataFrame with {len(df)} resumes")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Average text length: {df['resume_text'].str.len().mean():.0f} characters")
        print(f"  Average skills count: {df['skills'].str.len().mean():.1f}")
        print(f"  Average experience: {df['experience_years'].mean():.1f} years")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ResumeDataLoader()
    
    # Load resumes (limit to 100 for testing)
    resumes = loader.load_resumes(limit=100)
    
    # Create DataFrame
    df = loader.create_dataframe()
    
    # Display sample data
    print("\n=== Sample Resume Data ===")
    for i in range(min(3, len(df))):
        print(f"\nResume {i+1}:")
        print(f"  Text length: {len(df.iloc[i]['resume_text'])} chars")
        print(f"  Skills: {df.iloc[i]['skills'][:10]}")  # First 10 skills
        print(f"  Experience: {df.iloc[i]['experience_years']} years")
        print(f"  Text preview: {df.iloc[i]['resume_text'][:200]}...")
