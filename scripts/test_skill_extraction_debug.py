"""
Debug script to check why average matched skills is only 0.9
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skill_extractor import SkillExtractor
from src.data_loader import ResumeDataLoader
import pandas as pd

# Initialize
skill_extractor = SkillExtractor()
print(f"Skill Extractor initialized with {len(skill_extractor.skills_set)} skills")
print("="*80)

# Test 1: Sample resume
print("\nTEST 1: Sample Resume Skill Extraction")
print("="*80)
loader = ResumeDataLoader('data/raw/master_resumes.jsonl')
resumes = loader.load_resumes(limit=10)

for i, resume in enumerate(resumes[:3], 1):
    print(f"\n--- Resume {i} ---")
    
    # Get structured skills
    structured_skills = resume.get('skills', [])
    if isinstance(structured_skills, list):
        print(f"Structured skills ({len(structured_skills)}): {structured_skills[:5]}")
    else:
        print(f"Structured skills (dict): {list(structured_skills.keys())[:5]}")
    
    # Extract from text
    resume_text = loader.extract_text_from_resume(resume)
    extracted_skills = skill_extractor.extract_skills(resume_text)
    
    print(f"Extracted from text ({len(extracted_skills)}): {extracted_skills[:10]}")
    print(f"First 500 chars of resume: {resume_text[:500]}")

# Test 2: Sample job descriptions
print("\n\nTEST 2: Sample Job Skill Extraction")
print("="*80)
jobs_df = pd.read_csv('data/raw/jobPostings/postings.csv', nrows=1000)
jobs_df = jobs_df[jobs_df['description'].notna()]

for i in range(3):
    job = jobs_df.iloc[i]
    print(f"\n--- Job {i+1}: {job['title']} ---")
    
    job_text = f"{job['title']}. {job['description']}"
    extracted_skills = skill_extractor.extract_skills(job_text)
    
    print(f"Extracted skills ({len(extracted_skills)}): {extracted_skills[:15]}")
    print(f"First 500 chars: {job_text[:500]}")

# Test 3: Direct text with known skills
print("\n\nTEST 3: Known Skills Text")
print("="*80)
test_text = """
Senior Software Engineer with 5 years experience in Python, Java, and JavaScript.
Expert in React, Angular, Vue.js frameworks. Proficient in AWS, Azure, Docker, Kubernetes.
Strong background in machine learning, TensorFlow, PyTorch, and scikit-learn.
Database experience: MySQL, PostgreSQL, MongoDB, Redis.
DevOps: Jenkins, GitLab CI, Terraform, Ansible.
"""

extracted = skill_extractor.extract_skills(test_text)
print(f"Test text with obvious skills:")
print(f"Extracted ({len(extracted)}): {sorted(extracted)}")

# Test 4: Check if regex pattern is compiled
print("\n\nTEST 4: Regex Pattern Status")
print("="*80)
if hasattr(skill_extractor, '_compiled_pattern'):
    print("✓ Regex pattern IS compiled (optimized)")
    print(f"  Pattern length: {len(skill_extractor._compiled_pattern.pattern)} chars")
else:
    print("✗ Regex pattern NOT compiled (will be slow)")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
