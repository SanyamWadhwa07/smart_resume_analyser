"""
Test the SkillExtractor to ensure it works correctly
"""
import sys
sys.path.append('src')

from skill_extractor import SkillExtractor

# Initialize
extractor = SkillExtractor()

# Test 1: Job description
job_desc = """
We are seeking a Senior Software Engineer with strong Python and JavaScript experience.
The ideal candidate will have:
- 5+ years of Python development
- Experience with React, Node.js, and Express
- Cloud experience (AWS, Docker, Kubernetes)
- Database skills (PostgreSQL, MongoDB, Redis)
- Machine learning knowledge (TensorFlow, PyTorch)
"""

print("="*80)
print("SKILL EXTRACTOR TEST")
print("="*80)

print("\nTest 1: Job Description")
print("-"*80)
print(job_desc[:200] + "...")
skills = extractor.extract_skills(job_desc)
print(f"\nExtracted {len(skills)} skills:")
for skill in sorted(skills):
    print(f"  • {skill}")

# Test 2: Resume text
resume_text = """
Experienced Full-Stack Developer with expertise in:
- Languages: Python, Java, JavaScript, TypeScript
- Frontend: React, Angular, Vue.js
- Backend: Django, Flask, Spring Boot
- DevOps: Docker, Kubernetes, Jenkins, GitLab CI
- Databases: MySQL, PostgreSQL, MongoDB
- Cloud: AWS (EC2, S3, Lambda), Azure
"""

print("\n\nTest 2: Resume Text")
print("-"*80)
print(resume_text[:200] + "...")
resume_skills = extractor.extract_skills(resume_text)
print(f"\nExtracted {len(resume_skills)} skills:")
for skill in sorted(resume_skills):
    print(f"  • {skill}")

# Test 3: Skill matching
print("\n\nTest 3: Skill Matching")
print("-"*80)
match_result = extractor.calculate_skill_match_score(resume_skills, skills)
print(f"Jaccard Similarity: {match_result['jaccard']:.2%}")
print(f"Coverage (% of job skills in resume): {match_result['coverage']:.2%}")
print(f"Matched Skills ({match_result['matched_count']}):")
for skill in sorted(match_result['matched_skills']):
    print(f"  ✓ {skill}")
print(f"\nMissing Skills ({len(match_result['missing_skills'])}):")
for skill in sorted(match_result['missing_skills']):
    print(f"  ✗ {skill}")

print("\n" + "="*80)
print("✓ Skill Extractor Test Complete!")
print("="*80)
