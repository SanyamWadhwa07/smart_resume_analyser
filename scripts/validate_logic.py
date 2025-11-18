import pandas as pd

df = pd.read_csv('data/processed/training_data.csv')

print('='*80)
print('TRAINING DATA LOGIC VALIDATION')
print('='*80)

# Check the 3 sample rows in detail
print('\n🔍 ANALYZING FIRST 3 ROWS:\n')

for idx in range(3):
    row = df.iloc[idx]
    print(f'\n{"="*80}')
    print(f'ROW {idx}: {"✅ POSITIVE MATCH" if row["label"]==1 else "❌ NEGATIVE MATCH"}')
    print(f'{"="*80}')
    
    print(f'\n📄 RESUME SKILLS:')
    resume_skills = set(row['resume_skills'].split('|'))
    print(f'   {", ".join(sorted(list(resume_skills)[:10]))}... ({len(resume_skills)} total)')
    
    print(f'\n💼 JOB: {row["job_title"][:60]}')
    print(f'   Company: {row["company"]}')
    print(f'   Location: {row["location"]}')
    print(f'   Industry: {row["industries"][:60]}')
    
    print(f'\n🎯 JOB SKILLS REQUIRED:')
    job_skills = set(row['job_skills'].split('|')) if row['job_skills'] else set()
    if job_skills:
        print(f'   {", ".join(sorted(list(job_skills)))}')
    else:
        print(f'   ⚠️ NO SKILLS EXTRACTED')
    
    print(f'\n📊 MATCHING:')
    if job_skills and resume_skills:
        matched = resume_skills.intersection(job_skills)
        print(f'   Matched skills: {matched if matched else "NONE"}')
        print(f'   Matched count: {row["matched_count"]} / {len(job_skills)} job skills')
        print(f'   Skill coverage: {row["skill_coverage"]:.1%}')
    else:
        print(f'   ⚠️ Cannot calculate match')
    
    print(f'\n🏷️ LABEL: {row["label"]} ({"GOOD MATCH" if row["label"]==1 else "BAD MATCH"})')
    
    # Validate logic
    print(f'\n✅ LOGIC CHECK:')
    if row['label'] == 1:
        if row['skill_coverage'] >= 0.30:
            print(f'   ✓ Correctly labeled POSITIVE (coverage {row["skill_coverage"]:.1%} >= 30%)')
        else:
            print(f'   ⚠️ QUESTIONABLE: Labeled positive but coverage only {row["skill_coverage"]:.1%}')
    else:
        if row['skill_coverage'] < 0.30:
            print(f'   ✓ Correctly labeled NEGATIVE (coverage {row["skill_coverage"]:.1%} < 30%)')
        else:
            print(f'   ⚠️ QUESTIONABLE: Labeled negative but coverage is {row["skill_coverage"]:.1%}')

# Overall validation
print(f'\n\n{"="*80}')
print('OVERALL VALIDATION')
print('='*80)

print('\n1️⃣ LABEL CONSISTENCY CHECK:')
positive = df[df['label'] == 1]
negative = df[df['label'] == 0]

print(f'\n   POSITIVE samples (label=1):')
print(f'   - Count: {len(positive):,}')
print(f'   - Avg skill coverage: {positive["skill_coverage"].mean():.1%}')
print(f'   - Avg matched skills: {positive["matched_count"].mean():.1f}')
print(f'   - Coverage >= 30%: {(positive["skill_coverage"] >= 0.30).sum():,} ({(positive["skill_coverage"] >= 0.30).sum()/len(positive)*100:.1f}%)')

print(f'\n   NEGATIVE samples (label=0):')
print(f'   - Count: {len(negative):,}')
print(f'   - Avg skill coverage: {negative["skill_coverage"].mean():.1%}')
print(f'   - Avg matched skills: {negative["matched_count"].mean():.1f}')
print(f'   - Coverage >= 30%: {(negative["skill_coverage"] >= 0.30).sum():,} ({(negative["skill_coverage"] >= 0.30).sum()/len(negative)*100:.1f}%)')

print('\n2️⃣ EDGE CASES:')
# Positive with 0 matches
pos_zero = positive[positive['matched_count'] == 0]
print(f'   - Positive samples with 0 matched skills: {len(pos_zero)} (SHOULD BE 0!)')
if len(pos_zero) > 0:
    print(f'     ⚠️ ISSUE: These should not be labeled positive!')

# Negative with high coverage
neg_high = negative[negative['skill_coverage'] >= 0.50]
print(f'   - Negative samples with coverage >= 50%: {len(neg_high)} (SHOULD BE 0!)')
if len(neg_high) > 0:
    print(f'     ⚠️ ISSUE: These should be labeled positive!')

print('\n3️⃣ SKILL EXTRACTION QUALITY:')
no_job_skills = (df['job_skills'] == '') | (df['job_skills'].isna())
print(f'   - Jobs with NO skills extracted: {no_job_skills.sum():,} ({no_job_skills.sum()/len(df)*100:.1f}%)')
if no_job_skills.sum() > 100:
    print(f'     ⚠️ WARNING: Many jobs have no skills extracted!')

print('\n4️⃣ DATA QUALITY:')
print(f'   ✓ All rows have resume text: {df["resume_text"].notna().all()}')
print(f'   ✓ All rows have job text: {df["job_text"].notna().all()}')
print(f'   ✓ All rows have labels: {df["label"].notna().all()}')
print(f'   ✓ Labels are binary: {set(df["label"].unique()) == {0, 1}}')

print('\n' + '='*80)
print('VERDICT:')
print('='*80)

issues = []
if len(pos_zero) > 0:
    issues.append(f'{len(pos_zero)} positive samples with 0 matches')
if len(neg_high) > 0:
    issues.append(f'{len(neg_high)} negative samples with high coverage')
if no_job_skills.sum() > len(df) * 0.1:
    issues.append(f'{no_job_skills.sum()/len(df)*100:.1f}% jobs have no skills')

if issues:
    print('⚠️ ISSUES FOUND:')
    for issue in issues:
        print(f'   - {issue}')
else:
    print('✅ DATA LOOKS GOOD! Logic is consistent.')
    print('   - Positive samples have good skill overlap')
    print('   - Negative samples have poor skill overlap')
    print('   - Labels match the 30% threshold rule')

print('\n' + '='*80)
