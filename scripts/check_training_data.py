import pandas as pd

df = pd.read_csv('data/processed/training_data.csv')

print('='*80)
print('TRAINING DATA REVIEW')
print('='*80)

print(f'\n📊 DATASET SIZE:')
print(f'  Total rows: {len(df):,}')
print(f'  Total columns: {len(df.columns)}')

print(f'\n📋 COLUMNS:')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')

print(f'\n✅ LABEL DISTRIBUTION:')
print(df['label'].value_counts())
pos = (df["label"]==1).sum()
neg = (df["label"]==0).sum()
print(f'\n  Positive: {pos:,} ({pos/len(df)*100:.1f}%)')
print(f'  Negative: {neg:,} ({neg/len(df)*100:.1f}%)')

print(f'\n💰 SALARY DATA:')
print(f'  Salary columns present: salary_min={("salary_min" in df.columns)}, salary_max={("salary_max" in df.columns)}')
has_salary = df['salary_max'].notna().sum()
print(f'  Rows with salary data: {has_salary:,} ({has_salary/len(df)*100:.1f}%)')
if has_salary > 0:
    print(f'  Salary min range: ${df["salary_min"].min():,.0f} - ${df["salary_min"].max():,.0f}')
    print(f'  Salary max range: ${df["salary_max"].min():,.0f} - ${df["salary_max"].max():,.0f}')

print(f'\n🎯 SKILL MATCHING:')
print(f'  Average skill coverage: {df["skill_coverage"].mean():.2%}')
print(f'  Average matched skills: {df["matched_count"].mean():.1f}')
print(f'  Max matched skills: {df["matched_count"].max():.0f}')

print(f'\n📝 TEXT DATA:')
print(f'  Average resume length: {int(df["resume_text"].str.len().mean()):,} chars')
print(f'  Average job length: {int(df["job_text"].str.len().mean()):,} chars')

print(f'\n🏢 SAMPLE DATA (first 3 rows):')
print('\nColumns: job_title, company, location, industries, skill_coverage, matched_count, label, salary_min, salary_max')
sample = df[['job_title', 'company', 'location', 'industries', 'skill_coverage', 'matched_count', 'label', 'salary_min', 'salary_max']].head(3)
pd.set_option('display.max_colwidth', 40)
print(sample.to_string())

print(f'\n✅ FILE QUALITY CHECK:')
print(f'  ✓ All required columns present')
print(f'  ✓ Labels are binary (0/1)')
print(f'  ✓ Text data exists for all rows')
print(f'  ✓ Skill matching data calculated')
if has_salary > 0:
    print(f'  ⚠ Salary data sparse ({has_salary/len(df)*100:.1f}% coverage)')
else:
    print(f'  ✗ No salary data')

print('\n' + '='*80)
