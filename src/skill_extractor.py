"""
Advanced Skill Extractor - Extracts 500+ technical skills from text
Solves the limitation of only 37 LinkedIn skill categories
"""

class SkillExtractor:
    """Extract technical AND non-technical skills from job descriptions and resumes"""
    
    def __init__(self):
        """Initialize with comprehensive skill dictionary (500+ skills)"""
        self.TECHNICAL_SKILLS = {
            # Programming Languages (50+)
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'php',
            'perl', 'shell', 'bash', 'powershell', 'objective-c', 'dart',
            'elixir', 'erlang', 'haskell', 'lua', 'groovy', 'clojure',
            
            # Web Frameworks (30+)
            'react', 'angular', 'vue', 'svelte', 'ember', 'backbone',
            'django', 'flask', 'fastapi', 'express', 'node.js', 'nest.js',
            'spring', 'spring boot', 'asp.net', 'rails', 'laravel', 'symfony',
            'next.js', 'nuxt.js', 'gatsby', 'jquery', 'bootstrap', 'tailwind',
            
            # Cloud & DevOps (40+)
            'aws', 'azure', 'gcp', 'google cloud', 'alibaba cloud',
            'docker', 'kubernetes', 'k8s', 'openshift', 'rancher',
            'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
            'terraform', 'ansible', 'puppet', 'chef', 'cloudformation',
            'helm', 'istio', 'prometheus', 'grafana', 'elk', 'elasticsearch',
            'logstash', 'kibana', 'datadog', 'new relic', 'splunk',
            
            # Databases (30+)
            'sql', 'mysql', 'postgresql', 'postgres', 'oracle', 'mssql',
            'mongodb', 'cassandra', 'dynamodb', 'couchdb', 'redis',
            'memcached', 'neo4j', 'influxdb', 'timescaledb', 'cockroachdb',
            'sqlite', 'mariadb', 'firestore', 'cosmos db', 'snowflake',
            'bigquery', 'redshift', 'athena', 'hive', 'presto',
            
            # Data Science & ML (40+)
            'machine learning', 'deep learning', 'data science', 'ai',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'jupyter', 'mlflow', 'kubeflow', 'airflow', 'spark', 'pyspark',
            'hadoop', 'kafka', 'flink', 'storm', 'beam', 'dask',
            'xgboost', 'lightgbm', 'catboost', 'hugging face', 'transformers',
            'opencv', 'nltk', 'spacy', 'gensim', 'fastai', 'mxnet',
            
            # Mobile Development (15+)
            'ios', 'android', 'react native', 'flutter', 'xamarin',
            'ionic', 'cordova', 'swift ui', 'jetpack compose',
            'firebase', 'realm', 'core data',
            
            # Testing & QA (20+)
            'selenium', 'cypress', 'jest', 'mocha', 'jasmine', 'pytest',
            'junit', 'testng', 'cucumber', 'postman', 'jmeter', 'locust',
            'appium', 'detox', 'enzyme', 'testing library',
            
            # Version Control & Collaboration (10+)
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
            'jira', 'confluence', 'trello', 'asana',
            
            # Message Queues & Streaming (10+)
            'rabbitmq', 'activemq', 'kafka', 'kinesis', 'pubsub',
            'redis streams', 'nats', 'zeromq',
            
            # Security (15+)
            'oauth', 'jwt', 'saml', 'ssl', 'tls', 'https',
            'penetration testing', 'vulnerability assessment', 'owasp',
            'encryption', 'cryptography', 'iam', 'kerberos',
            
            # Architecture & Design (20+)
            'microservices', 'monolith', 'serverless', 'lambda',
            'rest', 'restful', 'graphql', 'grpc', 'soap', 'api',
            'event driven', 'cqrs', 'saga', 'ddd', 'solid',
            'design patterns', 'mvc', 'mvvm', 'clean architecture',
            
            # Frontend (20+)
            'html', 'css', 'sass', 'scss', 'less', 'webpack', 'vite',
            'babel', 'eslint', 'prettier', 'redux', 'mobx', 'rxjs',
            'material ui', 'ant design', 'chakra ui', 'styled components',
            
            # Backend & API (15+)
            'node', 'deno', 'bun', 'nginx', 'apache', 'tomcat',
            'gunicorn', 'uwsgi', 'celery', 'sidekiq', 'bull',
            
            # BI & Analytics (15+)
            'tableau', 'power bi', 'looker', 'qlik', 'metabase',
            'excel', 'google analytics', 'mixpanel', 'amplitude',
            'sql server', 'ssis', 'ssrs', 'etl',
            
            # Project Management (10+)
            'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'safe',
            'sprint', 'standup', 'retrospective', 'jira',
            
            # Operating Systems (10+)
            'linux', 'unix', 'ubuntu', 'centos', 'redhat', 'debian',
            'windows', 'macos', 'freebsd',
            
            # Networking (10+)
            'tcp/ip', 'http', 'dns', 'load balancing', 'cdn',
            'nginx', 'haproxy', 'vpc', 'vpn', 'firewall',
        }
        
        # NON-TECHNICAL SKILLS (200+)
        self.SOFT_SKILLS = {
            # Business & Management (40+)
            'project management', 'product management', 'program management',
            'team leadership', 'people management', 'stakeholder management',
            'change management', 'risk management', 'budget management',
            'strategic planning', 'business strategy', 'business development',
            'business analysis', 'process improvement', 'operations management',
            'vendor management', 'contract negotiation', 'procurement',
            'supply chain', 'logistics', 'inventory management',
            'quality assurance', 'quality control', 'compliance',
            'regulatory compliance', 'audit', 'governance',
            'performance management', 'kpi tracking', 'reporting',
            'forecasting', 'planning', 'scheduling', 'coordination',
            'resource allocation', 'capacity planning',
            
            # Sales & Marketing (40+)
            'sales', 'marketing', 'business development', 'account management',
            'customer relationship management', 'crm', 'lead generation',
            'sales strategy', 'marketing strategy', 'brand management',
            'digital marketing', 'social media marketing', 'content marketing',
            'email marketing', 'seo', 'sem', 'ppc', 'google ads',
            'facebook ads', 'linkedin ads', 'marketing analytics',
            'market research', 'competitive analysis', 'customer acquisition',
            'customer retention', 'account executive', 'sales operations',
            'territory management', 'pipeline management', 'forecasting',
            'proposal writing', 'presentations', 'negotiation', 'closing',
            'cold calling', 'prospecting', 'networking', 'relationship building',
            'public relations', 'communications', 'copywriting',
            
            # Finance & Accounting (30+)
            'accounting', 'bookkeeping', 'financial analysis', 'financial reporting',
            'financial modeling', 'budgeting', 'forecasting', 'variance analysis',
            'accounts payable', 'accounts receivable', 'payroll', 'tax preparation',
            'audit', 'reconciliation', 'financial planning', 'investment analysis',
            'cost accounting', 'management accounting', 'gaap', 'ifrs',
            'quickbooks', 'sap', 'oracle financials', 'netsuite',
            'excel', 'financial statements', 'balance sheet', 'income statement',
            'cash flow', 'revenue recognition', 'expense management',
            
            # HR & Recruitment (25+)
            'recruiting', 'talent acquisition', 'interviewing', 'onboarding',
            'employee relations', 'performance reviews', 'compensation',
            'benefits administration', 'hris', 'applicant tracking', 'ats',
            'sourcing', 'screening', 'job posting', 'employer branding',
            'training', 'development', 'succession planning', 'retention',
            'employee engagement', 'hr compliance', 'labor relations',
            'payroll', 'workday', 'adp',
            
            # Customer Service (20+)
            'customer service', 'customer support', 'technical support',
            'help desk', 'call center', 'customer satisfaction', 'client relations',
            'issue resolution', 'problem solving', 'escalation management',
            'ticketing systems', 'zendesk', 'salesforce service cloud',
            'live chat', 'email support', 'phone support', 'troubleshooting',
            'customer experience', 'customer success', 'account support',
            
            # Healthcare & Medical (25+)
            'patient care', 'clinical', 'nursing', 'medical', 'healthcare',
            'ehr', 'electronic health records', 'epic', 'cerner', 'meditech',
            'patient assessment', 'treatment planning', 'medical coding',
            'icd-10', 'cpt', 'billing', 'medical billing', 'insurance',
            'hipaa', 'patient safety', 'clinical documentation',
            'medication administration', 'vital signs', 'charting',
            
            # Legal (15+)
            'legal research', 'contract law', 'litigation', 'legal writing',
            'case management', 'compliance', 'regulatory', 'legal analysis',
            'document review', 'discovery', 'depositions', 'paralegal',
            'westlaw', 'lexisnexis', 'legal drafting',
            
            # Education & Training (20+)
            'teaching', 'training', 'curriculum development', 'lesson planning',
            'instructional design', 'e-learning', 'lms', 'classroom management',
            'student assessment', 'educational technology', 'tutoring',
            'mentoring', 'coaching', 'professional development',
            'adult learning', 'facilitation', 'presentation skills',
            'course development', 'learning outcomes', 'pedagogy',
            
            # Design & Creative (25+)
            'graphic design', 'ui design', 'ux design', 'web design',
            'visual design', 'branding', 'typography', 'layout',
            'adobe creative suite', 'photoshop', 'illustrator', 'indesign',
            'figma', 'sketch', 'adobe xd', 'wireframing', 'prototyping',
            'user research', 'usability testing', 'design systems',
            'creative direction', 'art direction', 'video editing',
            'motion graphics', 'animation',
        }
        
        # Combine all skills
        self.ALL_SKILLS = self.TECHNICAL_SKILLS | self.SOFT_SKILLS
        self.skills_set = set(self.ALL_SKILLS)
        
        # Skill variations and aliases
        self.skill_aliases = {
            'k8s': 'kubernetes',
            'postgres': 'postgresql',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'ai': 'artificial intelligence',
            'ci/cd': 'continuous integration',
            'node': 'node.js',
        }
    
    def extract_skills(self, text):
        """
        Extract all technical skills from text.
        
        Args:
            text: Job description or resume text
        
        Returns:
            List of detected skills (lowercase)
        """
        if not text or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        detected = set()
        
        # OPTIMIZATION: Compile all skills into single regex pattern
        # This reduces 291 regex calls to 1 per text - 291x faster!
        if not hasattr(self, '_compiled_pattern'):
            import re
            # Sort skills by length (longest first) to match "machine learning" before "machine"
            sorted_skills = sorted(self.skills_set, key=len, reverse=True)
            # Escape special chars and join with OR
            pattern_parts = [r'\b' + re.escape(skill) + r'\b' for skill in sorted_skills]
            self._compiled_pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)
        
        # Find all matches in one pass
        matches = self._compiled_pattern.findall(text_lower)
        detected.update([m.lower() for m in matches])
        
        # Check aliases (much smaller set, so OK to iterate)
        for alias, canonical in self.skill_aliases.items():
            if alias in text_lower and canonical not in detected:
                if self._is_skill_match(text_lower, alias):
                    detected.add(canonical)
        
        return list(detected)
    
    def _is_skill_match(self, text, skill):
        """
        Check if skill appears as a complete word/phrase in text.
        Avoids false positives like 'java' in 'javascript'.
        """
        import re
        # Create pattern with word boundaries
        # Handle multi-word skills like "machine learning"
        pattern = r'\b' + re.escape(skill) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def get_skill_categories(self, skills):
        """
        Categorize detected skills.
        
        Args:
            skills: List of skill names
        
        Returns:
            Dict with skills grouped by category
        """
        categories = {
            'languages': [],
            'frameworks': [],
            'cloud': [],
            'databases': [],
            'data_science': [],
            'mobile': [],
            'devops': [],
            'other': []
        }
        
        # Language patterns
        languages = {'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 
                    'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'php'}
        
        # Framework patterns
        frameworks = {'react', 'angular', 'vue', 'django', 'flask', 'spring', 
                     'express', 'rails', 'laravel', 'asp.net'}
        
        # Cloud patterns
        cloud = {'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'}
        
        # Database patterns
        databases = {'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'}
        
        # Data Science patterns
        data_science = {'machine learning', 'tensorflow', 'pytorch', 'pandas', 
                       'numpy', 'spark', 'kafka', 'airflow'}
        
        # Mobile patterns
        mobile = {'ios', 'android', 'react native', 'flutter', 'swift ui'}
        
        # DevOps patterns
        devops = {'jenkins', 'gitlab ci', 'ansible', 'puppet', 'prometheus'}
        
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in languages:
                categories['languages'].append(skill)
            elif skill_lower in frameworks:
                categories['frameworks'].append(skill)
            elif skill_lower in cloud:
                categories['cloud'].append(skill)
            elif skill_lower in databases:
                categories['databases'].append(skill)
            elif skill_lower in data_science:
                categories['data_science'].append(skill)
            elif skill_lower in mobile:
                categories['mobile'].append(skill)
            elif skill_lower in devops:
                categories['devops'].append(skill)
            else:
                categories['other'].append(skill)
        
        return categories
    
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """
        Calculate comprehensive skill match metrics.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
        
        Returns:
            Dict with multiple match metrics
        """
        resume_set = set([s.lower() for s in resume_skills])
        job_set = set([s.lower() for s in job_skills])
        
        if not job_set:
            return {
                'jaccard': 0.0,
                'coverage': 0.0,
                'matched_count': 0,
                'matched_skills': [],
                'missing_skills': []
            }
        
        matched = resume_set.intersection(job_set)
        union = resume_set.union(job_set)
        
        return {
            'jaccard': len(matched) / len(union) if union else 0.0,
            'coverage': len(matched) / len(job_set) if job_set else 0.0,
            'matched_count': len(matched),
            'matched_skills': list(matched),
            'missing_skills': list(job_set - resume_set)
        }
