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

def get_sample_examples():
    """Return sample resume and job description examples"""
    examples = {
        # ============ GOOD FIT EXAMPLES (High Match) ============
        "✅ Data Scientist (Excellent Match)": {
            "resume": """JOHN DOE
Data Scientist | Machine Learning Engineer

SUMMARY
Results-driven Data Scientist with 5+ years of experience in machine learning, deep learning, and statistical analysis. Expert in Python, TensorFlow, and scikit-learn. Proven track record of building predictive models that drive business decisions.

TECHNICAL SKILLS
• Programming: Python, R, SQL, Java
• Machine Learning: TensorFlow, PyTorch, Keras, scikit-learn, XGBoost
• Data Analysis: Pandas, NumPy, SciPy, Matplotlib, Seaborn
• Big Data: Apache Spark, Hadoop, Hive
• Cloud: AWS (SageMaker, EC2, S3), Azure ML
• Databases: PostgreSQL, MongoDB, Redis
• Tools: Jupyter, Git, Docker, Kubernetes

EXPERIENCE
Senior Data Scientist | Tech Corp | 2021 - Present
• Developed recommendation system using collaborative filtering, increasing user engagement by 35%
• Built NLP models for sentiment analysis with 92% accuracy using BERT transformers
• Deployed machine learning pipelines on AWS SageMaker for real-time predictions
• Led team of 3 data scientists in customer churn prediction project

Data Scientist | Analytics Inc | 2019 - 2021
• Created time-series forecasting models using LSTM and Prophet for sales prediction
• Implemented A/B testing framework that improved conversion rates by 15%
• Performed feature engineering and model optimization reducing prediction latency by 40%

EDUCATION
M.S. in Computer Science - Machine Learning Track | Stanford University | 2019
B.S. in Mathematics | UC Berkeley | 2017

CERTIFICATIONS
• AWS Certified Machine Learning Specialty
• TensorFlow Developer Certificate""",
            "job": """SENIOR DATA SCIENTIST

Company: AI Innovations Inc.
Location: San Francisco, CA
Type: Full-time

ABOUT THE ROLE
We are seeking an experienced Data Scientist to join our AI team. You will build and deploy machine learning models that power our products used by millions of users.

REQUIREMENTS
• 4+ years of experience in data science or machine learning
• Strong programming skills in Python and SQL
• Expertise in machine learning frameworks (TensorFlow, PyTorch, or scikit-learn)
• Experience with deep learning and neural networks
• Proficiency in data manipulation using Pandas and NumPy
• Experience deploying models to production
• Knowledge of cloud platforms (AWS, Azure, or GCP)
• Strong statistical analysis skills

PREFERRED QUALIFICATIONS
• Master's degree in Computer Science, Statistics, or related field
• Experience with NLP and transformers (BERT, GPT)
• Familiarity with MLOps and model monitoring
• Experience with big data tools (Spark, Hadoop)
• Knowledge of Docker and Kubernetes

RESPONSIBILITIES
• Design and implement machine learning models for various use cases
• Collaborate with engineering teams to deploy models to production
• Perform exploratory data analysis and feature engineering
• Monitor model performance and retrain as needed
• Communicate findings to stakeholders

WHAT WE OFFER
• Competitive salary ($150K - $200K)
• Stock options
• Health, dental, and vision insurance
• 401(k) matching
• Remote work options"""
        },
        
        "✅ Full-Stack Developer (Great Match)": {
            "resume": """JANE SMITH
Full-Stack Software Engineer

CONTACT
Email: jane.smith@email.com | Phone: (555) 123-4567
LinkedIn: linkedin.com/in/janesmith | GitHub: github.com/janesmith

SUMMARY
Passionate Full-Stack Engineer with 4 years of experience building scalable web applications. Proficient in React, Node.js, and cloud technologies. Strong focus on clean code and user experience.

TECHNICAL SKILLS
• Frontend: React, TypeScript, JavaScript, HTML5, CSS3, Redux, Next.js
• Backend: Node.js, Express, Python, Django, REST APIs, GraphQL
• Databases: PostgreSQL, MongoDB, MySQL, Redis
• Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, CI/CD, Jenkins
• Tools: Git, Jira, Agile/Scrum, Jest, Mocha

PROFESSIONAL EXPERIENCE
Software Engineer | WebTech Solutions | 2021 - Present
• Developed responsive React applications serving 100K+ daily active users
• Built RESTful APIs using Node.js and Express with 99.9% uptime
• Implemented microservices architecture reducing deployment time by 50%
• Optimized database queries improving page load times by 30%
• Collaborated with UX team to enhance user interface and accessibility

Junior Developer | StartupXYZ | 2020 - 2021
• Created full-stack features using React and Node.js
• Integrated third-party APIs (Stripe, SendGrid, Twilio)
• Wrote unit tests achieving 85% code coverage
• Participated in code reviews and agile sprint planning

EDUCATION
B.S. in Computer Science | MIT | 2020

PROJECTS
E-commerce Platform: Built full-stack marketplace using MERN stack with payment integration
Real-time Chat App: Developed WebSocket-based chat application using Socket.io and Redis""",
            "job": """FULL-STACK SOFTWARE ENGINEER

Company: Tech Startup Inc.
Location: Remote (US)
Salary: $120K - $160K

JOB DESCRIPTION
We're looking for a talented Full-Stack Engineer to help build our next-generation SaaS platform. You'll work on both frontend and backend, shipping features that delight our customers.

REQUIRED SKILLS
• 3+ years of professional software development experience
• Strong proficiency in JavaScript/TypeScript and React
• Backend experience with Node.js or Python
• Experience with RESTful API design and development
• Solid understanding of databases (SQL and NoSQL)
• Experience with Git and version control
• Knowledge of responsive web design

NICE TO HAVE
• Experience with Next.js or other React frameworks
• Familiarity with GraphQL
• AWS or cloud platform experience
• Docker and containerization knowledge
• CI/CD pipeline setup experience
• TypeScript expertise

RESPONSIBILITIES
• Build and maintain web applications using React and Node.js
• Design and implement RESTful APIs
• Write clean, maintainable, and well-tested code
• Collaborate with product and design teams
• Participate in code reviews and technical discussions
• Optimize application performance and scalability

BENEFITS
• Competitive salary and equity
• Flexible remote work
• Health and wellness benefits
• Professional development budget
• Unlimited PTO"""
        },
        
        "✅ DevOps Engineer (Strong Match)": {
            "resume": """MICHAEL CHEN
DevOps Engineer | Cloud Infrastructure Specialist

SUMMARY
DevOps Engineer with 6 years of experience in cloud infrastructure, CI/CD automation, and container orchestration. Expert in AWS, Kubernetes, and infrastructure-as-code. Passionate about building reliable, scalable systems.

TECHNICAL SKILLS
• Cloud Platforms: AWS (EC2, ECS, EKS, Lambda, S3, RDS, CloudFormation), Azure, GCP
• Container & Orchestration: Docker, Kubernetes, Helm, Docker Swarm
• CI/CD: Jenkins, GitLab CI, GitHub Actions, CircleCI, ArgoCD
• Infrastructure as Code: Terraform, Ansible, CloudFormation, Pulumi
• Scripting: Python, Bash, PowerShell
• Monitoring: Prometheus, Grafana, ELK Stack, Datadog, CloudWatch
• Version Control: Git, GitHub, GitLab
• Databases: PostgreSQL, MySQL, MongoDB, Redis

EXPERIENCE
Senior DevOps Engineer | CloudTech Inc. | 2020 - Present
• Designed and implemented Kubernetes-based microservices platform serving 1M+ users
• Reduced deployment time by 70% using GitOps and ArgoCD
• Built CI/CD pipelines processing 500+ deployments per week
• Implemented infrastructure as code with Terraform managing 200+ AWS resources
• Set up monitoring and alerting reducing MTTR by 60%

DevOps Engineer | StartupCo | 2018 - 2020
• Migrated legacy infrastructure to AWS reducing costs by 40%
• Automated deployment processes using Jenkins and Docker
• Implemented backup and disaster recovery procedures
• Managed production systems with 99.95% uptime

EDUCATION
B.S. in Computer Engineering | UC Berkeley | 2018

CERTIFICATIONS
• AWS Certified Solutions Architect - Professional
• Certified Kubernetes Administrator (CKA)
• HashiCorp Certified: Terraform Associate""",
            "job": """SENIOR DEVOPS ENGINEER

Company: FinTech Solutions
Location: New York, NY (Hybrid)
Salary: $140K - $180K

ROLE OVERVIEW
Join our platform team to build and maintain cloud infrastructure that powers our financial services platform. You'll work on cutting-edge DevOps practices and tools.

REQUIRED SKILLS
• 5+ years of DevOps/SRE experience
• Expert-level knowledge of AWS services
• Strong experience with Kubernetes and container orchestration
• Proficiency in Infrastructure as Code (Terraform, CloudFormation)
• Experience building and maintaining CI/CD pipelines
• Strong scripting skills (Python, Bash)
• Experience with monitoring and logging tools

PREFERRED SKILLS
• AWS certifications (Solutions Architect, DevOps Engineer)
• Kubernetes certification (CKA/CKAD)
• Experience with GitOps (ArgoCD, Flux)
• Knowledge of service mesh (Istio, Linkerd)
• Experience in financial services or regulated industries
• Familiarity with security best practices

RESPONSIBILITIES
• Design and maintain cloud infrastructure on AWS
• Build and optimize CI/CD pipelines
• Manage Kubernetes clusters and deployments
• Implement infrastructure as code using Terraform
• Monitor system performance and reliability
• Automate operational tasks
• Participate in on-call rotation

BENEFITS
• Competitive salary and bonus
• Stock options
• Premium health insurance
• 401(k) with matching"""
        },
        
        # ============ NOT A GOOD FIT EXAMPLES (Moderate Mismatch) ============
        "⚠️ Junior Dev for Senior Role": {
            "resume": """SARAH WILLIAMS
Junior Software Developer

ABOUT ME
Recent computer science graduate with 1 year of internship experience. Eager to learn and grow in software development. Basic knowledge of web technologies.

SKILLS
• Programming: Python, Java
• Web: HTML, CSS, JavaScript basics
• Tools: Git, Visual Studio Code
• Databases: MySQL (learning)

EXPERIENCE
Software Development Intern | Local Startup | Summer 2024
• Fixed bugs in existing codebase
• Wrote simple Python scripts for data processing
• Attended daily standup meetings
• Learned about agile development

Teaching Assistant | University | 2023 - 2024
• Helped students with Java programming assignments
• Graded homework and exams

EDUCATION
B.S. in Computer Science | State University | 2024
GPA: 3.5/4.0

PROJECTS
Personal Website: Created portfolio website using HTML, CSS, and JavaScript
Todo App: Built basic task manager using React (learning project)""",
            "job": """SENIOR SOFTWARE ARCHITECT

Company: Enterprise Corp
Location: Seattle, WA
Salary: $180K - $220K

POSITION
We need a seasoned software architect to lead our enterprise platform modernization. This is a senior technical leadership role requiring extensive experience.

REQUIREMENTS
• 10+ years of software engineering experience
• 5+ years in architectural roles
• Expert in microservices architecture and design patterns
• Deep knowledge of cloud platforms (AWS/Azure)
• Experience with distributed systems and scalability
• Strong background in system design and trade-offs
• Leadership experience managing technical teams
• Excellent communication with C-level executives

TECHNICAL REQUIREMENTS
• Advanced proficiency in Java, C#, or similar
• Experience with Spring Boot, .NET Core
• Knowledge of event-driven architectures (Kafka, RabbitMQ)
• Database design expertise (SQL and NoSQL)
• API design and governance
• Security best practices and compliance

RESPONSIBILITIES
• Define technical architecture for enterprise applications
• Lead architectural review boards
• Mentor senior engineers
• Make critical technology decisions
• Create technical roadmaps
• Present to executive leadership"""
        },
        
        "⚠️ Wrong Tech Stack": {
            "resume": """ROBERT MARTINEZ
Mobile App Developer

PROFILE
iOS developer with 4 years building native mobile applications. Specialized in Swift and iOS ecosystem. Published 8 apps on the App Store with 500K+ downloads.

TECHNICAL SKILLS
• Languages: Swift, Objective-C, some Java
• iOS: UIKit, SwiftUI, Core Data, CoreAnimation
• Tools: Xcode, Instruments, TestFlight
• Backend: Firebase, basic REST API integration
• Version Control: Git, GitHub
• Design: Figma (basic)

EXPERIENCE
iOS Developer | Mobile Apps Inc. | 2021 - Present
• Developed 5 consumer iOS apps from scratch
• Implemented in-app purchases and subscriptions
• Integrated push notifications using Firebase
• Optimized app performance reducing load times by 40%
• Worked with designers to implement UI/UX

Mobile Developer | AppStudio | 2020 - 2021
• Built iOS features for e-commerce app
• Fixed bugs and improved app stability
• Collaborated with backend team on API integration

EDUCATION
B.S. in Information Technology | 2020

PORTFOLIO
Fitness Tracker App - 100K+ downloads
Recipe Sharing App - 50K+ users
Meditation App - Featured on App Store""",
            "job": """SENIOR BACKEND ENGINEER - JAVA/KOTLIN

Company: Enterprise Solutions
Location: Boston, MA
Salary: $150K - $190K

DESCRIPTION
We're seeking a backend engineer to work on our high-performance microservices platform. This role focuses on server-side development, databases, and system architecture.

MUST HAVE
• 5+ years backend development experience
• Expert in Java and/or Kotlin
• Strong experience with Spring Boot framework
• Deep knowledge of relational databases (PostgreSQL, Oracle)
• Experience with message queues (Kafka, RabbitMQ)
• RESTful API design and implementation
• Microservices architecture experience
• Understanding of distributed systems

PREFERRED
• Experience with gRPC
• Knowledge of Elasticsearch
• Redis/caching strategies
• Cloud platforms (AWS/GCP)
• Kubernetes deployment
• Performance optimization

RESPONSIBILITIES
• Design and build scalable backend services
• Optimize database queries and performance
• Implement message-driven architectures
• Write comprehensive unit and integration tests
• Participate in code reviews
• Mentor junior developers
• On-call support rotation

TECH STACK
Java 17, Spring Boot, PostgreSQL, Kafka, Redis, Kubernetes, AWS"""
        },
        
        "⚠️ Skill Gap - Different Domain": {
            "resume": """EMILY BROWN
Graphic Designer | UI/UX Designer

BIO
Creative designer with 5 years creating beautiful user interfaces and brand identities. Passionate about visual design, typography, and user experience.

SKILLS
• Design Tools: Figma, Adobe XD, Sketch, Photoshop, Illustrator
• Prototyping: InVision, Marvel, Principle
• UI/UX: Wireframing, User Research, Usability Testing
• Web: Basic HTML/CSS (reading level)
• Soft Skills: Communication, Collaboration, Presentation

WORK EXPERIENCE
Senior UI/UX Designer | Design Agency | 2022 - Present
• Created user interfaces for 15+ client projects
• Conducted user research and usability testing
• Designed design systems and component libraries
• Collaborated with developers on implementation
• Presented designs to stakeholders

UI Designer | Tech Startup | 2020 - 2022
• Designed mobile and web app interfaces
• Created wireframes and prototypes
• Worked closely with product managers
• Maintained brand consistency

EDUCATION
B.A. in Graphic Design | Art Institute | 2020

PORTFOLIO
www.emilybrown-designs.com
Behance: behance.net/emilyb""",
            "job": """FRONTEND ENGINEER - REACT/TYPESCRIPT

Company: SaaS Platform Inc.
Location: Austin, TX
Salary: $130K - $160K

ROLE
We need a frontend engineer to build complex web applications using modern JavaScript frameworks. You'll write production code, not just design mockups.

REQUIRED
• 4+ years professional frontend development
• Expert in React and TypeScript
• Strong JavaScript fundamentals (ES6+)
• Experience with state management (Redux, MobX)
• HTML5, CSS3, SASS/LESS
• RESTful API integration
• Git version control
• Unit testing (Jest, React Testing Library)
• Build tools (Webpack, Vite)

NICE TO HAVE
• Next.js or other SSR frameworks
• GraphQL and Apollo Client
• CSS-in-JS (styled-components)
• Micro-frontend architectures
• CI/CD experience
• Accessibility standards (WCAG)

RESPONSIBILITIES
• Develop complex React applications
• Write clean, maintainable TypeScript code
• Implement responsive designs from Figma
• Optimize application performance
• Write comprehensive tests
• Collaborate with backend engineers
• Code reviews and mentoring

NOTE: This is an engineering role requiring strong programming skills."""
        },
        
        # ============ WORST FIT EXAMPLES (Major Mismatch) ============
        "❌ Completely Wrong Field": {
            "resume": """DAVID ANDERSON
Marketing Manager | Digital Marketing Specialist

SUMMARY
Results-driven marketing professional with 8 years of experience in digital marketing, brand management, and social media strategy. Proven track record of increasing brand awareness and driving customer engagement.

EXPERTISE
• Digital Marketing Strategy
• Social Media Management (Facebook, Instagram, LinkedIn, Twitter)
• Content Marketing & Copywriting
• SEO/SEM & Google Analytics
• Email Marketing (Mailchimp, HubSpot)
• Marketing Automation
• Brand Development
• Customer Relationship Management (Salesforce)
• Budget Management
• Team Leadership

EXPERIENCE
Marketing Manager | Consumer Brands Inc. | 2020 - Present
• Led digital marketing campaigns increasing online sales by 45%
• Managed $500K annual marketing budget
• Grew social media following from 50K to 200K
• Developed content strategy improving engagement by 60%
• Coordinated with agencies and vendors

Digital Marketing Specialist | Retail Company | 2017 - 2020
• Created and executed email marketing campaigns
• Managed PPC campaigns with $100K budget
• Analyzed marketing metrics and ROI
• Wrote blog posts and website copy

EDUCATION
B.A. in Marketing | Business School | 2017

CERTIFICATIONS
• Google Analytics Certified
• HubSpot Content Marketing Certification
• Facebook Blueprint Certified""",
            "job": """MACHINE LEARNING ENGINEER - COMPUTER VISION

Company: Autonomous Vehicles Inc.
Location: San Francisco, CA
Salary: $160K - $220K

POSITION
We're building self-driving technology and need an ML engineer specializing in computer vision. You'll work on perception systems for autonomous vehicles.

REQUIREMENTS
• MS/PhD in Computer Science, Robotics, or related field
• 5+ years in machine learning and computer vision
• Expert in deep learning frameworks (PyTorch, TensorFlow)
• Strong experience with CNNs, object detection, segmentation
• Experience with SLAM, 3D reconstruction
• Proficiency in Python and C++
• Strong mathematics background (linear algebra, calculus, statistics)
• Publication record in top ML/CV conferences (preferred)

TECHNICAL SKILLS
• Computer Vision: OpenCV, PCL, YOLO, Mask R-CNN
• Deep Learning: PyTorch, TensorFlow, ONNX
• Sensors: LiDAR, cameras, radar processing
• Frameworks: ROS, CUDA programming
• Cloud: AWS SageMaker, GPU clusters
• Tools: Docker, Kubernetes, MLflow

RESPONSIBILITIES
• Develop perception algorithms for autonomous vehicles
• Train and optimize deep learning models
• Process sensor data (cameras, LiDAR, radar)
• Implement real-time object detection and tracking
• Conduct research and experiments
• Deploy models to embedded systems"""
        },
        
        "❌ No Relevant Experience": {
            "resume": """JESSICA TAYLOR
Restaurant Manager | Hospitality Professional

PROFILE
Dedicated restaurant manager with 10 years in the hospitality industry. Expert in customer service, team management, and operations. Passionate about creating exceptional dining experiences.

SKILLS
• Customer Service Excellence
• Team Leadership & Training
• Inventory Management
• Food Safety & Hygiene
• POS Systems (Toast, Square)
• Scheduling & Staffing
• Budget & Cost Control
• Conflict Resolution
• Event Planning
• Microsoft Office (Word, Excel)

EXPERIENCE
Restaurant Manager | Fine Dining Restaurant | 2018 - Present
• Manage daily operations for 100-seat restaurant
• Lead team of 25 staff members
• Oversee $2M annual revenue
• Maintain 4.5-star rating on Yelp and Google
• Handle customer complaints and ensure satisfaction
• Manage inventory and vendor relationships
• Create staff schedules and conduct training

Assistant Manager | Casual Dining Chain | 2014 - 2018
• Supervised front and back of house operations
• Trained new employees on procedures
• Maintained food quality standards
• Processed payroll and managed cash handling

EDUCATION
B.A. in Hospitality Management | 2014

ACHIEVEMENTS
• Manager of the Year 2022
• Increased revenue by 30% over 3 years
• Reduced staff turnover by 40%""",
            "job": """SENIOR DATA ENGINEER - BIG DATA

Company: Data Analytics Corp
Location: Chicago, IL
Salary: $150K - $190K

ABOUT
We need a data engineer to build and maintain our big data infrastructure processing petabytes of data daily.

REQUIREMENTS
• 6+ years in data engineering
• Expert in Apache Spark and Hadoop ecosystem
• Strong programming in Python, Scala, or Java
• Experience with cloud data platforms (AWS, GCP, Azure)
• Proficiency in SQL and database optimization
• Experience with data warehousing (Snowflake, Redshift, BigQuery)
• Knowledge of streaming platforms (Kafka, Kinesis)
• ETL/ELT pipeline development

TECHNICAL STACK
• Big Data: Spark, Hadoop, Hive, Presto
• Databases: PostgreSQL, Cassandra, MongoDB
• Cloud: AWS (EMR, S3, Glue, Athena)
• Streaming: Kafka, Flink
• Orchestration: Airflow, Prefect
• Languages: Python, Scala, SQL
• Infrastructure: Docker, Kubernetes, Terraform

RESPONSIBILITIES
• Design and build scalable data pipelines
• Optimize data processing workflows
• Implement data quality checks
• Manage data warehouse architecture
• Support data scientists and analysts
• Monitor pipeline performance
• Ensure data security and compliance"""
        },
        
        "❌ Entry Level vs Expert Required": {
            "resume": """CHRIS JOHNSON
Recent Graduate | Computer Science Student

SUMMARY
Motivated recent graduate with strong academic background in computer science. Completed coursework in programming fundamentals, data structures, and algorithms. Looking for entry-level position to start my career.

EDUCATION
B.S. in Computer Science | State University | May 2024
GPA: 3.6/4.0
Relevant Coursework: Intro to Programming, Data Structures, Algorithms, Database Systems, Web Development

SKILLS
• Programming: Python (intermediate), Java (basic)
• Web: HTML, CSS, JavaScript (beginner)
• Tools: VS Code, Git (basic)
• Databases: MySQL (classroom projects)
• Microsoft Office Suite

PROJECTS (School)
Student Grade Calculator: Python program to calculate GPAs
Simple Blog Website: HTML/CSS website for class project
Library Management System: Java database project (team of 4)

INTERNSHIP
IT Support Intern | Local Business | Summer 2023 (3 months)
• Helped employees with computer issues
• Set up new workstations
• Created user accounts
• Updated software

ACTIVITIES
• Computer Science Club Member
• Dean's List (3 semesters)
• Volunteered at local coding workshop for kids

INTERESTS
Learning new technologies, gaming, basketball""",
            "job": """PRINCIPAL SOFTWARE ENGINEER - DISTRIBUTED SYSTEMS

Company: Tech Giant Corp
Location: Seattle, WA
Salary: $220K - $300K + equity

ROLE
We're seeking a principal engineer to architect and build our next-generation distributed systems platform. This is a high-impact technical leadership role.

REQUIREMENTS
• 12+ years of software engineering experience
• 5+ years architecting large-scale distributed systems
• Expert in system design, scalability, and performance
• Deep knowledge of distributed computing concepts
• Experience leading technical teams and initiatives
• Track record of shipping major platform projects
• Strong influence on engineering culture and practices
• M.S./Ph.D. in Computer Science preferred

TECHNICAL EXPERTISE
• Languages: Go, Java, C++, or Rust (expert level)
• Distributed Systems: Consensus algorithms, CAP theorem
• Databases: Designing for scale, sharding, replication
• Networking: TCP/IP, HTTP/2, gRPC, load balancing
• Performance: Profiling, optimization, benchmarking
• Cloud: AWS/GCP at massive scale
• Architecture: Microservices, event-driven, CQRS

RESPONSIBILITIES
• Architect systems handling billions of requests/day
• Design for fault tolerance and high availability
• Lead cross-team technical initiatives
• Mentor senior and staff engineers
• Define technical standards and best practices
• Make critical technology decisions
• Present to executive leadership
• Participate in engineering hiring

IMPACT
Your work will affect millions of users globally."""
        }
    }
    return examples

def single_analysis_mode():
    """Single resume-job analysis"""
    
    # Sample Examples Section
    with st.expander("💡 Try Sample Examples", expanded=False):
        st.markdown("**Click a sample to auto-fill resume and job description:**")
        
        examples = get_sample_examples()
        
        # Filter out the "not good fit" examples (those with ⚠️ emoji)
        filtered_examples = {k: v for k, v in examples.items() if not k.startswith('⚠️')}
        
        cols = st.columns(len(filtered_examples))
        
        selected_example = None
        for idx, (role, col) in enumerate(zip(filtered_examples.keys(), cols)):
            with col:
                if st.button(f"📋 {role}", key=f"sample_{idx}", use_container_width=True):
                    selected_example = role
        
        # Store selected example in session state
        if selected_example:
            st.session_state['sample_resume'] = filtered_examples[selected_example]['resume']
            st.session_state['sample_job'] = filtered_examples[selected_example]['job']
            st.session_state['use_sample'] = True
            st.success(f"✅ Loaded {selected_example} example!")
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Resume Upload")
        # Auto-select Paste Text if sample is loaded
        default_resume_method = "Paste Text" if st.session_state.get('use_sample') else "Upload File"
        resume_input_method = st.radio(
            "Input Method", 
            ["Upload File", "Paste Text"], 
            key="resume_method",
            horizontal=True,
            index=1 if st.session_state.get('use_sample') else 0
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
            # Use session state for sample data if available
            default_resume = st.session_state.get('sample_resume', '')
            resume_text = st.text_area(
                "Paste Resume Text", 
                height=300, 
                key="resume_text",
                placeholder="Paste your resume text here...",
                value=default_resume
            )
    
    with col2:
        st.subheader("💼 Job Description Upload")
        # Auto-select Paste Text if sample is loaded
        default_job_method = "Paste Text" if st.session_state.get('use_sample') else "Upload File"
        job_input_method = st.radio(
            "Input Method", 
            ["Upload File", "Paste Text"], 
            key="job_method",
            horizontal=True,
            index=1 if st.session_state.get('use_sample') else 0
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
            # Use session state for sample data if available
            default_job = st.session_state.get('sample_job', '')
            job_text = st.text_area(
                "Paste Job Description Text", 
                height=300, 
                key="job_text",
                placeholder="Paste job description here...",
                value=default_job
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