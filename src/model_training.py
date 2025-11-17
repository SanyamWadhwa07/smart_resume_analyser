"""
Model Training Module for Smart Resume - Job Fit Analyzer
Random Forest classifier with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score)
import joblib
import os

# Import your preprocessing and feature engineering modules
from data_preprocessing import load_resumes, load_job_postings, load_skills_mapping, preprocess_resumes, preprocess_job_postings
from feature_engineering import compute_features

class ResumeJobMatcher:
    """
    Random Forest based Resume-Job Fit Classifier
    """
    def __init__(self, model_dir='models'):
        self.model = None
        self.best_params = None
        self.model_dir = model_dir
        self.training_history = {}
        os.makedirs(model_dir, exist_ok=True)

    def prepare_training_data(self, features_df, target_col='fit'):
        feature_cols = ['skill_match', 'experience_gap', 'education_match', 'industry_match']
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        print(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train_simple(self, X, y, n_estimators=200, max_depth=20, test_size=0.2):
        """
        Train Random Forest classifier
        
        Args:
            X: Feature matrix (can be passed directly without test split)
            y: Labels
            n_estimators: Number of trees
            max_depth: Max tree depth
            test_size: Test set size (only used if splitting needed)
        """
        # If data is already split, don't split again
        if test_size > 0 and len(X) > 10:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except:
                # If stratify fails, try without it
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        if len(X_test) > 0:
            metrics = self._evaluate_model(X_test, y_test)
            self.training_history = {'metrics': metrics}
        
        return self.model

    def _evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        print("\nMODEL PERFORMANCE ON TEST SET:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Not Fit', 'Good Fit']))
        print("\nCONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return metrics

    def predict_job_fit(self, features):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if isinstance(features, dict):
            feature_array = np.array([[
                features.get('skill_match', 0),
                features.get('experience_gap', 0),
                features.get('education_match', 0),
                features.get('industry_match', 0)
            ]])
        else:
            feature_array = np.array(features).reshape(1, -1)
        fit_probability = self.model.predict_proba(feature_array)[0][1]
        fit_label = self.model.predict(feature_array)[0]
        return {
            'fit_probability': float(fit_probability * 100),
            'fit_label': 'Good Fit' if fit_label == 1 else 'Not a Fit',
            'prediction': int(fit_label),
            'confidence': float(max(self.model.predict_proba(feature_array)[0]))
        }

    def save_model(self, filename='random_forest_model.pkl'):
        if self.model is None:
            raise ValueError("No model to save.")
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename='random_forest_model.pkl'):
        filepath = os.path.join(self.model_dir, filename)
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model

# Function to prepare features from real resume/job data
def prepare_real_feature_dataset():
    """Prepare feature dataset from actual resume and job posting data."""
    resumes = load_resumes()
    jobs = load_job_postings()
    
    # No need for skills_map parameter anymore
    resumes = preprocess_resumes(resumes)
    jobs = preprocess_job_postings(jobs)
    
    # Sample a reasonable number of combinations
    # If datasets are large, sample to avoid memory issues
    max_resumes = min(50, len(resumes))
    max_jobs = min(20, len(jobs))
    
    resumes_sample = resumes.sample(n=max_resumes, random_state=42) if len(resumes) > max_resumes else resumes
    jobs_sample = jobs.sample(n=max_jobs, random_state=42) if len(jobs) > max_jobs else jobs
    
    print(f"Creating feature dataset from {len(resumes_sample)} resumes and {len(jobs_sample)} jobs...")
    
    # Pair each resume with each job
    data = []
    for _, r_row in resumes_sample.iterrows():
        for _, j_row in jobs_sample.iterrows():
            features = compute_features(r_row, j_row)
            # For demo, use skill_match as binary target (1 if >0.5 else 0)
            target = 1 if features['skill_match'] > 0.5 else 0
            features['fit'] = target
            data.append(features)
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} training samples")
    return df

# Example usage
if __name__ == "__main__":
    # Use real data for training
    df = prepare_real_feature_dataset()
    print("Sample Training Data:")
    print(df.head())
    print(f"Class distribution: {df['fit'].value_counts().to_dict()}")
    matcher = ResumeJobMatcher()
    X, y = matcher.prepare_training_data(df)
    matcher.train_simple(X, y, n_estimators=200)
    matcher.save_model()
"""
Model Training Module for Smart Resume - Job Fit Analyzer
Random Forest classifier with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score)
import joblib
import os
import seaborn as sns

class ResumeJobMatcher:
    """
    Random Forest based Resume-Job Fit Classifier
    """
    def __init__(self, model_dir='models'):
        """Initialize matcher"""
        self.model = None
        self.best_params = None
        self.model_dir = model_dir
        self.training_history = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_training_data(self, features_df, target_col='fit'):
        """
        Prepare feature matrix and target vector
        
        Args:
            features_df (DataFrame): DataFrame with features and target
            target_col (str): Name of target column
        
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        # Feature columns - check if extended features are available
        base_feature_cols = ['tfidf_similarity', 'doc2vec_similarity', 
                             'skill_jaccard', 'skill_coverage']
        extended_feature_cols = base_feature_cols + ['resume_skill_count', 'job_skill_count']
        
        # Try to use extended features if available, otherwise use base features
        if all(col in features_df.columns for col in extended_feature_cols):
            feature_cols = extended_feature_cols
            print("Using extended feature set (6 features)")
        else:
            feature_cols = base_feature_cols
            print("Using base feature set (4 features)")
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        print(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_with_hyperparameter_tuning(self, X, y, cv=5, n_iter=20, test_size=0.2):
        """
        Train Random Forest with hyperparameter tuning
        
        Args:
            X (array): Feature matrix
            y (array): Target vector
            cv (int): Cross-validation folds
            n_iter (int): Number of parameter settings sampled
            test_size (float): Test set proportion
        
        Returns:
            tuple: (model, metrics) trained model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Define hyperparameter grid
        param_distributions = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'class_weight': ['balanced', None]
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Randomized Search
        print(f"\\nStarting hyperparameter tuning ({n_iter} iterations)...")
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='f1'
        )
        
        # Fit the model
        rf_random.fit(X_train, y_train)
        
        # Save best model and parameters
        self.model = rf_random.best_estimator_
        self.best_params = rf_random.best_params_
        
        print(f"\\n{'='*60}")
        print("BEST HYPERPARAMETERS:")
        print('='*60)
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        # Evaluate on test set
        metrics = self._evaluate_model(X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        print(f"\\nCross-validation F1: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
        
        # Store training history
        self.training_history = {
            'best_params': self.best_params,
            'metrics': metrics,
            'cv_scores': cv_scores
        }
        
        return self.model, metrics
    
    def train_simple(self, X, y, n_estimators=200, max_depth=20, test_size=0.2):
        """
        Train Random Forest with default/simple parameters
        
        Args:
            X (array): Feature matrix
            y (array): Target vector
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            test_size (float): Test set proportion
        
        Returns:
            RandomForestClassifier: Trained model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(X_test, y_test)
        
        self.training_history = {'metrics': metrics}
        
        return self.model
    
    def _evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
        
        Returns:
            dict: Performance metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print results
        print(f"\\n{'='*60}")
        print("MODEL PERFORMANCE ON TEST SET:")
        print('='*60)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\\n{'='*60}")
        print("CLASSIFICATION REPORT:")
        print('='*60)
        print(classification_report(y_test, y_pred, target_names=['Not Fit', 'Good Fit']))
        
        print(f"\\n{'='*60}")
        print("CONFUSION MATRIX:")
        print('='*60)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return metrics
    
    def predict_job_fit(self, features):
        """
        Predict job fit for new resume-job pair
        
        Args:
            features (dict or array): Feature dictionary or array
                Expected keys: tfidf_similarity, doc2vec_similarity, 
                              skill_jaccard, skill_coverage
        
        Returns:
            dict: Prediction results with probability and label
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert features to array
        if isinstance(features, dict):
            # Check how many features the model expects
            n_features_expected = self.model.n_features_in_
            
            # Build feature array based on what the model expects
            if n_features_expected == 6:
                # Model trained with 6 features (including skill counts)
                feature_array = np.array([[
                    features.get('tfidf_similarity', 0),
                    features.get('doc2vec_similarity', 0),
                    features.get('skill_jaccard', 0),
                    features.get('skill_coverage', 0),
                    features.get('resume_skill_count', 0),
                    features.get('job_skill_count', 0)
                ]])
            else:
                # Model trained with 4 features (default)
                feature_array = np.array([[
                    features.get('tfidf_similarity', 0),
                    features.get('doc2vec_similarity', 0),
                    features.get('skill_jaccard', 0),
                    features.get('skill_coverage', 0)
                ]])
        else:
            feature_array = np.array(features).reshape(1, -1)
        
        # Predict probability
        fit_probability = self.model.predict_proba(feature_array)[0][1]
        fit_label = self.model.predict(feature_array)[0]
        
        return {
            'fit_probability': float(fit_probability * 100),  # Convert to percentage
            'fit_label': 'Good Fit' if fit_label == 1 else 'Not a Fit',
            'prediction': int(fit_label),
            'confidence': float(max(self.model.predict_proba(feature_array)[0]))
        }
    
    def predict_batch(self, features_list):
        """
        Predict for multiple resume-job pairs
        
        Args:
            features_list (list): List of feature dictionaries
        
        Returns:
            list: List of prediction results
        """
        results = []
        for features in features_list:
            results.append(self.predict_job_fit(features))
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model
        
        Returns:
            DataFrame: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        feature_names = ['TF-IDF Similarity', 'Doc2Vec Similarity', 
                        'Skill Jaccard', 'Skill Coverage']
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': importances * 100
        }).sort_values('Importance', ascending=False)
        
        print("\\nFEATURE IMPORTANCE:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_model(self, filename='random_forest_model.pkl'):
        """
        Save trained model
        
        Args:
            filename (str): Model filename
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)
        print(f"\\nModel saved to {filepath}")
    
    def load_model(self, filename='random_forest_model.pkl'):
        """
        Load trained model
        
        Args:
            filename (str): Model filename
        
        Returns:
            RandomForestClassifier: Loaded model
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


# Example usage
if __name__ == "__main__":
    # Create synthetic training data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    tfidf_sim = np.random.uniform(0.2, 1.0, n_samples)
    doc2vec_sim = np.random.uniform(0.1, 1.0, n_samples)
    skill_jaccard = np.random.uniform(0.0, 0.9, n_samples)
    skill_coverage = np.random.uniform(0.1, 1.0, n_samples)
    
    # Generate target based on features (more realistic distribution)
    # Good fit if: high TF-IDF AND (good skill coverage OR high doc2vec)
    fit = ((tfidf_sim > 0.6) & ((skill_coverage > 0.5) | (doc2vec_sim > 0.7))).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'tfidf_similarity': tfidf_sim,
        'doc2vec_similarity': doc2vec_sim,
        'skill_jaccard': skill_jaccard,
        'skill_coverage': skill_coverage,
        'fit': fit
    })
    
    print("Sample Training Data:")
    print(df.head())
    print(f"\\nClass distribution: {df['fit'].value_counts().to_dict()}")
    
    # Train model
    matcher = ResumeJobMatcher()
    X, y = matcher.prepare_training_data(df)
    
    # Simple training
    print("\\n" + "="*60)
    print("TRAINING WITH SIMPLE PARAMETERS")
    print("="*60)
    matcher.train_simple(X, y, n_estimators=200)
    
    # Get feature importance
    matcher.get_feature_importance()
    
    # Test prediction
    test_features = {
        'tfidf_similarity': 0.75,
        'doc2vec_similarity': 0.80,
        'skill_jaccard': 0.60,
        'skill_coverage': 0.70
    }
    
    result = matcher.predict_job_fit(test_features)
    print(f"\\nTest Prediction:")
    print(f"Job Fit: {result['fit_probability']:.1f}%")
    print(f"Label: {result['fit_label']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Save model
    matcher.save_model()