import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and clean datasets"""
    
    @staticmethod
    def validate_assessments_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate assessments dataset structure and content"""
        errors = []
        required_columns = ['title', 'description', 'skills_measured', 'domain', 'difficulty', 'duration_min']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty values in critical columns
        critical_columns = ['title', 'skills_measured', 'domain']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                errors.append(f"Empty values found in critical column: {col}")
        
        # Validate difficulty levels
        valid_difficulties = ['Beginner', 'Intermediate', 'Advanced']
        if 'difficulty' in df.columns:
            invalid_difficulties = df[~df['difficulty'].isin(valid_difficulties)]['difficulty'].unique()
            if len(invalid_difficulties) > 0:
                errors.append(f"Invalid difficulty levels: {invalid_difficulties}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_jobs_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate jobs dataset structure and content"""
        errors = []
        required_columns = ['role', 'description', 'required_skills', 'domain', 'experience_level']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate experience levels
        valid_experience = ['Beginner', 'Intermediate', 'Advanced']
        if 'experience_level' in df.columns:
            invalid_exp = df[~df['experience_level'].isin(valid_experience)]['experience_level'].unique()
            if len(invalid_exp) > 0:
                errors.append(f"Invalid experience levels: {invalid_exp}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_skills_string(skills: str) -> List[str]:
        """Clean and standardize skills string"""
        if pd.isna(skills):
            return []
        skills_clean = re.sub(r'[^\w\s,]', '', str(skills))
        return [skill.strip().lower() for skill in skills_clean.split(',') if skill.strip()]

class RecommendationValidator:
    """Validate recommendation inputs and outputs"""
    
    @staticmethod
    def validate_job_role(job_role: str, available_roles: List[str]) -> Tuple[bool, str]:
        """Validate job role input"""
        if not job_role or not isinstance(job_role, str):
            return False, "Job role must be a non-empty string"
        
        if job_role not in available_roles:
            return False, f"Job role '{job_role}' not found in available roles"
        
        return True, "Valid job role"
    
    @staticmethod
    def validate_top_k(top_k: int, max_k: int = 20) -> Tuple[bool, str]:
        """Validate top_k parameter"""
        if not isinstance(top_k, int):
            return False, "top_k must be an integer"
        
        if top_k <= 0:
            return False, "top_k must be positive"
        
        if top_k > max_k:
            return False, f"top_k cannot exceed {max_k}"
        
        return True, "Valid top_k value"
    
    @staticmethod
    def sanitize_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure recommendation data is clean and properly formatted"""
        sanitized = []
        for i, rec in enumerate(recommendations):
            try:
                sanitized_rec = {
                    'Assessment': str(rec.get('Assessment', f'Unknown_{i}')).strip(),
                    'Domain': str(rec.get('Domain', 'Unknown')).strip(),
                    'Difficulty': str(rec.get('Difficulty', 'Intermediate')).strip(),
                    'Skills': str(rec.get('Skills', '')),
                    'Duration': int(rec.get('Duration', 30)),
                    'Score': max(0.0, min(1.0, float(rec.get('Score', 0.0)))),
                    'Method': str(rec.get('Method', 'unknown')),
                    'Explanation': str(rec.get('Explanation', 'No explanation available'))
                }
                sanitized.append(sanitized_rec)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to sanitize recommendation {i}: {e}")
                continue
                
        return sanitized

class SHLBusinessRules:
    """SHL-specific business logic"""
    
    @staticmethod
    def should_include_cognitive_test(job_role: str, current_recommendations: List[Dict]) -> bool:
        """Determine if cognitive test should be added (for all roles)"""
        cognitive_in_recommendations = any(
            'cognitive' in str(rec.get('Domain', '')).lower() or 
            'cognitive' in str(rec.get('Assessment', '')).lower()
            for rec in current_recommendations
        )
        return not cognitive_in_recommendations
    
    @staticmethod
    def get_cognitive_assessments(assessments_df: pd.DataFrame) -> pd.DataFrame:
        """Get cognitive assessments from catalogue"""
        cognitive_keywords = ['cognitive', 'reasoning', 'logic', 'numerical', 'verbal']
        mask = assessments_df['domain'].str.lower().str.contains('|'.join(cognitive_keywords)) | \
               assessments_df['title'].str.lower().str.contains('|'.join(cognitive_keywords))
        return assessments_df[mask]
    
    @staticmethod
    def validate_difficulty_match(assessment_difficulty: str, job_experience: str) -> float:
        """Calculate compatibility score between assessment difficulty and job experience level"""
        difficulty_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
        exp_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
        
        diff_score = difficulty_map.get(assessment_difficulty.lower(), 1)
        exp_score = exp_map.get(job_experience.lower(), 1)
        
        # Perfect match = 1.0, off by one level = 0.7, off by two levels = 0.3
        difference = abs(diff_score - exp_score)
        return 1.0 - (difference * 0.3)