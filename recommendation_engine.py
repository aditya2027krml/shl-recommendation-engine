import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from utils import DataValidator, RecommendationValidator, SHLBusinessRules

# Set up logging
logger = logging.getLogger(__name__)

class AdvancedRecommendationEngine:
    """
    Advanced Recommendation Engine for SHL Assessments
    Supports multiple recommendation strategies and SHL-specific business logic
    """
    
    def __init__(self, assessments_df: pd.DataFrame, jobs_df: pd.DataFrame):
        """Initialize with data validation and embedding generation"""
        # Validate data
        assessments_valid, assessment_errors = DataValidator.validate_assessments_data(assessments_df)
        jobs_valid, job_errors = DataValidator.validate_jobs_data(jobs_df)
        
        if not assessments_valid:
            logger.error(f"Assessment data validation failed: {assessment_errors}")
            raise ValueError(f"Invalid assessments data: {assessment_errors}")
        if not jobs_valid:
            logger.error(f"Jobs data validation failed: {job_errors}")
            raise ValueError(f"Invalid jobs data: {job_errors}")
        
        self.assessments = assessments_df.copy()
        self.jobs = jobs_df.copy()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create combined text for embeddings
        logger.info("Preparing text data for embeddings...")
        self.assessments["combined_text"] = (
            self.assessments["title"].astype(str) + " " +
            self.assessments["description"].astype(str) + " " +
            self.assessments["skills_measured"].astype(str) + " " +
            self.assessments["domain"].astype(str)
        )
        
        self.jobs["combined_text"] = (
            self.jobs["role"].astype(str) + " " +
            self.jobs["description"].astype(str) + " " +
            self.jobs["required_skills"].astype(str) + " " +
            self.jobs["domain"].astype(str)
        )
        
        # Generate embeddings
        logger.info("Generating embeddings for assessments and jobs...")
        self.assessment_embeddings = self.model.encode(
            self.assessments["combined_text"].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True
        )
        self.job_embeddings = self.model.encode(
            self.jobs["combined_text"].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logger.info(f"Embeddings generated: assessments={self.assessment_embeddings.shape}, jobs={self.job_embeddings.shape}")
    
    def semantic_similarity(self, job_role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic similarity-based recommendations using Sentence-BERT embeddings
        
        Args:
            job_role: Target job role for recommendations
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Input validation
            is_valid, message = RecommendationValidator.validate_job_role(job_role, self.jobs["role"].tolist())
            if not is_valid:
                logger.error(f"Invalid job role: {message}")
                return []
            
            is_valid, message = RecommendationValidator.validate_top_k(top_k)
            if not is_valid:
                logger.error(f"Invalid top_k: {message}")
                return []
            
            job_idx = self.jobs[self.jobs["role"] == job_role].index[0]
            sim_scores = cosine_similarity([self.job_embeddings[job_idx]], self.assessment_embeddings)[0]
            
            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            recommendations = self._format_recommendations(top_indices, sim_scores, job_role, "semantic")
            
            # Apply SHL business rules
            recommendations = self._apply_shl_business_rules(job_role, recommendations)
            
            return RecommendationValidator.sanitize_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error in semantic similarity for {job_role}: {e}")
            return []
    
    def skill_based_matching(self, job_role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Skill overlap-based recommendations using Jaccard similarity
        
        Args:
            job_role: Target job role for recommendations
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Input validation
            is_valid, message = RecommendationValidator.validate_job_role(job_role, self.jobs["role"].tolist())
            if not is_valid:
                logger.error(f"Invalid job role: {message}")
                return []
            
            is_valid, message = RecommendationValidator.validate_top_k(top_k)
            if not is_valid:
                logger.error(f"Invalid top_k: {message}")
                return []
            
            job_row = self.jobs[self.jobs["role"] == job_role].iloc[0]
            job_skills = set(DataValidator.clean_skills_string(job_row["required_skills"]))
            
            scores = []
            for idx, assessment in self.assessments.iterrows():
                assessment_skills = set(DataValidator.clean_skills_string(assessment["skills_measured"]))
                
                # Calculate Jaccard similarity
                if job_skills and assessment_skills:
                    overlap = len(job_skills.intersection(assessment_skills))
                    union = len(job_skills.union(assessment_skills))
                    skill_score = overlap / union if union > 0 else 0
                else:
                    skill_score = 0
                
                # Domain bonus for better matching
                job_domain = str(job_row["domain"]).lower()
                assessment_domain = str(assessment["domain"]).lower()
                domain_bonus = 0.2 if job_domain in assessment_domain or assessment_domain in job_domain else 0
                
                # Experience level compatibility
                exp_compatibility = SHLBusinessRules.validate_difficulty_match(
                    assessment["difficulty"], job_row["experience_level"]
                )
                
                final_score = (skill_score * 0.6) + (domain_bonus * 0.2) + (exp_compatibility * 0.2)
                scores.append((idx, final_score))
            
            # Get top_k recommendations
            top_indices = [idx for idx, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]]
            top_scores = [score for idx, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]]
            
            recommendations = self._format_recommendations(top_indices, top_scores, job_role, "skill_based")
            
            # Apply SHL business rules
            recommendations = self._apply_shl_business_rules(job_role, recommendations)
            
            return RecommendationValidator.sanitize_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error in skill-based matching for {job_role}: {e}")
            return []
    
    def role_specific_filtering(self, job_role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rule-based filtering using ideal_role, domain, and experience level matching
        
        Args:
            job_role: Target job role for recommendations
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Input validation
            is_valid, message = RecommendationValidator.validate_job_role(job_role, self.jobs["role"].tolist())
            if not is_valid:
                logger.error(f"Invalid job role: {message}")
                return []
            
            is_valid, message = RecommendationValidator.validate_top_k(top_k)
            if not is_valid:
                logger.error(f"Invalid top_k: {message}")
                return []
            
            job_row = self.jobs[self.jobs["role"] == job_role].iloc[0]
            job_domain = str(job_row["domain"]).lower()
            exp_level = str(job_row["experience_level"]).lower()
            
            filtered_assessments = self.assessments.copy()
            
            # Filter by ideal_role match (highest priority)
            role_mask = filtered_assessments["ideal_role"].apply(
                lambda x: any(job_role.lower() in role.lower() for role in str(x).split(", ")) 
                if pd.notna(x) else False
            )
            
            # Filter by domain relevance (secondary priority)
            domain_mask = filtered_assessments["domain"].apply(
                lambda x: job_domain in str(x).lower() or str(x).lower() in job_domain 
                if pd.notna(x) else False
            )
            
            # Combine filters - prioritize role match, then domain match
            if role_mask.any():
                filtered_assessments = filtered_assessments[role_mask]
                logger.debug(f"Filtered by role match: {len(filtered_assessments)} assessments")
            elif domain_mask.any():
                filtered_assessments = filtered_assessments[domain_mask]
                logger.debug(f"Filtered by domain match: {len(filtered_assessments)} assessments")
            else:
                logger.debug("No role or domain matches found, using all assessments")
            
            # Score by experience level matching and other factors
            def calculate_role_score(assessment: pd.Series) -> float:
                base_score = 0.5  # Base score for passing filters
                
                # Experience level compatibility
                exp_score = SHLBusinessRules.validate_difficulty_match(assessment["difficulty"], exp_level)
                
                # Duration preference (shorter assessments preferred)
                duration_score = 1.0 - (min(assessment["duration_min"], 120) / 120) * 0.2
                
                # Skills relevance (even if not in ideal_role)
                job_skills = set(DataValidator.clean_skills_string(job_row["required_skills"]))
                assessment_skills = set(DataValidator.clean_skills_string(assessment["skills_measured"]))
                if job_skills and assessment_skills:
                    skill_overlap = len(job_skills.intersection(assessment_skills))
                    skill_score = min(skill_overlap / len(job_skills), 0.3)
                else:
                    skill_score = 0
                
                return base_score + (exp_score * 0.3) + (duration_score * 0.1) + skill_score
            
            filtered_assessments["role_score"] = filtered_assessments.apply(calculate_role_score, axis=1)
            
            # Sort and get top_k
            top_assessments = filtered_assessments.nlargest(top_k, "role_score")
            recommendations = self._format_recommendations(
                top_assessments.index.tolist(), 
                top_assessments["role_score"].tolist(), 
                job_role, 
                "role_based"
            )
            
            # Apply SHL business rules
            recommendations = self._apply_shl_business_rules(job_role, recommendations)
            
            return RecommendationValidator.sanitize_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error in role-specific filtering for {job_role}: {e}")
            return []
    
    def hybrid_recommendation(self, job_role: str, top_k: int = 5, 
                            weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> List[Dict[str, Any]]:
        """
        Hybrid recommendation combining all three methods with weighted scoring
        
        Args:
            job_role: Target job role for recommendations
            top_k: Number of top recommendations to return
            weights: Weights for (semantic, skill, role) methods
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Input validation
            is_valid, message = RecommendationValidator.validate_job_role(job_role, self.jobs["role"].tolist())
            if not is_valid:
                logger.error(f"Invalid job role: {message}")
                return []
            
            is_valid, message = RecommendationValidator.validate_top_k(top_k)
            if not is_valid:
                logger.error(f"Invalid top_k: {message}")
                return []
            
            # Validate weights
            if abs(sum(weights) - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0: {weights}. Normalizing...")
                total = sum(weights)
                weights = tuple(w/total for w in weights)
            
            # Get recommendations from all methods (get more than needed for diversity)
            semantic_recs = self.semantic_similarity(job_role, top_k * 3)
            skill_recs = self.skill_based_matching(job_role, top_k * 3)
            role_recs = self.role_specific_filtering(job_role, top_k * 3)
            
            # Combine and weight scores
            all_recs = {}
            
            for rec in semantic_recs:
                key = rec["Assessment"]
                all_recs[key] = all_recs.get(key, 0) + rec["Score"] * weights[0]
            
            for rec in skill_recs:
                key = rec["Assessment"]
                all_recs[key] = all_recs.get(key, 0) + rec["Score"] * weights[1]
            
            for rec in role_recs:
                key = rec["Assessment"]
                all_recs[key] = all_recs.get(key, 0) + rec["Score"] * weights[2]
            
            # Get top_k overall
            top_assessments = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Format results
            hybrid_recs = []
            for assessment, score in top_assessments:
                assessment_row = self.assessments[self.assessments["title"] == assessment].iloc[0]
                
                # Determine which methods contributed to this recommendation
                contributing_methods = []
                if any(assessment == rec["Assessment"] for rec in semantic_recs[:top_k]):
                    contributing_methods.append("semantic")
                if any(assessment == rec["Assessment"] for rec in skill_recs[:top_k]):
                    contributing_methods.append("skill")
                if any(assessment == rec["Assessment"] for rec in role_recs[:top_k]):
                    contributing_methods.append("role")
                
                hybrid_recs.append({
                    "Assessment": assessment,
                    "Domain": assessment_row["domain"],
                    "Difficulty": assessment_row["difficulty"],
                    "Skills": assessment_row["skills_measured"],
                    "Duration": assessment_row["duration_min"],
                    "Score": round(score, 3),
                    "Method": "hybrid",
                    "Contributing_Methods": ", ".join(contributing_methods),
                    "Explanation": f"Combined recommendation using {', '.join(contributing_methods)} matching. " +
                                  f"Final score: {score:.3f} from weighted combination."
                })
            
            # Apply SHL business rules
            hybrid_recs = self._apply_shl_business_rules(job_role, hybrid_recs)
            
            return RecommendationValidator.sanitize_recommendations(hybrid_recs)
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendation for {job_role}: {e}")
            return []
    
    def get_all_recommendation_methods(self, job_role: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommendations from all methods for comparison
        
        Args:
            job_role: Target job role for recommendations
            top_k: Number of top recommendations to return per method
            
        Returns:
            Dictionary with recommendations from all methods
        """
        return {
            "semantic": self.semantic_similarity(job_role, top_k),
            "skill_based": self.skill_based_matching(job_role, top_k),
            "role_based": self.role_specific_filtering(job_role, top_k),
            "hybrid": self.hybrid_recommendation(job_role, top_k)
        }
    
    def _format_recommendations(self, indices: List[int], scores: List[float], 
                              job_role: str, method: str) -> List[Dict[str, Any]]:
        """
        Format recommendations into standardized dictionary format
        
        Args:
            indices: List of assessment indices
            scores: List of corresponding scores
            job_role: Target job role
            method: Recommendation method used
            
        Returns:
            List of formatted recommendation dictionaries
        """
        recommendations = []
        for idx, score in zip(indices, scores):
            assessment = self.assessments.iloc[idx]
            recommendations.append({
                "Assessment": assessment["title"],
                "Domain": assessment["domain"],
                "Difficulty": assessment["difficulty"],
                "Skills": assessment["skills_measured"],
                "Duration": assessment["duration_min"],
                "Score": round(score, 3),
                "Method": method,
                "Explanation": self._generate_explanation(assessment, job_role, method, score)
            })
        return recommendations
    
    def _generate_explanation(self, assessment: pd.Series, job_role: str, 
                            method: str, score: float) -> str:
        """
        Generate human-readable explanation for recommendations
        
        Args:
            assessment: Assessment data
            job_role: Target job role
            method: Recommendation method used
            score: Recommendation score
            
        Returns:
            Explanation string
        """
        base_explanations = {
            "semantic": f"Content analysis shows strong alignment with {job_role} role requirements.",
            "skill_based": f"Skill overlap found in {assessment['skills_measured']} relevant to {job_role}.",
            "role_based": f"Specifically designed for {job_role}-type roles with appropriate difficulty level.",
            "hybrid": f"Combined analysis shows comprehensive relevance to {job_role} position.",
            "shl_business_rule": f"Included as essential assessment for comprehensive {job_role} evaluation."
        }
        
        explanation = base_explanations.get(method, "Recommended based on multiple relevance factors.")
        
        # Add score context
        if score > 0.8:
            explanation += " Exceptional match with role requirements."
        elif score > 0.6:
            explanation += " Strong relevance for this position."
        elif score > 0.4:
            explanation += " Good match for skill development."
        else:
            explanation += " Moderate relevance for supplementary evaluation."
        
        # Add domain-specific context
        if "cognitive" in assessment["domain"].lower():
            explanation += " Cognitive assessments provide foundational reasoning evaluation."
        elif "technical" in assessment["domain"].lower():
            explanation += " Technical skills assessment crucial for role performance."
        
        return explanation
    
    def _apply_shl_business_rules(self, job_role: str, recommendations: List[Dict]) -> List[Dict]:
        """
        Apply SHL-specific business rules to recommendations
        
        Args:
            job_role: Target job role
            recommendations: Current recommendations
            
        Returns:
            Updated recommendations with business rules applied
        """
        try:
            updated_recommendations = recommendations.copy()
            
            # Rule 1: Ensure at least one cognitive test for all roles
            if SHLBusinessRules.should_include_cognitive_test(job_role, recommendations):
                cognitive_assessments = SHLBusinessRules.get_cognitive_assessments(self.assessments)
                if not cognitive_assessments.empty:
                    # Find the most relevant cognitive assessment
                    best_cognitive = cognitive_assessments.iloc[0]
                    cognitive_rec = {
                        "Assessment": best_cognitive["title"],
                        "Domain": best_cognitive["domain"],
                        "Difficulty": best_cognitive["difficulty"],
                        "Skills": best_cognitive["skills_measured"],
                        "Duration": best_cognitive["duration_min"],
                        "Score": 0.5,  # Base score for mandatory cognitive test
                        "Method": "shl_business_rule",
                        "Explanation": "Cognitive assessment included as per SHL best practices for comprehensive evaluation across all roles."
                    }
                    updated_recommendations.append(cognitive_rec)
                    logger.info(f"Added cognitive assessment for {job_role} as per business rules")
            
            # Rule 2: Limit maximum number of recommendations to 10
            if len(updated_recommendations) > 10:
                updated_recommendations = sorted(updated_recommendations, key=lambda x: x["Score"], reverse=True)[:10]
                logger.debug(f"Limited recommendations to top 10 for {job_role}")
            
            # Rule 3: Ensure diversity of domains in top recommendations
            domain_counts = {}
            diverse_recommendations = []
            
            for rec in sorted(updated_recommendations, key=lambda x: x["Score"], reverse=True):
                domain = rec["Domain"]
                if domain_counts.get(domain, 0) < 2:  # Max 2 per domain in top recommendations
                    diverse_recommendations.append(rec)
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Re-sort by score after ensuring diversity
            diverse_recommendations.sort(key=lambda x: x["Score"], reverse=True)
            
            return diverse_recommendations
            
        except Exception as e:
            logger.warning(f"Error applying SHL business rules for {job_role}: {e}")
            return recommendations
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the recommendation engine
        
        Returns:
            Dictionary with engine statistics
        """
        return {
            "total_assessments": len(self.assessments),
            "total_jobs": len(self.jobs),
            "assessment_domains": self.assessments["domain"].nunique(),
            "job_domains": self.jobs["domain"].nunique(),
            "embedding_dimensions": self.assessment_embeddings.shape[1],
            "difficulty_distribution": self.assessments["difficulty"].value_counts().to_dict(),
            "experience_level_distribution": self.jobs["experience_level"].value_counts().to_dict()
        }


# Utility function for easy instantiation
def create_recommendation_engine(assessments_path: str = "assessment_catalogue.csv",
                               jobs_path: str = "job_descriptions.csv") -> AdvancedRecommendationEngine:
    """
    Helper function to create recommendation engine from file paths
    
    Args:
        assessments_path: Path to assessments CSV file
        jobs_path: Path to jobs CSV file
        
    Returns:
        Initialized AdvancedRecommendationEngine instance
    """
    try:
        assessments = pd.read_csv(assessments_path)
        jobs = pd.read_csv(jobs_path)
        
        # Clean column names
        assessments.columns = assessments.columns.str.strip()
        jobs.columns = jobs.columns.str.strip()
        
        engine = AdvancedRecommendationEngine(assessments, jobs)
        logger.info("Recommendation engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create recommendation engine: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Example of how to use the engine
    try:
        engine = create_recommendation_engine()
        
        # Get stats
        stats = engine.get_engine_stats()
        print("Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test recommendations
        test_job = "AI Research Intern"
        print(f"\nTesting recommendations for: {test_job}")
        
        # Get hybrid recommendations
        recommendations = engine.hybrid_recommendation(test_job, top_k=3)
        
        print(f"Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['Assessment']} (Score: {rec['Score']})")
            print(f"   Explanation: {rec['Explanation']}")
            print()
            
    except Exception as e:
        print(f"Error in example usage: {e}")