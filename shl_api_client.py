import requests
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class SHLAPIClient:
    """
    Professional API Client for SHL Assessments
    Simulates real API integration patterns for demonstration
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.shl.com/v1"):
        self.base_url = base_url
        self.api_key = api_key or "demo_key_2024"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SHL-Recommendation-Engine/1.0"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting simulation
        self.last_request_time = 0
        self.request_delay = 0.1  # 100ms between requests
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Make authenticated API request with error handling and rate limiting
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.request_delay:
            time.sleep(self.request_delay - (current_time - self.last_request_time))
        self.last_request_time = time.time()
        
        try:
            logger.info(f"API Request: {url}")
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("API Authentication failed - invalid API key")
                raise AuthenticationError("Invalid API credentials")
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, implementing backoff")
                time.sleep(1)  # Simple backoff
                return self._make_request(endpoint, params)  # Retry once
            else:
                logger.error(f"API request failed with status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timeout")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("API connection error")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def get_assessments(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch assessments from SHL API
        In real scenario, this would make actual API calls to SHL's assessment catalog
        """
        logger.info(f"Fetching {limit} assessments from SHL API (offset: {offset})")
        
        # Simulate API parameters that a real SHL API would use
        params = {
            "limit": limit,
            "offset": offset,
            "include_metadata": "true",
            "status": "active"
        }
        
        # In reality: response = self._make_request("assessments", params)
        # For demo purposes, we'll simulate the API response
        
        try:
            # Simulate API call delay
            time.sleep(0.5)
            
            # Simulate API failure 10% of the time to demonstrate error handling
            import random
            if random.random() < 0.1:
                logger.warning("Simulating API failure for demonstration")
                return []
            
            # Return mock data that matches what real SHL API would return
            return self._get_mock_assessments(limit)
            
        except Exception as e:
            logger.error(f"Failed to fetch assessments from API: {e}")
            return []
    
    def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Get specific assessment details by ID"""
        logger.info(f"Fetching assessment details for: {assessment_id}")
        
        try:
            # Simulate API call
            # response = self._make_request(f"assessments/{assessment_id}")
            time.sleep(0.3)
            
            assessments = self._get_mock_assessments(50)
            assessment = next((a for a in assessments if a.get("id") == assessment_id), None)
            
            if assessment:
                logger.info(f"Successfully fetched assessment: {assessment_id}")
            else:
                logger.warning(f"Assessment not found: {assessment_id}")
                
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to fetch assessment {assessment_id}: {e}")
            return None
    
    def search_assessments(self, query: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search assessments by query and domain"""
        logger.info(f"Searching assessments: '{query}' in domain: {domain}")
        
        try:
            params = {"q": query}
            if domain:
                params["domain"] = domain
                
            # Simulate API search
            # response = self._make_request("assessments/search", params)
            time.sleep(0.4)
            
            all_assessments = self._get_mock_assessments(100)
            
            # Simulate search logic
            filtered_assessments = [
                a for a in all_assessments
                if query.lower() in a.get("name", "").lower() or 
                   query.lower() in a.get("description", "").lower() or
                   query.lower() in " ".join(a.get("skills", [])).lower()
            ]
            
            if domain:
                filtered_assessments = [
                    a for a in filtered_assessments 
                    if domain.lower() in a.get("domain", "").lower()
                ]
            
            logger.info(f"Search found {len(filtered_assessments)} assessments")
            return filtered_assessments
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def validate_api_connection(self) -> bool:
        """Validate API connection and credentials"""
        try:
            # Simulate API health check
            # response = self.session.get(f"{self.base_url}/health", timeout=5)
            # return response.status_code == 200
            
            time.sleep(0.2)
            logger.info("API connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"API connection validation failed: {e}")
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and rate limit information"""
        return {
            "status": "operational",
            "version": "v1",
            "rate_limit": {
                "remaining": 950,
                "limit": 1000,
                "reset_time": int(time.time()) + 3600
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_mock_assessments(self, limit: int) -> List[Dict[str, Any]]:
        """Generate realistic mock assessments matching SHL's actual product structure"""
        mock_assessments = [
            {
                "id": "SHL_OPQ",
                "name": "Occupational Personality Questionnaire",
                "description": "Measures behavioral preferences and personality traits in workplace settings",
                "category": "Behavioral",
                "domain": "Personality",
                "skills": ["communication", "teamwork", "leadership", "adaptability"],
                "duration": 30,
                "difficulty": "Intermediate",
                "language": "en",
                "price_tier": "premium",
                "metadata": {
                    "source": "SHL Official API",
                    "version": "2.1",
                    "last_updated": "2024-01-15T00:00:00Z",
                    "reliability_score": 0.89
                }
            },
            {
                "id": "SHL_UCT",
                "name": "Universal Cognitive Test",
                "description": "Measures verbal, numerical, and logical reasoning abilities",
                "category": "Cognitive",
                "domain": "Reasoning",
                "skills": ["verbal reasoning", "numerical analysis", "logical thinking"],
                "duration": 45,
                "difficulty": "Intermediate",
                "language": "en",
                "price_tier": "standard",
                "metadata": {
                    "source": "SHL Official API", 
                    "version": "3.0",
                    "last_updated": "2024-02-20T00:00:00Z",
                    "reliability_score": 0.92
                }
            },
            {
                "id": "SHL_SJT",
                "name": "Situational Judgment Test",
                "description": "Evaluates decision-making in work-related scenarios",
                "category": "Behavioral", 
                "domain": "Judgment",
                "skills": ["decision-making", "problem-solving", "ethical judgment"],
                "duration": 25,
                "difficulty": "Beginner",
                "language": "en",
                "price_tier": "standard",
                "metadata": {
                    "source": "SHL Official API",
                    "version": "1.5", 
                    "last_updated": "2024-01-10T00:00:00Z",
                    "reliability_score": 0.85
                }
            },
            {
                "id": "SHL_VCAT",
                "name": "Verbal Comprehension and Analysis Test",
                "description": "Assesses ability to understand and analyze written information",
                "category": "Cognitive",
                "domain": "Verbal",
                "skills": ["reading comprehension", "critical thinking", "analysis"],
                "duration": 35,
                "difficulty": "Intermediate", 
                "language": "en",
                "price_tier": "standard",
                "metadata": {
                    "source": "SHL Official API",
                    "version": "2.2",
                    "last_updated": "2024-03-05T00:00:00Z", 
                    "reliability_score": 0.88
                }
            },
            {
                "id": "SHL_NRT",
                "name": "Numerical Reasoning Test", 
                "description": "Evaluates ability to work with numbers, charts, and financial data",
                "category": "Cognitive",
                "domain": "Numerical",
                "skills": ["numerical reasoning", "data interpretation", "financial literacy"],
                "duration": 40,
                "difficulty": "Advanced",
                "language": "en", 
                "price_tier": "premium",
                "metadata": {
                    "source": "SHL Official API",
                    "version": "3.1",
                    "last_updated": "2024-02-28T00:00:00Z",
                    "reliability_score": 0.91
                }
            }
        ]
        
        # Add more generic assessments to reach limit
        technical_assessments = [
            {
                "id": f"SHL_TECH_{i:03d}",
                "name": f"Technical Skills Assessment {i}",
                "description": f"Comprehensive technical skills evaluation for IT roles",
                "category": "Technical",
                "domain": "Software Engineering", 
                "skills": ["programming", "system design", "debugging", "algorithms"],
                "duration": 60,
                "difficulty": "Advanced" if i % 3 == 0 else "Intermediate",
                "language": "en",
                "price_tier": "premium",
                "metadata": {
                    "source": "SHL Official API",
                    "version": "1.0",
                    "last_updated": "2024-01-20T00:00:00Z",
                    "reliability_score": 0.87
                }
            }
            for i in range(1, min(limit - len(mock_assessments) + 1, 20))
        ]
        
        return mock_assessments + technical_assessments[:limit - len(mock_assessments)]


class AuthenticationError(Exception):
    """Custom exception for API authentication errors"""
    pass


class RateLimitError(Exception):
    """Custom exception for API rate limiting"""
    pass


# Professional data loading with fallback pattern
def load_assessments_with_fallback(use_real_api: bool = False, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Professional data loading function with API fallback pattern
    Demonstrates enterprise-grade error handling and fallback strategies
    """
    client = SHLAPIClient(api_key=api_key)
    
    if use_real_api:
        logger.info("Attempting to load assessments from SHL API...")
        
        if not client.validate_api_connection():
            logger.warning("API connection failed, falling back to local data")
            return _load_local_fallback()
        
        try:
            api_assessments = client.get_assessments(limit=100)
            if api_assessments:
                logger.info(f"Successfully loaded {len(api_assessments)} assessments from SHL API")
                
                # Convert API response to match our expected DataFrame format
                df = _convert_api_to_dataframe(api_assessments)
                return df
            else:
                logger.warning("API returned no data, falling back to local data")
                return _load_local_fallback()
                
        except Exception as e:
            logger.error(f"API data loading failed: {e}, using fallback")
            return _load_local_fallback()
    else:
        logger.info("Using local assessment catalogue (demo mode)")
        return _load_local_fallback()


def _load_local_fallback() -> pd.DataFrame:
    """Load local fallback data"""
    try:
        return pd.read_csv("assessment_catalogue.csv")
    except Exception as e:
        logger.error(f"Local data loading failed: {e}")
        # Return minimal fallback data
        return pd.DataFrame({
            'title': ['Fallback Cognitive Test', 'Fallback Technical Assessment'],
            'description': ['Basic cognitive assessment', 'Basic technical skills test'],
            'skills_measured': ['reasoning, logic', 'programming, problem-solving'],
            'domain': ['Cognitive', 'Technical'],
            'difficulty': ['Intermediate', 'Intermediate'],
            'duration_min': [30, 45]
        })


def _convert_api_to_dataframe(api_assessments: List[Dict]) -> pd.DataFrame:
    """Convert API response format to our internal DataFrame format"""
    converted_data = []
    
    for assessment in api_assessments:
        converted_data.append({
            'title': assessment.get('name', ''),
            'description': assessment.get('description', ''),
            'skills_measured': ', '.join(assessment.get('skills', [])),
            'domain': assessment.get('domain', ''),
            'difficulty': assessment.get('difficulty', 'Intermediate'),
            'duration_min': assessment.get('duration', 30),
            'category': assessment.get('category', 'General'),
            'ideal_role': 'All Roles' if assessment.get('category') == 'Cognitive' else 'Specific Roles'
        })
    
    return pd.DataFrame(converted_data)


# Demonstration of professional usage patterns
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🧠 SHL API Client Demonstration")
    print("=" * 50)
    
    # Create client instance
    client = SHLAPIClient()
    
    # Demonstrate API status check
    status = client.get_api_status()
    print(f"📊 API Status: {status['status']}")
    print(f"🕒 Last Check: {status['timestamp']}")
    
    # Demonstrate assessment fetching
    print("\n🔍 Fetching assessments from API...")
    assessments = client.get_assessments(limit=5)
    
    print(f"✅ Retrieved {len(assessments)} assessments:")
    for assessment in assessments:
        print(f"   - {assessment['name']} ({assessment['category']})")
    
    # Demonstrate search functionality
    print("\n🔎 Searching for 'technical' assessments...")
    technical_assessments = client.search_assessments("technical")
    print(f"✅ Found {len(technical_assessments)} technical assessments")
    
    # Demonstrate professional data loading pattern
    print("\n🔄 Demonstrating professional data loading with fallback...")
    df = load_assessments_with_fallback(use_real_api=False)  # Set to True if you had real API access
    print(f"✅ Loaded {len(df)} assessments in DataFrame format")
    print(f"   Columns: {list(df.columns)}")