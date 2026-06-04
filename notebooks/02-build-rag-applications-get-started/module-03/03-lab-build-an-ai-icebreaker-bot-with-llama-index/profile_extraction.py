import config

import requests
from typing import Dict, Optional, Any

def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
    mock: bool = False
) -> Dict[str, Any]:
    """Extract LinkedIn profile data using API or loads a premade JSON file.

    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.

    Returns:
        Dictionary containing the LinkedIn profile data.
    """
    try:
        if mock:
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            # Ensure API key is provided when mock is False
            if not api_key:
                raise ValueError("API key is required when mock is set to False.")
            
            # Set up the API endpoint and headers
            api_endpoint = config.LINKEDIN_API_ENDPOINT
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # Prepare parameters for the request
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include"
            }

            # Send API request
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)
        
        # Check if response is successful
        if response.status_code == 200:
            try:
                # Parse the JSON response
                data = response.json()
                
                # Clean the data, remove empty values and unwanted fields
                data = {
                    k: v
                    for k, v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }

                # Remove profile picture URLs from groups to clean the data
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)

                return data
            except ValueError as e:
                return {}
        else:
            return {}
            
    except Exception as e:
        return {}
