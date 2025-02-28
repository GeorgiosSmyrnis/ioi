import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_openrouter_usage_and_credits():
    # Get API key from environment variables
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # API endpoints
    usage_url = "https://openrouter.ai/api/v1/auth/key"
    credits_url = "https://openrouter.ai/api/v1/credits"

    # Headers required for OpenRouter API
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        # Fetch usage data
        usage_response = requests.get(usage_url, headers=headers)
        usage_response.raise_for_status()  # Raise an exception for bad status codes
        usage_data = usage_response.json()

        # Fetch credits data
        credits_response = requests.get(credits_url, headers=headers)
        print(credits_response.json())
        credits_response.raise_for_status()  # Raise an exception for bad status codes
        credits_data = credits_response.json()

        return usage_data, credits_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None, None

if __name__ == "__main__":
    usage_data, credits_data = get_openrouter_usage_and_credits()
    if usage_data:
        print("OpenRouter Usage Information:")
        data = usage_data.get('data', {})
        print(f"Label: {data.get('label', 'N/A')}")
        print(f"Limit: {data.get('limit', 'N/A')}")
        print(f"Usage: {data.get('usage', 'N/A')}")
        print(f"Limit Remaining: {data.get('limit_remaining', 'N/A')}")
        print(f"Is Free Tier: {data.get('is_free_tier', 'N/A')}")
        rate_limit = data.get('rate_limit', {})
        print(f"Rate Limit Requests: {rate_limit.get('requests', 'N/A')}")
        print(f"Rate Limit Interval: {rate_limit.get('interval', 'N/A')}")

    if credits_data:
        print("OpenRouter Credits Information:")
        data = credits_data.get('data', {})
        print(f"Total Credits: {data.get('total_credits', 'N/A')}")
        print(f"Total Usage: {data.get('total_usage', 'N/A')}")
        # Print any other relevant information from the response