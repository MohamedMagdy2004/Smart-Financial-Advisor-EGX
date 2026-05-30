#!/usr/bin/env python3
"""
Diagnostic script to verify Groq API configuration and connectivity.
"""
import os
import sys
import logging
from pathlib import Path

# Load .env file first
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "="*60)
print("GROQ API CONFIGURATION DIAGNOSTICS")
print("="*60 + "\n")

# 1. Check environment variables
print("1. CHECKING ENVIRONMENT VARIABLES:")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if groq_api_key:
    print(f"   ✓ GROQ_API_KEY is set (length: {len(groq_api_key)} chars)")
    print(f"   ✓ First 20 chars: {groq_api_key[:20]}...")
    print(f"   ✓ Last 10 chars: ...{groq_api_key[-10:]}")
else:
    print("   ✗ GROQ_API_KEY is NOT set!")
    print("   → Add it to your .env file or set it as an environment variable")

print(f"\n   ✓ GROQ_MODEL: {groq_model}")

# 2. Check .env file
print("\n2. CHECKING .env FILE:")
env_file_paths = [".env", "../.env", "../../.env"]
env_found = False

for env_path in env_file_paths:
    if os.path.exists(env_path):
        print(f"   ✓ Found .env at: {os.path.abspath(env_path)}")
        with open(env_path, "r") as f:
            content = f.read()
            if "GROQ_API_KEY" in content:
                print(f"   ✓ GROQ_API_KEY is defined in {env_path}")
                env_found = True
            else:
                print(f"   ✗ GROQ_API_KEY is NOT defined in {env_path}")

if not env_found and env_file_paths:
    print("   ✗ .env file not found or GROQ_API_KEY not configured")

# 3. Test imports
print("\n3. CHECKING DEPENDENCIES:")
try:
    import requests
    print("   ✓ requests library is available")
except ImportError:
    print("   ✗ requests library NOT found")
    sys.exit(1)

# 4. Test API connectivity
print("\n4. TESTING GROQ API CONNECTIVITY:")
if not groq_api_key:
    print("   ⚠ Skipping API test - GROQ_API_KEY not set")
else:
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": groq_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'API is working' and nothing else."},
                ],
                "temperature": 0.7,
                "max_tokens": 50,
            },
            timeout=10,
        )
        
        if response.status_code == 200:
            print(f"   ✓ API returned status {response.status_code}")
            content = response.json()["choices"][0]["message"]["content"]
            print(f"   ✓ API Response: {content}")
        else:
            print(f"   ✗ API returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ API request failed: {str(e)}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {str(e)}")

print("\n" + "="*60)
print("DIAGNOSTICS COMPLETE")
print("="*60 + "\n")

if groq_api_key:
    print("✓ Configuration looks good! You can start the backend.")
else:
    print("✗ Configuration issue detected.")
    print("   Please set GROQ_API_KEY in your .env file or environment.")
