import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    print("Please ensure you have GEMINI_API_KEY='YOUR_API_KEY' in your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Attempting to list models available with your API key...")
    try:
        found_gemini_pro = False
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"Model: {m.name}, Supported methods: {m.supported_generation_methods}")
                if m.name == "models/gemini-pro":
                    found_gemini_pro = True
        if found_gemini_pro:
            print("\n'models/gemini-pro' found and supports generateContent. Your API key should be valid for this model.")
        else:
            print("\n'models/gemini-pro' was NOT found among supported models. This indicates an issue with your API key's permissions or regional availability for this model.")
    except Exception as e:
        print(f"Error listing models (likely API key issue): {e}")