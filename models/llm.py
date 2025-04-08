import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


def load_llm():
    model = genai.GenerativeModel("gemini-1.5-pro")
    api_key = os.getenv("API_KEY")
    genai.configure(api_key=api_key)
    return model
