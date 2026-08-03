"""
src/config.py
Central configuration for FinFlow AI Quality Platform
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME   = os.getenv("MODEL_NAME", "llama3-8b-8192")
    MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "2000"))

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")

    # Paths
    ROOT_DIR     = Path(__file__).parent.parent
    DATA_DIR     = ROOT_DIR / "data"
    REPORTS_DIR  = ROOT_DIR / "reports"
    PROMPTS_DIR  = ROOT_DIR / "data" / "prompts"

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        print(f"✓ Config loaded — model: {cls.MODEL_NAME}")
        return True

config = Config()