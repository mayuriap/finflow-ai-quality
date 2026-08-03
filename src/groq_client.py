
"""
src/groq_client.py
Wrapper around the Groq API.
All AI calls in this project go through this class.
"""
import json
import re
from groq import Groq
from loguru import logger
from src.config import config

class GroqClient:

    def __init__(self):
        config.validate()
        self.client     = Groq(api_key=config.GROQ_API_KEY)
        self.model      = config.MODEL_NAME
        self.max_tokens = config.MAX_TOKENS
        logger.info(f"GroqClient initialised — model: {self.model}")

    def ask(self, prompt: str, system: str = None) -> str:
        """
        Send a prompt to Groq LLM and return text response.
        Core method used by all AI testing modules.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Sending prompt ({len(prompt)} chars)")
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            max_tokens  = self.max_tokens,
            temperature = 0.3,
        )
        result = response.choices[0].message.content
        logger.debug(f"Response received ({len(result)} chars)")
        return result

    def ask_structured(
        self, prompt: str, system: str = None
    ) -> dict:
        """
        Ask Groq and parse JSON response.
        Used when we need structured output for test generation.
        """
        json_system = (
            (system or "") +
            "\n\nRespond ONLY with valid JSON. "
            "No explanation, no markdown backticks, no preamble. "
            "Start your response with { and end with }"
        )
        response = self.ask(prompt, system=json_system)
        cleaned  = re.sub(r"```json|```", "", response).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw: {cleaned}")
            raise

    def ask_list(
        self, prompt: str, system: str = None
    ) -> list:
        """
        Ask Groq and parse JSON array response.
        Used when we need a list of test cases or checks.
        """
        result = self.ask_structured(prompt, system)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return [result]