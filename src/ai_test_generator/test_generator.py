
"""

src/ai_test_generator/test_generator.py
 
AI-Powered Test Case Generator
================================
Reads plain English requirements and uses an LLM to automatically
generate comprehensive pytest test cases.
 
This demonstrates AI-Enabled Quality Engineering — the AI acts as
a senior test engineer, identifying test scenarios a human might miss.
"""
 
import sys
import os
import json
import re
from pathlib import Path
from loguru import logger
 
# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
 
from src.groq_client import GroqClient
 
 
class AITestGenerator:
    """
    Uses Groq LLM to generate pytest test cases from requirements.
 
    Workflow:
    1. Load requirements from text file
    2. Send to LLM with system prompt defining test engineering context
    3. LLM returns structured test cases (JSON)
    4. Convert to executable Python pytest code
    5. Save to file ready to run
    """
 
    def __init__(self):
        self.client = GroqClient()
        self.system_prompt = """
You are a Senior SDET with 14 years of experience in financial services.
Your speciality is writing comprehensive pytest test cases that cover:
- Happy path scenarios
- Boundary value analysis
- Negative test cases
- Edge cases
- Business rule validation
 
When given requirements, you identify ALL possible test scenarios
including ones developers typically forget to test.
 
You output ONLY valid JSON — no explanation, no markdown.
"""
 
    def load_requirements(self, file_path: str) -> str:
        """Load requirements from a text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Requirements file not found: {path}")
        content = path.read_text(encoding="utf-8")
        logger.info(f"Loaded requirements: {len(content)} chars from {path.name}")
        return content
 
    def generate_test_scenarios(self, requirements: str) -> list:
        """
        Ask the LLM to identify all test scenarios from requirements.
        Returns a list of test scenario objects.
        """
        prompt = f"""
You are a Senior SDET. Analyse these requirements and generate test scenarios.
 
IMPORTANT: Return a JSON ARRAY with AT LEAST 15 test scenarios.
Each object in the array must have exactly these fields:
- scenario_id: string like "TS-001"
- requirement_id: string like "REQ-001"
- scenario_name: short descriptive name
- test_type: one of happy_path, boundary, negative, edge_case
- description: what this test validates
- input_data: example values as a string
- expected_result: what should happen
- expected_error_code: error code string or null
 
Generate scenarios for EVERY requirement below.
Cover happy path, boundaries, nulls, invalid values for each.
 
Requirements:
{requirements}
 
Return ONLY a valid JSON array starting with [ and ending with ]
No explanation. No markdown. Just the JSON array.
"""
        logger.info("Generating test scenarios from requirements...")
        scenarios = self.client.ask_list(prompt, self.system_prompt)
        logger.info(f"Generated {len(scenarios)} test scenarios")
        return scenarios
 
    def generate_pytest_code(
        self, scenarios: list, module_name: str = "payment"
    ) -> str:
        """
        Convert test scenarios into executable pytest code.
        """
        scenarios_json = json.dumps(scenarios, indent=2)
 
        prompt = f"""
Convert these test scenarios into a complete, executable Python pytest file.
 
Requirements:
- Use pytest framework
- Import uuid and datetime at the top
- Each scenario becomes one test function
- Function names follow pattern: test_<scenario_id>_<short_name>
- Use descriptive docstrings explaining what each test validates
- Use assert statements with clear failure messages
- Group tests in classes by requirement ID like TestREQ001, TestREQ002
- Include a mock_payment_service pytest fixture that accepts these params:
  amount=None, currency=None, customer_id=None,
  idempotency_key=None, settlement_date=None, transaction_date=None
- The fixture validates inputs and returns dict with result and error_code
- Add parametrize where multiple similar scenarios exist
- Include ALL scenarios — do not skip any
 
VALID_CURRENCIES = ["USD", "GBP", "EUR", "JPY", "CHF", "AUD", "CAD", "HKD"]
 
Test scenarios:
{scenarios_json}
 
Return ONLY the complete Python code starting with imports.
No explanation. No markdown backticks. Just the Python code.
"""
        logger.info("Generating pytest code from scenarios...")
        code = self.client.ask(prompt, self.system_prompt)
 
        # Clean up any accidental markdown
        code = re.sub(r"```python|```", "", code).strip()
 
        # Remove any leading non-Python lines
        lines = code.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from') or line.startswith('#') or line.startswith('"""'):
                start_idx = i
                break
        code = '\n'.join(lines[start_idx:])
 
        return code
 
    def generate_and_save(
        self,
        requirements_file: str,
        output_file: str = None
    ) -> dict:
        """
        Full pipeline: requirements file → pytest code file.
        Returns summary of what was generated.
        """
        # Load requirements
        requirements = self.load_requirements(requirements_file)
 
        # Generate scenarios
        scenarios = self.generate_test_scenarios(requirements)
 
        # Generate pytest code
        code = self.generate_pytest_code(scenarios)
 
        # Save to file
        if output_file is None:
            req_name = Path(requirements_file).stem
            output_file = f"tests/ai_validation/test_{req_name}_ai_generated.py"
 
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code, encoding="utf-8")
 
        logger.success(
            f"Generated {len(scenarios)} test scenarios "
            f"→ saved to {output_file}"
        )
 
        return {
            "requirements_file": requirements_file,
            "output_file":       output_file,
            "scenarios_count":   len(scenarios),
            "scenarios":         scenarios,
            "code_lines":        len(code.splitlines()),
            "status":            "success"
        }