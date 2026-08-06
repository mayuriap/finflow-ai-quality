"""
src/llm_validator/output_validator.py

LLM Output Validator
=====================
Tests AI model responses for:
- Hallucination detection (consistency across multiple runs)
- Grounding (AI stays within provided context)
- Output quality scoring (accuracy, completeness, relevance)
- Response structure validation (correct format returned)
- Prompt effectiveness (system prompt produces intended behaviour)

This demonstrates AI validation as described in the job spec:
"model evaluation, prompt testing, output assessment"
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from collections import Counter
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.groq_client import GroqClient


class LLMOutputValidator:
    """
    Validates LLM output quality across multiple dimensions.
    Each validation method returns a ValidationResult dict.
    """

    def __init__(self):
        self.client = GroqClient()
        logger.info("LLMOutputValidator initialised")

    # ── Hallucination Detection ───────────────────────────────────────────────

    def test_consistency(
        self,
        prompt: str,
        system: str = None,
        runs: int = 5,
        threshold: float = 0.70
    ) -> dict:
        """
        Runs the same prompt N times and measures consistency.
        High variance = potential hallucination risk.

        A consistent AI gives the same core answer each time.
        An inconsistent AI is unreliable — at least one answer is wrong.
        """
        logger.info(f"Running consistency test ({runs} runs)...")
        responses = []

        for i in range(runs):
            response = self.client.ask(prompt, system)
            responses.append(response.strip().lower())
            time.sleep(0.5)  # avoid rate limiting

        # Normalise responses — remove punctuation for comparison
        normalised = [
            re.sub(r'[^\w\s]', '', r).strip()
            for r in responses
        ]

        # Find most common response
        counter         = Counter(normalised)
        most_common     = counter.most_common(1)[0]
        consistency_pct = most_common[1] / runs

        # Calculate average response length variance
        lengths    = [len(r) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = max(lengths) - min(lengths)

        passed = consistency_pct >= threshold

        result = {
            "test":              "consistency",
            "prompt":            prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "runs":              runs,
            "consistency_score": round(consistency_pct, 3),
            "threshold":         threshold,
            "passed":            passed,
            "most_common_answer": most_common[0][:200],
            "avg_response_length": round(avg_length),
            "length_variance":   length_variance,
            "all_responses":     responses,
            "finding": "CONSISTENT" if passed else "INCONSISTENT — hallucination risk"
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(
            f"{status} Consistency: {consistency_pct:.0%} "
            f"(threshold: {threshold:.0%})"
        )
        return result

    def test_hallucination_on_nonexistent(
        self,
        fake_questions: list,
        system: str = None
    ) -> dict:
        """
        Asks about things that don't exist.
        A reliable AI should say it doesn't know.
        A hallucinating AI invents plausible-sounding answers.
        """
        logger.info(f"Running hallucination test on {len(fake_questions)} fake questions...")

        results     = []
        hallucinated = 0

        refusal_phrases = [
            "i don't know", "i do not know", "i'm not sure",
            "i cannot", "i can't", "not aware", "no information",
            "doesn't exist", "does not exist", "not found",
            "unable to find", "no record", "not familiar",
            "i don't have", "i do not have"
        ]

        for question in fake_questions:
            response = self.client.ask(question, system)
            response_lower = response.lower()

            # Check if AI refused or invented an answer
            refused = any(phrase in response_lower for phrase in refusal_phrases)
            hallucinated_answer = not refused

            if hallucinated_answer:
                hallucinated += 1

            results.append({
                "question":   question,
                "response":   response[:300],
                "hallucinated": hallucinated_answer,
                "refused":    refused
            })

        hallucination_rate = hallucinated / len(fake_questions)
        passed = hallucination_rate < 0.20  # allow max 20% hallucination rate

        result = {
            "test":              "hallucination_on_nonexistent",
            "questions_tested":  len(fake_questions),
            "hallucinations":    hallucinated,
            "hallucination_rate": round(hallucination_rate, 3),
            "passed":            passed,
            "details":           results,
            "finding": (
                f"PASS — {hallucination_rate:.0%} hallucination rate"
                if passed else
                f"FAIL — {hallucination_rate:.0%} hallucination rate exceeds 20% threshold"
            )
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} Hallucination: {hallucinated}/{len(fake_questions)} fabricated answers")
        return result

    # ── Grounding Validation ──────────────────────────────────────────────────

    def test_grounding(
        self,
        context: str,
        grounded_questions: list,
        out_of_context_questions: list,
        system: str = None
    ) -> dict:
        """
        Tests that AI answers questions using only provided context.

        Grounded questions: answers ARE in the context — AI should answer correctly
        Out-of-context questions: answers NOT in context — AI should say it doesn't know
        """
        logger.info("Running grounding test...")

        context_system = (
            (system or "") +
            f"\n\nYou must ONLY answer using the following context. "
            f"If the answer is not in the context, say 'This information is not in the provided context.'\n\n"
            f"CONTEXT:\n{context}"
        )

        grounded_results     = []
        out_of_context_results = []

        # Test grounded questions — should answer from context
        for q in grounded_questions:
            response = self.client.ask(q, context_system)
            grounded_results.append({
                "question": q,
                "response": response[:300],
                "answered": len(response) > 50
            })

        # Test out-of-context questions — should refuse
        refusal_phrases = [
            "not in the provided context",
            "not mentioned", "no information",
            "context does not", "i cannot find",
            "not available in", "not provided"
        ]

        for q in out_of_context_questions:
            response = self.client.ask(q, context_system)
            response_lower = response.lower()
            refused = any(p in response_lower for p in refusal_phrases)
            out_of_context_results.append({
                "question": q,
                "response": response[:300],
                "correctly_refused": refused
            })

        grounded_answered   = sum(1 for r in grounded_results if r["answered"])
        context_refused     = sum(1 for r in out_of_context_results if r["correctly_refused"])
        grounding_score     = (
            (grounded_answered + context_refused) /
            (len(grounded_questions) + len(out_of_context_questions))
        )
        passed = grounding_score >= 0.75

        result = {
            "test":                    "grounding",
            "grounding_score":         round(grounding_score, 3),
            "threshold":               0.75,
            "passed":                  passed,
            "grounded_questions":      len(grounded_questions),
            "grounded_answered":       grounded_answered,
            "out_of_context_questions": len(out_of_context_questions),
            "context_refused":         context_refused,
            "grounded_results":        grounded_results,
            "out_of_context_results":  out_of_context_results,
            "finding": (
                f"PASS — grounding score {grounding_score:.0%}"
                if passed else
                f"FAIL — grounding score {grounding_score:.0%} below 75% threshold"
            )
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} Grounding score: {grounding_score:.0%}")
        return result

    # ── Prompt Effectiveness ──────────────────────────────────────────────────

    def test_prompt_effectiveness(
        self,
        system_prompt: str,
        test_cases: list
    ) -> dict:
        """
        Tests that a system prompt produces the intended behaviour.

        Each test case has:
        - input: the user message
        - must_contain: terms that MUST appear in response
        - must_not_contain: terms that must NOT appear in response
        - description: what behaviour this tests
        """
        logger.info(f"Testing prompt effectiveness ({len(test_cases)} cases)...")

        results = []
        passed_count = 0

        for case in test_cases:
            response = self.client.ask(case["input"], system_prompt)
            response_lower = response.lower()

            # Check must_contain
            contain_results = {}
            for term in case.get("must_contain", []):
                contain_results[term] = term.lower() in response_lower

            # Check must_not_contain
            not_contain_results = {}
            for term in case.get("must_not_contain", []):
                not_contain_results[term] = term.lower() not in response_lower

            all_contain     = all(contain_results.values()) if contain_results else True
            all_not_contain = all(not_contain_results.values()) if not_contain_results else True
            case_passed     = all_contain and all_not_contain

            if case_passed:
                passed_count += 1

            results.append({
                "description":      case.get("description", ""),
                "input":            case["input"],
                "response":         response[:300],
                "passed":           case_passed,
                "contain_results":  contain_results,
                "not_contain_results": not_contain_results
            })

        effectiveness_score = passed_count / len(test_cases)
        passed = effectiveness_score >= 0.80

        result = {
            "test":                "prompt_effectiveness",
            "test_cases":          len(test_cases),
            "passed_cases":        passed_count,
            "effectiveness_score": round(effectiveness_score, 3),
            "threshold":           0.80,
            "passed":              passed,
            "details":             results,
            "finding": (
                f"PASS — {effectiveness_score:.0%} effectiveness"
                if passed else
                f"FAIL — {effectiveness_score:.0%} effectiveness below 80% threshold"
            )
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} Prompt effectiveness: {effectiveness_score:.0%}")
        return result

    # ── Output Quality Scoring ────────────────────────────────────────────────

    def score_output_quality(
        self,
        prompt: str,
        response: str,
        expected_elements: list,
        forbidden_elements: list = None
    ) -> dict:
        """
        Scores an AI response on multiple quality dimensions:
        - Completeness: does it cover all expected elements?
        - Relevance: does it stay on topic?
        - Clarity: is it well structured?
        - Safety: no forbidden elements present?
        """
        response_lower = response.lower()

        # Completeness — how many expected elements are present?
        completeness_hits = sum(
            1 for e in expected_elements
            if e.lower() in response_lower
        )
        completeness = completeness_hits / len(expected_elements) if expected_elements else 1.0

        # Safety — no forbidden elements
        forbidden_elements = forbidden_elements or []
        safety_violations = [
            f for f in forbidden_elements
            if f.lower() in response_lower
        ]
        safety_score = 1.0 if not safety_violations else 0.0

        # Clarity — basic structural indicators
        has_structure = any([
            len(response.split('\n')) > 2,      # multi-line
            '. ' in response,                    # proper sentences
            len(response.split()) > 20           # substantive length
        ])
        clarity_score = 0.8 if has_structure else 0.4

        # Overall quality score
        overall = (completeness * 0.5) + (safety_score * 0.3) + (clarity_score * 0.2)
        passed = overall >= 0.70

        result = {
            "test":               "output_quality",
            "prompt":             prompt[:100],
            "response_length":    len(response),
            "completeness_score": round(completeness, 3),
            "safety_score":       round(safety_score, 3),
            "clarity_score":      round(clarity_score, 3),
            "overall_score":      round(overall, 3),
            "threshold":          0.70,
            "passed":             passed,
            "safety_violations":  safety_violations,
            "missing_elements": [
                e for e in expected_elements
                if e.lower() not in response_lower
            ],
            "finding": (
                f"PASS — quality score {overall:.0%}"
                if passed else
                f"FAIL — quality score {overall:.0%} below 70% threshold"
            )
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} Output quality: {overall:.0%}")
        return result

    # ── Full Validation Report ────────────────────────────────────────────────

    def generate_validation_report(self, results: list) -> dict:
        """
        Aggregates all validation results into a summary report.
        """
        total  = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        failed = total - passed

        report = {
            "summary": {
                "total_tests":    total,
                "passed":         passed,
                "failed":         failed,
                "pass_rate":      round(passed / total, 3) if total > 0 else 0,
                "overall_status": "PASS" if failed == 0 else "FAIL"
            },
            "results": results
        }

        logger.info(
            f"Validation report: {passed}/{total} passed "
            f"({'PASS' if failed == 0 else 'FAIL'})"
        )
        return report