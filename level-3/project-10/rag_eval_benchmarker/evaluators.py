# evaluators.py
import os
from dotenv import load_dotenv
from typing import List
from google import genai
from google.genai import types
from schemas import (
    FaithfulnessVerdict,
    AnswerRelevanceVerdict,
    ContextPrecisionVerdict,
    RAGTestCase,
    MetricResult
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model_used = "gemini-3.1-flash-lite"

class RAGEvaluator:
    """Automated LLM-as-a-Judge RAG evaluation engine."""

    def __init__(self, model_name: str = model_used):
        self.client = genai.Client()
        self.model_name = model_name

    def evaluate_faithfulness(self, test_case: RAGTestCase) -> FaithfulnessVerdict:
        """
        Evaluates whether the answer is strictly grounded in the retrieved context.
        Strategy: Decompose into atomic claims, then verify each against context.
        """
        combined_context = "\n---\n".join(test_case.retrieved_contexts)
        prompt = f"""
        You are an independent, strict fact-checking auditor. Evaluate whether the generated answer is strictly grounded in the provided context.

        [RETRIEVED CONTEXT]
        {combined_context}

        [GENERATED ANSWER]
        {test_case.generated_answer}

        Task:
        1. Extract all atomic factual claims made in the generated answer.
        2. Verify each claim against the retrieved context.
        3. Identify any claim that is unsupported, extrapolated, or contradicts the context.
        4. Calculate the score: (Total Claims - Unsupported Claims) / Total Claims. If there are 0 claims, return 1.0.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FaithfulnessVerdict,
                temperature=0.0
            )
        )
        return FaithfulnessVerdict.model_validate_json(response.text)

    def evaluate_answer_relevance(self, test_case: RAGTestCase) -> AnswerRelevanceVerdict:
        """
        Evaluates whether the answer directly addresses the user query.
        Strategy: Check for directness and penalize evasion, rambling, or off-topic content.
        """
        prompt = f"""
        You are a quality auditor evaluating query-to-answer alignment.

        [USER QUERY]
        {test_case.query}

        [GENERATED ANSWER]
        {test_case.generated_answer}

        Task:
        1. Determine if the answer directly and completely answers what was asked.
        2. Identify any redundant, tangential, or non-responsive information.
        3. Assign an alignment score from 0.0 (completely irrelevant) to 1.0 (precise and direct).
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnswerRelevanceVerdict,
                temperature=0.0
            )
        )
        return AnswerRelevanceVerdict.model_validate_json(response.text)

    def evaluate_context_precision(self, test_case: RAGTestCase) -> ContextPrecisionVerdict:
        """
        Evaluates the signal-to-noise ratio in retrieved context chunks.
        Strategy: Binary classification of each chunk's relevance to the query.
        """
        chunks_formatted = "\n".join([f"Chunk [{i}]: {chunk}" for i, chunk in enumerate(test_case.retrieved_contexts)])
        ground_truth_context = f"\n[GROUND TRUTH REFERENCE]\n{test_case.ground_truth}" if test_case.ground_truth else ""

        prompt = f"""
        You are an information retrieval auditor evaluating retrieval precision.

        [USER QUERY]
        {test_case.query}
        {ground_truth_context}

        [RETRIEVED CHUNKS]
        {chunks_formatted}

        Task:
        1. Evaluate each chunk independently in order.
        2. Mark `true` if the chunk contains information needed to answer the query, `false` if it is noise or irrelevant.
        3. Calculate score: (Number of True Chunks) / (Total Number of Chunks).
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContextPrecisionVerdict,
                temperature=0.0
            )
        )
        return ContextPrecisionVerdict.model_validate_json(response.text)

    def evaluate_case(self, test_case: RAGTestCase) -> MetricResult:
        """Runs all three evaluators on a single test case."""
        faith = self.evaluate_faithfulness(test_case)
        relevance = self.evaluate_answer_relevance(test_case)
        precision = self.evaluate_context_precision(test_case)

        overall = round((faith.score + relevance.score + precision.score) / 3.0, 3)

        return MetricResult(
            test_id=test_case.test_id,
            faithfulness=faith.score,
            answer_relevance=relevance.score,
            context_precision=precision.score,
            overall_score=overall
        )