# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class RAGTestCase(BaseModel):
    """Represents a single evaluation test case."""
    test_id: str
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    ground_truth: Optional[str] = None

class FaithfulnessVerdict(BaseModel):
    """Judge output for Faithfulness / Groundedness."""
    claims_extracted: List[str] = Field(
        description="Atomic factual claims identified in the generated answer."
    )
    unsupported_claims: List[str] = Field(
        description="Claims that cannot be directly verified from the retrieved contexts."
    )
    reasoning: str = Field(description="Step-by-step reasoning for the score.")
    score: float = Field(
        description="Ratio of supported claims: (total_claims - unsupported_claims) / total_claims. Float between 0.0 and 1.0."
    )

class AnswerRelevanceVerdict(BaseModel):
    """Judge output for Answer Relevance."""
    is_direct_answer: bool = Field(
        description="Whether the answer directly addresses the core question."
    )
    redundant_or_tangential_info: List[str] = Field(
        description="Points in the answer that do not address the prompt."
    )
    reasoning: str = Field(description="Step-by-step assessment of query-to-answer alignment.")
    score: float = Field(
        description="Score between 0.0 (completely irrelevant) and 1.0 (fully addressed)."
    )

class ContextPrecisionVerdict(BaseModel):
    """Judge output for Context Precision."""
    chunk_relevance: List[bool] = Field(
        description="Binary relevance indicator for each retrieved context chunk in order."
    )
    reasoning: str = Field(description="Explanation for why each chunk is or isn't relevant to answering the query.")
    score: float = Field(
        description="Context Precision score calculated as (relevant chunks / total chunks). Float between 0.0 and 1.0."
    )

class MetricResult(BaseModel):
    """Summary result for a single test case across all metrics."""
    test_id: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    overall_score: float