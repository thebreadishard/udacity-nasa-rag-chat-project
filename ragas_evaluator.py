import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

load_dotenv()

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        )
    )
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        )
    )
    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
        BleuScore(),
        RougeScore(),
        NonLLMContextPrecisionWithReference(),
    ]
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    # Evaluate the response using the metrics
    results = {}
    for metric in metrics:
        try:
            score = metric.single_turn_score(sample)
            results[metric.name] = score
        except Exception:
            results[metric.name] = None
    # Return the evaluation results
    return results
