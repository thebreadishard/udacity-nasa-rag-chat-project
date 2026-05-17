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
    if not question or not answer or not contexts:
        return {"error": "Invalid input: question, answer, and contexts are required"}
    
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


def batch_evaluate(dataset_path: str, collection=None, openai_key: str = None) -> Dict:
    """
    Load test questions from a JSON file and evaluate end-to-end.

    Dataset file format (test_questions.json):
      List of objects with "question" and optionally "answer" + "contexts":
        [
          {"question": "...", "answer": "...", "contexts": ["...", "..."]},
          {"question": "..."}
        ]

    If collection and openai_key are provided, retrieves documents and generates
    answers for any entry missing "answer" or "contexts" (end-to-end mode).
    Otherwise expects "answer" and "contexts" to be pre-supplied in each entry.

    Returns per-question results and aggregate (mean) for each metric.
    """
    import json
    from pathlib import Path

    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"Dataset file not found: {dataset_path}"}

    # Load questions from JSON file
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not entries:
        return {"error": "Dataset file is empty"}

    per_question = []
    metric_totals: Dict[str, List[float]] = {}

    for entry in entries:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        contexts = entry.get("contexts", [])

        # End-to-end mode: retrieve and generate if not pre-supplied
        if (not answer or not contexts) and collection is not None and openai_key:
            import rag_client
            import llm_client
            docs_result = rag_client.retrieve_documents(collection, question, n_results=3)
            if docs_result and docs_result.get("documents"):
                contexts = docs_result["documents"][0]
                context_str = rag_client.format_context(contexts, docs_result["metadatas"][0])
            else:
                contexts = []
                context_str = ""
            answer = llm_client.generate_response(openai_key, question, context_str, [])

        scores = evaluate_response_quality(question, answer, contexts)
        per_question.append({"question": question, "scores": scores})

        # Accumulate for aggregate
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                metric_totals.setdefault(metric, []).append(value)

    # Compute aggregate means
    aggregate = {metric: sum(vals) / len(vals) for metric, vals in metric_totals.items()}

    return {"per_question": per_question, "aggregate": aggregate}
