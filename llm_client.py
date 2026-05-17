import os
from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = (
        "You are an expert NASA mission specialist with deep knowledge of space exploration history. "
        "You have access to official NASA mission transcripts, technical documents, and archives covering "
        "Apollo 11, Apollo 13, and the Challenger missions. "
        "When answering questions, always cite the specific source documents provided in the context. "
        "Base your answers strictly on the provided context. "
        "If the context does not contain enough information to answer confidently, say so clearly "
        "rather than speculating or making ungrounded claims."
    )

    # Set context in messages
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following retrieved documents as context for your answer:\n\n{context}"
        })

    # Add chat history
    for turn in conversation_history:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    client = OpenAI(
        api_key=openai_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    )

    # Send request to OpenAI
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Return response
    return completion.choices[0].message.content