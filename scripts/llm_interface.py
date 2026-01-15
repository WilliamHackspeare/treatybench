"""
Unified LLM Interface

Provides a consistent API for calling Claude (Anthropic) and GPT (OpenAI) models.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
import openai


# Model configurations
MODELS = {
    # Claude models
    "claude-opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "max_tokens": 8000
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "max_tokens": 8000
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-20250514",
        "max_tokens": 4000
    },
    # OpenAI models
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "max_tokens": 8000
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 8000
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "max_tokens": 4000
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-turbo",
        "max_tokens": 4000
    }
}


# Initialize clients
_anthropic_client: Optional[anthropic.Anthropic] = None
_openai_client: Optional[openai.OpenAI] = None


def _get_anthropic_client() -> anthropic.Anthropic:
    """Get or create Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_openai_client() -> openai.OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


def call_llm(
    model_key: str,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7
) -> dict:
    """
    Call an LLM with a unified interface.

    Args:
        model_key: Model identifier (e.g., "claude-sonnet", "gpt-4o")
        messages: List of message dicts with "role" and "content"
        system_prompt: Optional system prompt
        temperature: Sampling temperature

    Returns:
        Dict with "content" (response text) and "usage" (token counts)
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    config = MODELS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]
    max_tokens = config["max_tokens"]

    if provider == "anthropic":
        return _call_anthropic(model_id, messages, system_prompt, max_tokens, temperature)
    elif provider == "openai":
        return _call_openai(model_id, messages, system_prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(
    model_id: str,
    messages: list[dict],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float
) -> dict:
    """Call Anthropic API."""
    client = _get_anthropic_client()

    # Convert messages to Anthropic format
    # Anthropic expects messages as list of {role, content} but handles system separately
    anthropic_messages = []
    for msg in messages:
        anthropic_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "temperature": temperature
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    return {
        "content": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        },
        "model": model_id,
        "provider": "anthropic"
    }


def _call_openai(
    model_id: str,
    messages: list[dict],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float
) -> dict:
    """Call OpenAI API."""
    client = _get_openai_client()

    # Build messages list with optional system prompt
    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # GPT-5.x models use max_completion_tokens instead of max_tokens
    if model_id.startswith("gpt-5"):
        response = client.chat.completions.create(
            model=model_id,
            messages=openai_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_id,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

    return {
        "content": response.choices[0].message.content,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        },
        "model": model_id,
        "provider": "openai"
    }


def get_available_models() -> list[str]:
    """Get list of available model keys."""
    return list(MODELS.keys())


def get_model_info(model_key: str) -> dict:
    """Get information about a specific model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    return MODELS[model_key].copy()


if __name__ == "__main__":
    # Test both providers
    print("Testing LLM Interface...\n")

    test_messages = [{"role": "user", "content": "Say 'Hello from [your provider]!' in exactly those words."}]

    print("=== Testing Claude ===")
    try:
        result = call_llm("claude-sonnet", test_messages)
        print(f"Response: {result['content']}")
        print(f"Tokens: {result['usage']}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Testing OpenAI ===")
    try:
        result = call_llm("gpt-4o-mini", test_messages)
        print(f"Response: {result['content']}")
        print(f"Tokens: {result['usage']}")
    except Exception as e:
        print(f"Error: {e}")
