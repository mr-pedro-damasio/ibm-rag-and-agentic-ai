import config
from langchain_openrouter import ChatOpenRouter


def llm_model(params: dict | None = None) -> ChatOpenRouter:
    cfg = {
        "model": config.LLM_MODEL,
        "api_key": config.OPENROUTER_API_KEY,
        "base_url": config.OPENROUTER_BASE_URL,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "max_completion_tokens": config.LLM_MAX_COMPLETION_TOKENS,
    }
    if params:
        cfg.update(params)
    return ChatOpenRouter(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_completion_tokens=cfg["max_completion_tokens"],
    )


def llm_response(prompt_text: str, params: dict | None = None) -> str:
    return llm_model(params).invoke(prompt_text).content
