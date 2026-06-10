import config
from langchain_openrouter import ChatOpenRouter
from typing import Optional, Dict
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def llm_model(params: Optional[Dict] = None) -> ChatOpenRouter:
    """Create and configure the LLM model with error handling."""
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
    
    logger.info(f"Creating LLM model: {cfg['model']}")
    logger.info(f"Configuration: temperature={cfg['temperature']}, max_tokens={cfg['max_tokens']}")
    
    try:
        model = ChatOpenRouter(
            model=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            max_completion_tokens=cfg["max_completion_tokens"],
        )
        logger.info("LLM model created successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to create LLM model: {e}")
        logger.error("Check your API key, base URL, and network connection.")
        raise


def llm_response(prompt_text: str, params: Optional[Dict] = None) -> str:
    """Get response from LLM with error handling."""
    try:
        logger.info(f"Getting LLM response for prompt (length: {len(prompt_text)})")
        response = llm_model(params).invoke(prompt_text)
        logger.info(f"LLM response received (length: {len(response.content)})")
        return response.content
    except Exception as e:
        logger.error(f"Failed to get LLM response: {e}")
        raise
