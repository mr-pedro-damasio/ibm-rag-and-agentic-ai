import logging
from dataclasses import dataclass
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import config

logger = logging.getLogger(__name__)


@dataclass
class LLMSetup:
    llm: OpenAILike
    embed_model: OpenAIEmbedding


def configure(model_name: str = None) -> LLMSetup:
    """Build LLM and embedding model objects and return them as an LLMSetup.

    Also writes to the LlamaIndex Settings singleton as a safety net for any
    internal LlamaIndex components that bypass explicit arguments.

    Args:
        model_name: Optional model override. If None, uses config.QUERY_MODEL.
                    Must be a fully namespaced OpenRouter model name,
                    e.g. "ibm/granite-3-2-8b-instruct".
    """
    llm_model = model_name or config.QUERY_MODEL

    logger.info("Configuring LlamaIndex Settings...")
    logger.info(f"  LLM model        : {llm_model}")
    logger.info(f"  Embedding model  : {config.EMBEDDING_MODEL}")
    logger.info(f"  Embedding dims   : {config.EMBEDDING_MODEL_DIMENSIONS}")
    logger.info(f"  Chunk size       : {config.CHUNK_SIZE}")

    llm = OpenAILike(
        model=llm_model,
        api_key=config.OPENROUTER_API_KEY,
        api_base=config.OPENROUTER_BASE_URL,
        is_chat_model=True,
        temperature=config.QUERY_MODEL_TEMPERATURE,
        max_tokens=config.QUERY_MODEL_MAX_TOKENS,
    )

    # OpenRouter uses the OpenAI embedding protocol.
    # The provider prefix must still be stripped here because OpenAIEmbedding
    # passes the model name directly to the OpenAI-compatible /embeddings endpoint,
    # which does not accept the "openai/" namespace prefix.
    bare_embedding_model = config.EMBEDDING_MODEL.split("/", 1)[-1]
    embed_model = OpenAIEmbedding(
        model=bare_embedding_model,
        api_key=config.OPENROUTER_API_KEY,
        api_base=config.OPENROUTER_BASE_URL,
        dimensions=config.EMBEDDING_MODEL_DIMENSIONS,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE

    logger.info("LlamaIndex Settings configured successfully.")
    return LLMSetup(llm=llm, embed_model=embed_model)
