import logging
from llama_index.core import PromptTemplate, VectorStoreIndex
import config

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Return three interesting facts about the person's career or education.

    Settings.llm and Settings.embed_model must be configured before calling.
    """
    logger.info("Generating initial facts...")
    engine = index.as_query_engine(
        similarity_top_k=config.QUERY_MODEL_TOP_K,
        text_qa_template=PromptTemplate(config.INITIAL_FACTS_TEMPLATE),
    )
    response = engine.query(
        "Provide three interesting facts about this person's career or education."
    )
    logger.info("Initial facts generated.")
    return response.response


def answer_user_query(index: VectorStoreIndex, user_query: str):
    """Answer a free-form question using the profile index.

    Settings.llm and Settings.embed_model must be configured before calling.

    Returns:
        LlamaIndex Response object. Callers access .response for the text.
    """
    logger.info(f"Answering user query: '{user_query}'")
    engine = index.as_query_engine(
        similarity_top_k=config.QUERY_MODEL_TOP_K,
        text_qa_template=PromptTemplate(config.USER_QUESTION_TEMPLATE),
    )
    answer = engine.query(user_query)
    logger.info("Query answered.")
    return answer
