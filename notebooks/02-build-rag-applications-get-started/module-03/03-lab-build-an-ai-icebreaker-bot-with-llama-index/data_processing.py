import json
import logging
from typing import Dict, Any, List, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
import config

logger = logging.getLogger(__name__)

# Top-level JSON keys that belong to each semantic section.
_HEADER_KEYS = {
    "first_name", "last_name", "full_name", "headline", "summary",
    "occupation", "country", "city", "state",
}
_SKILL_KEYS = {"skills", "languages"}
_ACTIVITY_KEYS = {
    "groups", "recommendations", "accomplishment_projects",
    "accomplishment_publications", "accomplishment_honors_awards",
    "volunteer_work",
}


def _make_document(content: Any, section: str) -> Document:
    text = json.dumps(content, ensure_ascii=False, indent=2)
    return Document(text=text, metadata={"section": section})


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Split the LinkedIn profile into semantically grouped nodes.

    Creates one Document per semantic section so that chunking boundaries
    align with the data's natural structure rather than arbitrary character
    counts across the raw JSON dump.
    """
    documents = []

    # Header — single-value fields that describe the person at a glance
    header = {k: v for k, v in profile_data.items() if k in _HEADER_KEYS}
    if header:
        documents.append(_make_document(header, "header"))
        logger.debug("Added header document.")

    # Experiences — one Document per job entry for precise retrieval
    for entry in profile_data.get("experiences", []):
        documents.append(_make_document(entry, "experience"))
    logger.debug(f"Added {len(profile_data.get('experiences', []))} experience documents.")

    # Education — one Document per entry
    for entry in profile_data.get("education", []):
        documents.append(_make_document(entry, "education"))
    logger.debug(f"Added {len(profile_data.get('education', []))} education documents.")

    # Skills and languages — grouped together (usually short lists)
    skills_block = {k: profile_data[k] for k in _SKILL_KEYS if k in profile_data}
    if skills_block:
        documents.append(_make_document(skills_block, "skills"))
        logger.debug("Added skills document.")

    # Activity — groups, recommendations, accomplishments
    activity_block = {k: profile_data[k] for k in _ACTIVITY_KEYS if k in profile_data}
    if activity_block:
        documents.append(_make_document(activity_block, "activity"))
        logger.debug("Added activity document.")

    # Anything not covered above — catch-all so no data is silently dropped
    covered = _HEADER_KEYS | {"experiences", "education"} | _SKILL_KEYS | _ACTIVITY_KEYS
    remainder = {k: v for k, v in profile_data.items() if k not in covered}
    if remainder:
        documents.append(_make_document(remainder, "other"))
        logger.debug("Added remainder document.")

    logger.info(f"Created {len(documents)} documents from profile data.")

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Split into {len(nodes)} nodes.")
    return nodes


def create_vector_database(nodes: List) -> VectorStoreIndex:
    """Store nodes in an in-memory vector index.

    Settings.embed_model must be configured (via llm_setup.configure())
    before calling this function.
    """
    logger.info("Building VectorStoreIndex...")
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    logger.info("VectorStoreIndex built successfully.")
    return index


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Return True if every node in the docstore has an entry in the vector store.

    LlamaIndex stores embeddings in the vector store, not on the docstore node
    objects, so we compare node IDs across both stores.
    """
    node_ids = set(index.docstore.docs.keys())
    embedded_ids = set(index._storage_context.vector_store.data.embedding_dict.keys())
    missing = node_ids - embedded_ids
    if missing:
        logger.warning(f"Nodes without embeddings: {missing}")
        return False
    logger.info(f"All {len(node_ids)} nodes are embedded.")
    return True
