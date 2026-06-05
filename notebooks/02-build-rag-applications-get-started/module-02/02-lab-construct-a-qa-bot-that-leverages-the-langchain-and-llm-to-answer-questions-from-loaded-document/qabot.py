import config
import logging

import gradio as gr
from pypdf import PdfReader
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Factory kept as a named function so the parameter list is explicit and the
# module-level LLM = llm_model(...) call site reads as configuration, not magic.
def llm_model(model_name: str, model_api_key: str, model_base_url: str, model_temperature: float, model_maxtokens: int) -> ChatOpenRouter:
    model = ChatOpenRouter(
        model=model_name,
        api_key=model_api_key,
        base_url=model_base_url,
        temperature=model_temperature,
        max_tokens=model_maxtokens,
    )
    return model

# RAG_PROMPT and LLM are module-level singletons — both are stateless and
# expensive to initialise (network calls to OpenRouter), so they are built
# once at startup and reused across every question in the session.
RAG_PROMPT = ChatPromptTemplate.from_template(config.RAG_PROMPT_TEMPLATE)

LLM = llm_model(
    model_name=config.LLM_MODEL,
    model_api_key=config.OPENROUTER_API_KEY,
    model_base_url=config.OPENROUTER_BASE_URL,
    model_temperature=config.LLM_TEMPERATURE,
    model_maxtokens=config.LLM_MAX_TOKENS,
)

def document_loader(file_path: str) -> list[Document]:
    logger.info("Loading PDF from: %s", file_path)
    reader = PdfReader(file_path)
    # extract_text() returns None for image-only pages; `or ""` keeps the page
    # in the list so that page-number metadata stays accurate across the whole PDF.
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i}
        )
        for i, page in enumerate(reader.pages)
    ]


def text_splitter(data_to_split: list[Document], chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> list[Document]:
    logger.info("Splitting %d pages into chunks (size=%d, overlap=%d)", len(data_to_split), chunk_size, chunk_overlap)
    # RecursiveCharacterTextSplitter tries to split on paragraph → sentence →
    # word boundaries before falling back to raw characters, so chunks tend to
    # end at natural breaks rather than mid-sentence.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(data_to_split)
    logger.info("Produced %d chunks", len(chunks))
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    # Embeds chunks into an in-memory Chroma store. The store is ephemeral —
    # it lives only for this session and is rebuilt on each new document upload.
    logger.info("Embedding %d chunks (model=%s, dim=%d)", len(chunks), config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSIONS)
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,  # routes the OpenAI SDK's embedding calls through OpenRouter
        dimensions=config.EMBEDDING_DIMENSIONS,
    )
    return Chroma.from_documents(chunks, embeddings)


def format_docs(docs: list[Document]) -> str:
    # Double newline acts as a clear paragraph separator; a single newline risks
    # fusing two unrelated chunks into a run-on passage inside the prompt context.
    return "\n\n".join(doc.page_content for doc in docs)


# Indexing step — runs once per document upload; result cached in gr.State so
# subsequent questions reuse the retriever without re-embedding.
def build_retriever(file: str | None) -> tuple[VectorStoreRetriever | None, str]:
    if file is None:
        return None, "No file uploaded."
    logger.info("Building retriever for: %s", file)
    try:
        splits = document_loader(file)
        chunks = text_splitter(splits)
        logger.info("Split into %d chunks; embedding now...", len(chunks))

        # Scanned/image-only PDFs produce no extractable text — warn the user
        # rather than passing an empty list to Chroma, which raises unhelpfully.
        if not chunks:
            return None, "No text could be extracted from this PDF. Is it a scanned image?"

        vectordb = build_vector_store(chunks)
        retriever_obj = vectordb.as_retriever()
        logger.info("Retriever ready")
        return retriever_obj, f"Document indexed: {len(chunks)} chunks ready."
    except Exception as exc:
        logger.exception("Failed to build retriever")
        return None, f"Failed to process document: {exc}"


# Query step — receives the cached retriever from gr.State and runs the LCEL chain.
def answer_question(query: str, retriever_obj: VectorStoreRetriever | None) -> str:
    if retriever_obj is None:
        return "Please process a document first."
    if not query or not query.strip():
        return "Please enter a question."
    logger.info("Answering query: %r", query)
    try:
        # LCEL chain — each stage pipes its output into the next:
        # 1. retriever: similarity-searches the vector store for the top-k most relevant chunks
        # 2. format_docs: joins those chunks into a single {context} string
        # 3. RAG_PROMPT: fills {context} and {question} into the prompt template
        # 4. LLM: generates the answer from the completed prompt
        # 5. StrOutputParser: unwraps the plain string from the AIMessage response object
        chain = (
            {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | LLM
            | StrOutputParser()
        )
        answer = chain.invoke(query)
        logger.info("Answer length: %d chars", len(answer))
        return answer
    except Exception as exc:
        logger.exception("Failed to answer question")
        return f"Failed to generate an answer: {exc}"


# gr.Blocks is Gradio's low-level layout API — components defined inside the
# `with` block are rendered top-to-bottom as a single-page web app.
with gr.Blocks(title="RAG Chatbot") as rag_application:
    # gr.State holds the retriever object server-side across button clicks for
    # this user session — Gradio passes it automatically as a function argument.
    retriever_state = gr.State(value=None)

    gr.Markdown("## RAG Chatbot\nUpload a PDF, process it, then ask questions.")

    # Two separate buttons enforce the intended flow: index once, query many times.
    # Combining them would re-embed the document on every question, wasting API calls.
    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        process_btn = gr.Button("Process Document", variant="primary")

    status_box = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("---")

    query_input = gr.Textbox(label="Question", lines=2, placeholder="Ask something about the document...")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)

    # .click() maps a button to a Python function — `inputs` lists the component
    # values to pass in; `outputs` lists the components to update with the return values.
    process_btn.click(
        fn=build_retriever,
        inputs=[file_input],
        outputs=[retriever_state, status_box],
    )
    ask_btn.click(
        fn=answer_question,
        inputs=[query_input, retriever_state],
        outputs=[answer_box],
    )


if __name__ == "__main__":
    logger.info("Starting QA Bot on %s:%d (share=%s)", config.GRADIO_SERVER_NAME, config.GRADIO_SERVER_PORT, config.GRADIO_SHARE)
    rag_application.launch(server_name=config.GRADIO_SERVER_NAME, server_port=config.GRADIO_SERVER_PORT, share=config.GRADIO_SHARE)
