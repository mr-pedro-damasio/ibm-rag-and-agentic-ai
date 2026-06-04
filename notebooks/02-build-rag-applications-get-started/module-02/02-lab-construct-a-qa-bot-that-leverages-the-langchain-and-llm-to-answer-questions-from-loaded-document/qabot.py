import logging
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    CHUNK_SIZE, CHUNK_OVERLAP,
    GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE,
    RAG_PROMPT_TEMPLATE,
)
from langchain_openrouter import ChatOpenRouter
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Factory kept as a named function so the parameter list is explicit and the
# module-level LLM = llm_model(...) call site reads as configuration, not magic.
def llm_model(model_name, model_api_key, model_base_url, model_temperature, model_maxtokens):
    model = ChatOpenRouter(
        model=model_name,
        api_key=model_api_key,
        base_url=model_base_url,
        temperature=model_temperature,
        max_tokens=model_maxtokens,
    )
    return model


RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# LLM is stateless — built once at module load rather than once per question.
LLM = llm_model(
    model_name=LLM_MODEL,
    model_api_key=OPENROUTER_API_KEY,
    model_base_url=OPENROUTER_BASE_URL,
    model_temperature=LLM_TEMPERATURE,
    model_maxtokens=LLM_MAX_TOKENS,
)


def document_loader(file_path):
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


def text_splitter(data_to_split, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
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


def build_vector_store(chunks):
    # Embeds chunks into an in-memory Chroma store. The store is ephemeral —
    # it lives only for this session and is rebuilt on each new document upload.
    logger.info("Embedding %d chunks (model=%s, dim=%d)", len(chunks), EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,  # routes the OpenAI SDK's embedding calls through OpenRouter
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return Chroma.from_documents(chunks, embeddings)


def format_docs(docs):
    # Double newline acts as a clear paragraph separator; a single newline risks
    # fusing two unrelated chunks into a run-on passage inside the prompt context.
    return "\n\n".join(doc.page_content for doc in docs)


# Indexing step — runs once per document upload; result cached in gr.State so
# subsequent questions reuse the retriever without re-embedding.
def build_retriever(file):
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
def answer_question(query, retriever_obj):
    if retriever_obj is None:
        return "Please process a document first."
    if not query or not query.strip():
        return "Please enter a question."
    logger.info("Answering query: %r", query)
    try:
        # LCEL chain — reads left to right:
        # retriever fetches the most relevant chunks → format_docs joins them into one
        # context string → RAG_PROMPT fills {context} and {question} → LLM generates
        # the answer → StrOutputParser extracts the plain string from the message object.
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

    gr.HTML("<hr>")

    query_input = gr.Textbox(label="Question", lines=2, placeholder="Ask something about the document...")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)

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


logger.info("Starting QA Bot on %s:%d (share=%s)", GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE)
rag_application.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT, share=GRADIO_SHARE)
