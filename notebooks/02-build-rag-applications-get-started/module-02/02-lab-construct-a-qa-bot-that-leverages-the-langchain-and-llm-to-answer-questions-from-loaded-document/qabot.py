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

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def llm_model(model_name, model_api_key, model_base_url, model_temperature, model_maxtokens):
    model = ChatOpenRouter(
        model=model_name,
        api_key=model_api_key,
        base_url=model_base_url,
        temperature=model_temperature,
        max_tokens=model_maxtokens,
    )
    return model


## Document loading
def document_loader(file_path):
    logger.info("Loading PDF from: %s", file_path)
    reader = PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i}
        )
        for i, page in enumerate(reader.pages)
    ]


## Text splitting
def text_splitter(data_to_split, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    logger.info("Splitting %d pages into chunks (size=%d, overlap=%d)", len(data_to_split), chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(data_to_split)
    logger.info("Produced %d chunks", len(chunks))
    return chunks


## Embedding model
def embedding_model(model_embedding, model_api_key, model_base_url, model_dimensions=1024):
    logger.info("Initialising embedding model: %s", model_embedding)
    embeddings_model = OpenAIEmbeddings(
        model=model_embedding,
        openai_api_key=model_api_key,
        openai_api_base=model_base_url,
        dimensions=model_dimensions
    )
    return embeddings_model


## Vector db
def vector_database(chunks, model_embedding, model_api_key, model_base_url, model_dimensions):
    logger.info("Embedding %d chunks into Chroma", len(chunks))
    embeddings_model = embedding_model(model_embedding, model_api_key, model_base_url, model_dimensions)
    vectordb = Chroma.from_documents(chunks, embeddings_model)
    return vectordb


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## Build retriever — indexing step, runs once per document
def build_retriever(file):
    if file is None:
        return None, "No file uploaded."
    logger.info("Building retriever for: %s", file)
    splits = document_loader(file)
    chunks = text_splitter(splits)
    logger.info("Split into %d chunks; embedding now...", len(chunks))
    vectordb = vector_database(chunks, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_DIMENSIONS)
    retriever_obj = vectordb.as_retriever()
    logger.info("Retriever ready")
    return retriever_obj, f"Document indexed: {len(chunks)} chunks ready."


## Answer question using cached retriever
def answer_question(query, retriever_obj):
    if retriever_obj is None:
        return "Please process a document first."
    if not query or not query.strip():
        return "Please enter a question."
    logger.info("Answering query: %r", query)
    llm = llm_model(
        model_name=LLM_MODEL,
        model_api_key=OPENROUTER_API_KEY,
        model_base_url=OPENROUTER_BASE_URL,
        model_temperature=LLM_TEMPERATURE,
        model_maxtokens=LLM_MAX_TOKENS,
    )
    chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(query)
    logger.info("Answer length: %d chars", len(answer))
    return answer


# Create Gradio Blocks interface
with gr.Blocks(title="RAG Chatbot") as rag_application:
    retriever_state = gr.State(value=None)

    gr.Markdown("## RAG Chatbot\nUpload a PDF, process it, then ask questions.")

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


# Launch the app
logger.info("Starting QA Bot on %s:%d (share=%s)", GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE)
rag_application.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT, share=GRADIO_SHARE)
