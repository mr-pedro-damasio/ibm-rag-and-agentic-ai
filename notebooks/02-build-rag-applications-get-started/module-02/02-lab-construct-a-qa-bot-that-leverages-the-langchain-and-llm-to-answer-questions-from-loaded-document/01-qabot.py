import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
import gradio as gr

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_EMBEDDINGS_MODEL = "openai/text-embedding-3-small"

GRADIO_SERVER_NAME = "localhost"
GRADIO_SERVER_PORT = 7860

def llm_model(model_name, model_api_key, model_base_url, model_temperature, model_maxtokens, model_maxcompletiontokens):

    model = ChatOpenRouter(
        model=model_name,
        api_key=model_api_key,
        base_url=model_base_url,
        temperature=model_temperature,
        max_tokens=model_maxtokens,
        max_completion_tokens=model_maxcompletiontokens
    )
    return model


def llm_model_response(prompt_text, model_name, model_api_key, model_base_url, model_temperature, model_maxtokens, model_maxcompletiontokens):
            
    model = llm_model(model_name=model_name, model_api_key=model_api_key, model_base_url=model_base_url, model_temperature=model_temperature, model_maxtokens=model_maxtokens, model_maxcompletiontokens=model_maxcompletiontokens)
    response = model.invoke(prompt_text)
    return response


## Document loading 
def document_loader(file):
    # Gradio with type="filepath" provides a string path.
    file_path = file if isinstance(file, str) else file.name
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document


## Text splitting
def text_splitter(data_to_split, chunk_size=1000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data_to_split)
    return chunks


## Embedding model
def embbeding_model(model_embedding, model_api_key, model_base_url, model_dimensions=1024):
    embeddings_model = OpenAIEmbeddings(
        model=model_embedding,
        openai_api_key=model_api_key,
        openai_api_base=model_base_url,
        dimensions=model_dimensions    
    )
    return embeddings_model


## Vector db
def vector_database(chunks, model_embedding, model_api_key, model_base_url, model_dimensions):
    embeddings_model = embbeding_model(model_embedding, model_api_key, model_base_url, model_dimensions)
    vectordb = Chroma.from_documents(chunks, embeddings_model)
    return vectordb


## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks, OPENROUTER_EMBEDDINGS_MODEL, OPENROUTER_API_KEY, OPENROUTER_DEFAULT_BASE_URL, 1024)
    retriever = vectordb.as_retriever()
    return retriever


## QA Chain
def retriever_qa(file, query):
    llm = llm_model(model_name=OPENROUTER_MODEL, model_api_key=OPENROUTER_API_KEY, model_base_url=OPENROUTER_DEFAULT_BASE_URL, model_temperature=0.7, model_maxtokens=2048, model_maxcompletiontokens=2048)
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke({"query": query})
    return response['result']


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


# Launch the app
rag_application.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT, share=True)