import os
import dotenv
dotenv.load_dotenv()

# LinkedIn
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"
LINKEDIN_API_ENDPOINT = "https://nubela.co/proxycurl/api/v2/linkedin"
PROXYCURL_API_KEY = os.getenv("PROXY_CURL_API_KEY")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Embedding model
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_MODEL_DIMENSIONS = 1024

# Query model
QUERY_MODEL = "openai/gpt-4o-mini"
QUERY_MODEL_TEMPERATURE = 0.0
QUERY_MODEL_MAX_TOKENS = 2048
QUERY_MODEL_TOP_K = 5
QUERY_MODEL_TOP_P = 1.0

# Chunking
CHUNK_SIZE = 500

# Prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.
Context information is below:
{context_str}
Based on the context provided, list 3 interesting facts about this person's career or education.
Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.
Context information is below:
{context_str}
Question: {query_str}
Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
