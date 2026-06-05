import config
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

def llm_model(params=None):

    # 1. Define sensible defaults
    config = {
        "model": os.getenv("OPENROUTER_MODEL"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_DEFAULT_BASE_URL"),
        "temperature": 0.5,
        "max_tokens": 256,
        "max_completion_tokens": 128
    }
    
    if params:
        config.update(params)
        
    # 3. Initialize the model
    model = ChatOpenRouter(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        max_completion_tokens=config["max_completion_tokens"]
    )

    return model

def llm_model_response(prompt_text, params=None):
            
    # 3. Initialize the model
    model = llm_model(params)

    response = model.invoke(prompt_text)

    return response


llm_model_name = "openai/gpt-4o-mini"

params = {
    "model": llm_model_name,
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "base_url": os.getenv("OPENROUTER_DEFAULT_BASE_URL"),
    "temperature": 0.8,
    "max_tokens": 1024,
    "max_completion_tokens": 512
}

model = llm_model(params=params)

print(model.invoke("What is the capital of France?"))
