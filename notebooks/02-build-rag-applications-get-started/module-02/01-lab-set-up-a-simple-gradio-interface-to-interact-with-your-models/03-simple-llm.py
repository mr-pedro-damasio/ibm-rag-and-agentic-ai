from llm_setup import llm_model, llm_response

# Only override what differs from config.py defaults
params = {
    "temperature": 0.8,
    "max_tokens": 1024,
    "max_completion_tokens": 512,
}

if __name__ == "__main__":
    model = llm_model(params=params)
    print(model.invoke("What is the capital of France?").content)
