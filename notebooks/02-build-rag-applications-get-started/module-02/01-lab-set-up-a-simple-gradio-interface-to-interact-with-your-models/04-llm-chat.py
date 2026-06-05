import gradio as gr
import config
from llm_setup import llm_model

# Only override what differs from config.py defaults
params = {
    "temperature": 0.8,
    "max_tokens": 1024,
    "max_completion_tokens": 512,
}

model = llm_model(params=params)


def generate_response(prompt_txt: str) -> str:
    return model.invoke(prompt_txt).content


chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="AI Chatbot",
    description="Ask any question and the chatbot will try to answer.",
    flagging_mode="never",
)

if __name__ == "__main__":
    chat_application.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
    )
