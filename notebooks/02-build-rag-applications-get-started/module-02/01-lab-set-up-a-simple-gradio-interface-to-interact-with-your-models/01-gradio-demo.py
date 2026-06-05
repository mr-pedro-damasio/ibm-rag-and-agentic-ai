import gradio as gr
import config


def add_numbers(num1: float, num2: float) -> float:
    return num1 + num2


demo = gr.Interface(
    fn=add_numbers,
    inputs=[gr.Number(), gr.Number()],
    outputs=gr.Number(),
)

demo.launch(server_name=config.GRADIO_SERVER_NAME, server_port=config.GRADIO_SERVER_PORT, share=config.GRADIO_SHARE)
