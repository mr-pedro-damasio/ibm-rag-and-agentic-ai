import config
import gradio as gr

def add_numbers(Num1, Num2):
    return Num1 + Num2

# Define the interface
demo = gr.Interface(
    fn=add_numbers, 
    inputs=[gr.Number(), gr.Number()], # Create two numerical input fields where users can enter numbers
    outputs=gr.Number() # Create numerical output fields
)

# Launch the interface
demo.launch(server_name=config.GRADIO_SERVER_NAME, server_port=config.GRADIO_SERVER_PORT, share=config.GRADIO_SHARE)
