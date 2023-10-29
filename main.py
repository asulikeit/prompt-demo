'''
    https://www.gradio.app/guides/creating-a-chatbot-fast
'''
import json, requests
import uvicorn
import gradio as gr
from fastapi import FastAPI


# Set no_proxy
session = requests.Session()
session.trust_env = False

app = FastAPI()

def inference(server_url, prompt, temperature, max_new_token, top_p, top_k, penalty):
    input = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_token": max_new_token,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": penalty
        }
    }
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    resp = session.post(server_url, data=json.dump(input).encode('utf-8'), headers=headers)
    return resp.json[0]['generated_text']


with gr.Blocks() as demo:
    gr.Markdown("""
        # Prompt Test Lab for LLM
        Test your LLM by prompt engineering.
    """)
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot()
                # chatbot = gr.ChatInterface()
            with gr.Row():
                prompt = gr.Textbox(label="Chat message", placeholder="Type chat messages")
        
        with gr.Column(scale=1):
            temperature = gr.Slider(label='Temperature', minimum=0, maximum=1, value=0.9)
            max_length = gr.Slider(label='Maximum_length', minimum=0, maximum=1000, value=200)
            top_p = gr.Slider(label='Top p', minimum=0, maximum=1, value=0.5)
            top_k = gr.Slider(label='Top k', minimum=0, maximum=500, value=50)
            reset = gr.Button('Reset')

    def answer(prompt, chat_history, temparature, max_length, top_p, top_k, req: gr.Request):
        server_url = None
        bot_message = None
        try:
            server_url = req.query_params['llm_url']
            bot_message = inference(prompt, temperature, max_length, top_p, top_k, False, server_url)
        except KeyError:
            bot_message = 'Please input llm_url as url parameters.'
        chat_history.append((prompt, bot_message))
        return "", chat_history

    prompt.submit(answer, [prompt, chatbot, temperature, max_length, top_p, top_k], [prompt, chatbot])
    reset.click(lambda: None, None, chatbot, queue=False)

app = gr.mount_gradio_app(app, demo, path='/')
    
if __name__ == 'main':
    uvicorn.run(
        app='main:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )
