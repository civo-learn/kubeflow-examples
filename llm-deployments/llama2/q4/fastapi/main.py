import json
from typing import Callable, List, Dict, Any, Generator
from functools import partial

import fastapi
import uvicorn
from fastapi import HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from anyio import create_memory_object_stream
from anyio.to_thread import run_sync
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel

config = {
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "stream": True,
}
llm = AutoModelForCausalLM.from_pretrained("llama-2-7b-chat.ggmlv3.q4_1.bin",
                                           model_type="llama",
                                           lib="avx2",
                                           gpu_layers=110, 
                                           threads=8,
                                           **config)
app = fastapi.FastAPI(title="Llama 2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    html_content = """
    <html>
        <head>
        </head>
        <body>
            Run 4-bit LLama 2!
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

class ChatCompletionRequestV0(BaseModel):
    prompt: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1024

@app.post("/v1/completions")
async def completion(request: ChatCompletionRequestV0, response_mode=None):
    response = llm(request.prompt)
    return response

async def generate_response(chat_chunks, llm):
    for chat_chunk in chat_chunks:
        response = {
            'choices': [
                {
                    'message': {
                        'role': 'system',
                        'content': llm.detokenize(chat_chunk)
                    },
                    'finish_reason': 'stop' if llm.is_eos_token(chat_chunk) else 'unknown'
                }
            ]
        }
        yield dict(data=json.dumps(response))
    yield dict(data="[DONE]")

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in request.messages])
    tokens = llm.tokenize(combined_messages)
    
    try:
        chat_chunks = llm.generate(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EventSourceResponse(generate_response(chat_chunks, llm))

async def stream_response(tokens, llm):
    try:
        iterator: Generator = llm.generate(tokens)
        for chat_chunk in iterator:
            response = {
                'choices': [
                    {
                        'message': {
                            'role': 'system',
                            'content': llm.detokenize(chat_chunk)
                        },
                        'finish_reason': 'stop' if llm.is_eos_token(chat_chunk) else 'unknown'
                    }
                ]
            }
            yield dict(data=json.dumps(response))
        yield dict(data="[DONE]")
    except Exception as e:
        print(f"Exception in event publisher: {str(e)}")

@app.post("/v2/chat/completions")
async def chatV2_endpoint(request: Request, body: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in body.messages])
    tokens = llm.tokenize(combined_messages)

    return EventSourceResponse(stream_response(tokens, llm))

@app.post("/v0/chat/completions")
async def chat(request: ChatCompletionRequestV0, response_mode=None):
    tokens = llm.tokenize(request.prompt)
    async def server_sent_events(chat_chunks, llm):
        for chat_chunk in llm.generate(chat_chunks):
            yield dict(data=json.dumps(llm.detokenize(chat_chunk)))
        yield dict(data="[DONE]")

    return EventSourceResponse(server_sent_events(tokens, llm))

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)