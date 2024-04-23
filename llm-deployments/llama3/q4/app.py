from flask import Flask, request
from llama_cpp import Llama
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

llm = Llama(
    model_path="llama3.gguf",
    n_ctx=2048,
)

@app.route("/v1/models/llama3:predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data["prompt"]
    max_tokens = data.get("max_tokens")
    echo = data.get("echo", False)
    stop = data.get("stop", ["Q:", "\n"])
    
    print(f"Prompt: {prompt}")
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=stop,
        echo=echo
    )
    return {
        "output": output
    }

if __name__ == "__main__":
    http_server = WSGIServer(("", 8080), app)
    http_server.serve_forever()
