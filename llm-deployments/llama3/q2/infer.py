import argparse
from kserve import KServeClient
import requests

# Set up the argument parser
parser = argparse.ArgumentParser(description="Send a request to KServe")
parser.add_argument('--prompt', type=str, required=True, help='Prompt for the model')
parser.add_argument('--token', type=int, default=32, help='Number of tokens to generate')

# Parse the arguments
args = parser.parse_args()

# KServe client setup
KServe = KServeClient()
namespace = "my-profile"
isvc_resp = KServe.get("llama3", namespace=namespace)
isvc_url = isvc_resp["status"]["address"]["url"]

print(f"Making an API call to: {isvc_url}")

# Make the request
response = requests.post(
    isvc_url + "/v1/models/serving:predict",
    json={"prompt": args.prompt,
          "max_tokens": args.token,},
)

print(response.text)
