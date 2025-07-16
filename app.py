import os
from flask import Flask, render_template, request, Response, jsonify
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Retrieve API keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# Check if API keys are set
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY is not set in the .env file.")
    exit(1)
if not GOOGLE_CLOUD_API_KEY:
    print("Warning: GOOGLE_CLOUD_API_KEY is not set in the .env file.")
    print("Image generation functionality may not work correctly.")

# Endpoint for the main HTML page
@app.route('/')
def index():
    # Flask will automatically look for 'templates' folder for HTML files.
    # For simplicity, if index.html is in the root, it will be served.
    # If you later move index.html to a 'templates' subfolder, Flask will find it there.
    return render_template('index.html')

# Endpoint for AI chat completions (text generation)
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    requested_model = data.get('model', 'anthropic/claude-3-haiku') # Get the model requested by the frontend
    messages = data.get('messages', [])
    stream = data.get('stream', True)
    max_tokens = data.get('max_tokens', 400)

    # Determine the actual model to use based on content
    actual_model_to_use = requested_model

    # Check if any message contains image data
    has_image_input = False
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    has_image_input = True
                    break
        if has_image_input:
            break

    # If there's image input, and the requested model is not vision-capable,
    # override it to a vision-capable model (like GPT-4o)
    if has_image_input and requested_model == 'anthropic/claude-3-haiku':
        actual_model_to_use = 'openai/gpt-4o' # Automatically switch to a vision model
        app.logger.info(f"Image input detected. Overriding model from {requested_model} to {actual_model_to_use}")
    
    # Headers for OpenRouter API, including your API key from .env
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Important for OpenRouter attribution: replace with your actual domain and app name
        "HTTP-Referer": "http://localhost:5000", # Change this to your deployed domain later
        "X-Title": "Z Chat App" # Your application name
    }

    payload = {
        "model": actual_model_to_use, # Use the potentially overridden model
        "messages": messages,
        "stream": stream, 
        "max_tokens": max_tokens
    }

    try:
        # Make the request to OpenRouter API
        # stream=True is crucial for handling streaming responses
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers,
                                 json=payload,
                                 stream=True, # Enable streaming for requests library
                                 timeout=120) # Set a reasonable timeout (e.g., 120 seconds)

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        def generate():
            # Iterate over the response stream from OpenRouter
            for chunk in response.iter_content(chunk_size=None):
                # OpenRouter sends SSE-like chunks. We yield them directly.
                # The frontend JavaScript is now adjusted to parse these raw chunks.
                yield chunk

        # Return a streaming response to the client
        return Response(generate(), content_type='text/event-stream')

    except requests.exceptions.RequestException as e:
        # Log the error on the server side
        app.logger.error(f"Error calling OpenRouter API: {e}")
        # Return a JSON error response to the client
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        app.logger.error("Failed to decode JSON from OpenRouter API response.")
        return jsonify({"error": "Invalid JSON response from AI service."}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in chat endpoint: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Endpoint for image generation using Google's Imagen
@app.route('/api/image-generate', methods=['POST'])
def image_generate():
    if not GOOGLE_CLOUD_API_KEY:
        return jsonify({"error": "Google Cloud API Key is not configured for image generation."}), 500

    data = request.json
    prompt = data.get('instances', {}).get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required for image generation."}), 400

    image_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={GOOGLE_CLOUD_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }

    try:
        # Make the request to Google Imagen API
        response = requests.post(image_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        # Return the JSON response directly from Imagen to the client
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling Google Imagen API: {e}")
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        app.logger.error("Failed to decode JSON from Google Imagen API response.")
        return jsonify({"error": "Invalid JSON response from image service."}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during image generation: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
