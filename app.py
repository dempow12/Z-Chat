import os
from flask import Flask, render_template, request, Response, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Retrieve API keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY is not set in the .env file.")
    exit(1)
if not GOOGLE_CLOUD_API_KEY:
    print("Warning: GOOGLE_CLOUD_API_KEY is not set in the .env file.")
    print("Image generation functionality may not work correctly.")

# Serve main HTML
@app.route('/')
def index():
    return render_template('index.html')

# Chat endpoint with POST + OPTIONS
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    data = request.json
    requested_model = data.get('model', 'anthropic/claude-3-haiku')
    messages = data.get('messages', [])
    stream = data.get('stream', True)
    max_tokens = data.get('max_tokens', 400)

    actual_model_to_use = requested_model

    has_image_input = any(
        isinstance(m.get('content'), list) and any(part.get('type') == 'image_url' for part in m['content'])
        for m in messages
    )

    if has_image_input and requested_model == 'anthropic/claude-3-haiku':
        actual_model_to_use = 'openai/gpt-4o'
        app.logger.info(f"Image input detected. Switching model to {actual_model_to_use}")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Z Chat App"
    }

    payload = {
        "model": actual_model_to_use,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        def generate():
            for chunk in response.iter_content(chunk_size=None):
                yield chunk

        return Response(generate(), content_type='text/event-stream')

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling OpenRouter API: {e}")
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        app.logger.error("Invalid JSON response from OpenRouter API.")
        return jsonify({"error": "Invalid JSON response from AI service."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error occurred."}), 500

# Image generation endpoint
@app.route('/api/image-generate', methods=['POST', 'OPTIONS'])
def image_generate():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    if not GOOGLE_CLOUD_API_KEY:
        return jsonify({"error": "Google Cloud API Key is not configured."}), 500

    data = request.json
    prompt = data.get('instances', {}).get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    image_api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"imagen-3.0-generate-002:predict?key={GOOGLE_CLOUD_API_KEY}"
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }

    try:
        response = requests.post(image_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling Google Imagen API: {e}")
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        app.logger.error("Invalid JSON from Google Imagen API.")
        return jsonify({"error": "Invalid JSON response from image service."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error during image generation: {e}")
        return jsonify({"error": "Unexpected error occurred."}), 500

# Error handlers
@app.errorhandler(405)
def method_not_allowed(e):
    return make_response(jsonify({"error": "Method Not Allowed", "message": str(e)}), 405)

@app.errorhandler(404)
def not_found(e):
    return make_response(jsonify({"error": "Not Found", "message": str(e)}), 404)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
