import os
from flask import Flask, render_template, send_from_directory, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# ✅ يخدم index.html
@app.route('/')
def index():
    return render_template('index.html')


# ✅ يخدم الملفات الثابتة (CSS/JS/صور)
@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


# ✅ البروكسي الخاص بـ OpenRouter (بدون تغيير js)
@app.route('/api/v1/chat/completions', methods=['POST'])
def proxy_openrouter():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "API key is not set"}), 500

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        req_data = request.get_json()
        stream = req_data.get("stream", False)

        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=req_data,
            stream=stream
        )

        if stream:
            def generate():
                for chunk in resp.iter_lines():
                    if chunk:
                        yield chunk + b"\n"
            return Response(generate(), content_type="text/event-stream")
        else:
            return jsonify(resp.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
