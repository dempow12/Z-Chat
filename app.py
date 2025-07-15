from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# المسار الجذر - يعرض index.html
@app.route('/')
def home():
    return render_template('index.html')

# لدعم تحميل ملفات مثل CSS/JS/صور
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
