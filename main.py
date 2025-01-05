from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import time
from backend import get_item_files, get_query_files, save_file, delete_file, update_needed
from retrieval import cache_item_embeddings, rank, get_retrieval_backends
import random
app = Flask(__name__)

# Use the 'static/recorded_queries' directory for saving audio files
MAX_RESULTS = 100
MAX_ITEM_DISPLAY = 100
CACHED_EMBEDDINGS = {b: cache_item_embeddings(b) for b in get_retrieval_backends()}

@app.route('/')
def index():
    random.seed(13)
    items = get_item_files()
    random.shuffle(items)
    items = sorted(items[:MAX_ITEM_DISPLAY], key=lambda x: x['title'])

    return render_template(
        'index.html',
        items=items,
        backends=get_retrieval_backends(),
        recorded_queries=get_query_files()
    )

@app.route('/record', methods=['POST'])
def record():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio data provided"}), 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    q = save_file(audio_file, file_name)
    return jsonify(q)

@app.route('/delete_query/<query_id>', methods=['POST']) ##
def delete_query(query_id):
    query = next((q for q in get_query_files() if q['id'] == query_id), None)
    if query:
        delete_file(query)
        return jsonify({"success": True})
    return jsonify({"error": "Query not found."}), 404

@app.route('/search_results', methods=['POST'])
def search_results_backend():
    data = request.json
    query_id = data.get('query_id') ##
    backend = data.get('backend')
    print(query_id)
    print(get_query_files())
    query = next((q for q in get_query_files() if q['id'] == query_id), None)
    if not query:
        return jsonify({"error": "Query not found."}), 404

    if CACHED_EMBEDDINGS[backend] is None:
        print("Caching embedding for backend 1...")
        CACHED_EMBEDDINGS[backend] = cache_item_embeddings(backend)

    # Get search results from both backends
    results_backend = rank(backend, query, CACHED_EMBEDDINGS[backend])[:MAX_RESULTS]

    return jsonify({"results": results_backend})

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    for backend in get_retrieval_backends():
        if CACHED_EMBEDDINGS.get(backend) is None or update_needed(CACHED_EMBEDDINGS.get(backend)):
            CACHED_EMBEDDINGS[backend] = cache_item_embeddings(backend)

    # Simulate processing logic
    return jsonify(success=True)

@app.route('/get_button_status', methods=['GET'])
def get_button_status():
    # Example logic to determine the status of the backend
    status = "ready"  # Options: "ready", "pending", "error"
    for backend in get_retrieval_backends():
        if CACHED_EMBEDDINGS.get(backend):
            if update_needed(CACHED_EMBEDDINGS.get(backend)):
                status = "pending"
        else:
            status = "error"
        return jsonify({"status": status})

if __name__ == '__main__':
    app.run(debug=True)

