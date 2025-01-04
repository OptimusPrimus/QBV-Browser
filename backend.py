import os
from glob import glob
from pydub.utils import mediainfo
import math

ITEM_FOLDER = os.path.join('static', 'items')
QUERY_FOLDER = os.path.join('static', 'queries')

def get_files(folder):
    files = [f for f in glob(os.path.join(folder, '*.wav'))]
    files += [f for f in glob(os.path.join(folder, '*.mp3'))]

    jsons = []

    for i, f in enumerate(files):
        d, s = get_audio_details(f)
        jsons.append({
            "title": f.split(os.sep)[-1].split(".")[0],
            "file": f,
            "duration": d,
            "size": s}
        )
    return jsons

def save_file(file, filename):
    file_path = os.path.join(QUERY_FOLDER, filename)
    file.save(file_path)

    q = {"title": file_path.split(os.sep)[-1].split(".")[0], "file": file_path}
    duration, size = get_audio_details(file_path)
    q['duration'] = duration
    q['size'] = size
    return q

def delete_file(query):
    if os.path.exists(query['file']):
        os.remove(query['file'])

# Helper function to get audio details (duration and size)
def get_audio_details(file_path):
    try:
        info = mediainfo(file_path)
        duration = float(info['duration'])  # Duration in seconds
        size = os.path.getsize(file_path) / 1024  # File size in KB
        return round(duration,2), round(size,2)
    except Exception as e:
        print(f"Error getting audio details: {e}")
        return 0, 0  # In case of error, return zero values

def get_item_files():
    return get_files(ITEM_FOLDER)

def get_query_files():
    return list(reversed(sorted(get_files(QUERY_FOLDER), key=lambda q: q['title'])))

def update_needed(items):
    files = [f for f in glob(os.path.join(ITEM_FOLDER, '*.wav'))]
    files += [f for f in glob(os.path.join(ITEM_FOLDER, '*.mp3'))]

    actual_files = set(files)
    current_files = set([i['file'] for i in items])
    return not actual_files == current_files


if __name__ == '__main__':
    print(get_item_files())
