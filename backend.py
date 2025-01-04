import os
from glob import glob

ITEM_FOLDER = os.path.join('static', 'items')
QUERY_FOLDER = os.path.join('static', 'queries')

FILE_CACHE = {}

def build_item_from_path(path):
    if path in FILE_CACHE:
        return FILE_CACHE[path]

    else:
        print(f'loading {path}')
        filename = path.split(os.sep)[-1].split(".")[0]
        duration, size = get_audio_details(path)
        item = {
            "title": filename,
            "file": path,
            "duration": duration,
            "size": size
        }
        FILE_CACHE[path] = item
        return item


def get_files(folder):
    files = [f for f in glob(os.path.join(folder, '*.wav'))]
    files += [f for f in glob(os.path.join(folder, '*.mp3'))]

    jsons = []
    for i, f in enumerate(files):
        jsons.append(build_item_from_path(f))
    return jsons

def save_file(file, filename):
    file_path = os.path.join(QUERY_FOLDER, filename)
    file.save(file_path)
    q = build_item_from_path(file_path)
    return q

def delete_file(query):
    if os.path.exists(query['file']):
        os.remove(query['file'])

    if query['file'] in FILE_CACHE:
        del FILE_CACHE[query['file']]

# Helper function to get audio details (duration and size)
def get_audio_details(file_path):
    try:
        size = os.stat(file_path).st_size
        # TODO: speed this up!
        #info = mediainfo(file_path)
        #duration = float(info['duration'])  # Duration in seconds
        size = size / 1024
        duration = 0
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
    current_files = set(items.keys())
    return not actual_files == current_files


if __name__ == '__main__':
    print(get_item_files())
