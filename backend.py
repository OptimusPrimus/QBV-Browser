import os
from glob import glob
import librosa
from torchvision.datasets import folder

ITEM_FOLDER = os.path.join('static', 'items')
QUERY_FOLDER = os.path.join('static', 'queries')

FILE_CACHE = {}

ESC50_classnames = {
    0: "dog",
    1: "rooster",
    2: "pig",
    3: "cow",
    4: "frog",
    5: "cat",
    6: "hen",
    7: "insects",
    8: "sheep",
    9: "crow",
    10: "rain",
    11: "sea_waves",
    12: "crackling_fire",
    13: "crickets",
    14: "chirping_birds",
    15: "water_drops",
    16: "wind",
    17: "pouring_water",
    18: "toilet_flush",
    19: "thunderstorm",
    20: "crying_baby",
    21: "sneezing",
    22: "clapping",
    23: "breathing",
    24: "coughing",
    25: "footsteps",
    26: "laughing",
    27: "brushing_teeth",
    28: "snoring",
    29: "drinking_sipping",
    30: "door_wood_knock",
    31: "mouse_click",
    32: "keyboard_typing",
    33: "door_wood_creaks",
    34: "can_opening",
    35: "washing_machine",
    36: "vacuum_cleaner",
    37: "clock_alarm",
    38: "clock_tick",
    39: "glass_breaking",
    40: "helicopter",
    41: "chainsaw",
    42: "siren",
    43: "car_horn",
    44: "engine",
    45: "train",
    46: "church_bells",
    47: "airplane",
    48: "fireworks",
    49: "hand_saw"
}

def get_updated_title(name, folder):
    if folder == 'esc50':
        name = name.split('.')[-2]
        c = int(name.split('-')[-1])
        return ESC50_classnames[c]
    elif name == 'item':
        return name[:20]
    else:
        return (folder + ' - '+ name)[:20]


def build_item_from_path(path):
    if path in FILE_CACHE:
        return FILE_CACHE[path]
    else:
        print(f'loading {path}')
        filename = path.split(os.sep)[-1]
        parent_folder = path.split(os.sep)[-2]
        duration, size = get_audio_details(path)
        item = {
            "id": path.replace(os.sep, '-').replace('.', '-'),
            "title": get_updated_title(filename, parent_folder),
            "folder": parent_folder,
            "file": path,
            "duration": duration,
            "size": size
        }
        FILE_CACHE[path] = item
        return item


def get_files(folder):
    files = [f for f in glob(os.path.join(folder, '*.wav'))]
    files += [f for f in glob(os.path.join(folder, '*.mp3'))]
    files += [f for f in glob(os.path.join(folder, '*/*.wav'))]
    files += [f for f in glob(os.path.join(folder, '*/*.mp3'))]

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
        duration = librosa.get_duration(filename=file_path)
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
    files += [f for f in glob(os.path.join(ITEM_FOLDER, '*/*.wav'))]
    files += [f for f in glob(os.path.join(ITEM_FOLDER, '*/*.mp3'))]

    actual_files = set(files)
    current_files = set(items.keys())

    return not (len(actual_files - current_files) == 0)


if __name__ == '__main__':
    print(get_item_files())
