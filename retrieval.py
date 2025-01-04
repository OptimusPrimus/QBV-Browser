from backend import get_query_files, get_item_files, get_audio_details
import os

import retrieval_backends.vggish
import retrieval_backends.panns

def get_retrieval_backends():
    return ["VGGish", "VGGish-align", "PANNs"]

# Search logic for different backends
def rank(backend_id, query, cache):

    query_path = query['file']
    item_paths = [i['file'] for i in get_item_files()]

    # Placeholder search logic
    if backend_id == "VGGish":
        similarities = retrieval_backends.vggish.rank_average(item_paths, query_path, cache=cache)
    elif backend_id == "PANNs":
        similarities = retrieval_backends.panns.rank_average(item_paths, query_path, cache=cache)
    elif backend_id == "VGGish-align":
        similarities = retrieval_backends.vggish.rank_align(item_paths, query_path, cache=cache)
    else:
        assert False
    # sort keys by similarity
    filenames_ranked = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
    # create return items
    items = []
    for i, f in enumerate(filenames_ranked):
        d, s = get_audio_details(f)
        items.append({
            "title": f.split(os.sep)[-1].split(".")[0],
            "file": f,
            "duration": d,
            "size": s,
            "similarity": round(similarities[f],2)
        }
        )
    return items

def cache_item_embeddings(backend_id):

    item_paths = [i['file'] for i in get_item_files()]
    if backend_id in ["VGGish", "VGGish-align"]:
        return retrieval_backends.vggish.forward_batch(item_paths)
    if backend_id in ["PANNs"]:
        return retrieval_backends.panns.forward_batch(item_paths)
    else:
        assert False













