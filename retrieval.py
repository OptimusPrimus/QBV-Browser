import torch
import sys
import torchaudio
from torch.nn.functional import cosine_similarity
from backend import get_query_files, get_item_files

# from qbv.helpers.get_module import get_module


def get_VGGish():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.postprocess = True
    model.to(device)
    model.eval()

    def forward_vggish(audio):
        audio = audio.numpy()[0]
        return model(audio, 16000)

    return forward_vggish, 16000


def forward_audio(fwd_fun, model_sr, path):
    audio, sr  = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, sr, model_sr)
    embedding = fwd_fun(audio)
    return embedding

def forward_batch(backend, batch):
    if backend == "VGGish":
        fwd_fun, model_sr = get_VGGish()

    for item in batch:
        file = item["file"]
        item['embedding'] = forward_audio(fwd_fun, model_sr, file)

    return batch

def cache_item_embeddings(backend):
    items = get_item_files()
    items = forward_batch(backend, items)
    return items

def compare_embeddings(backend, items, file):

    if backend == "VGGish":
        fwd_fun, model_sr = get_VGGish()

    q_e = forward_audio(fwd_fun, model_sr, file).mean(0)

    # compute similarity
    for item in items:
        item['similarity'] = cosine_similarity(item['embedding'].mean(0)[None], q_e[None])

    # sort by similarity
    items_light = []
    for item in sorted(items, key=lambda i: i['similarity'], reverse=True):
        items_light.append(
            {
                "file": item["file"],
                "title": item["title"],
                "duration": item["duration"],
                "size": item["size"],
                "similarity": round(item["similarity"].item(), 2)
            }
        )
    return items_light

def get_retrieval_backends():
    return ["2DFFT", "VGGish", "MN"]

# Search logic for different backends
def rank(backend, items, query):
    # Placeholder search logic
    if backend == "2DFFT":
        return sorted(get_item_files(), key=lambda x: x['title'])
    elif backend == "VGGish":
        return compare_embeddings(backend, items, query['file'])
    elif backend == "MN":
        return get_item_files()

if __name__ == "__main__":


    items = get_item_files()
    queries = get_query_files()

    embedding = forward_batch("VGGish", items)

    compare_embeddings("VGGish", items, queries[0]['file'])

    print(embedding)









