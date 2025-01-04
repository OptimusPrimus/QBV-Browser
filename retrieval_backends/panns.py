import torch
import torchaudio
import tqdm
from torch.nn.functional import cosine_similarity
from panns_inference import AudioTagging, SoundEventDetection, labels

import numpy as np
from numpy.linalg import norm

def get_PANNs():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    at = AudioTagging(checkpoint_path=None, device=device)

    def forward_panns(audio):
        with torch.no_grad():
            audio = audio.numpy()
            return at.inference(audio)[1]

    return forward_panns, 32000

def forward_audio(fwd_fun, model_sr, path):
    audio, sr  = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, sr, model_sr)
    embedding = fwd_fun(audio)
    return embedding

def forward_batch(batch):
    fwd_fun, model_sr = get_PANNs()
    embeddings = {}
    for path in tqdm.tqdm(batch):
        embeddings[path] = forward_audio(fwd_fun, model_sr, path)
    return embeddings

def rank_average(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_PANNs()
    query_embedding = forward_audio(fwd_fun, model_sr, query_path).mean(0)

    # compute similarity
    similarities = {}
    for item_path in item_paths:
        if cache is not None and item_path in cache:
            item_embedding = cache[item_path].mean(0)
        else:
            item_embedding = forward_audio(fwd_fun, model_sr, item_path).mean(0)
        sim = np.dot(item_embedding, query_embedding) / (norm(item_embedding)*norm(query_embedding))
        similarities[item_path] = sim.item()
    return similarities


if __name__ == '__main__':
    forward_batch(['../static/items/1-9887-B-49.wav'])