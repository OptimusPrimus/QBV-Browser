import torch
import torchaudio
import tqdm
from torch.nn.functional import cosine_similarity

import numpy as np
from numpy.linalg import norm

from qbv.helpers.utils_test import get_single_emb, padding
from qbv.helpers.cqt.cqt import cqt

def get_2DFT(duration=15):
    def forward_2DFT(audio):
        audio = audio.numpy()[0]
        audio = padding(audio, 8000, duration)
        c = cqt(audio, 12, 8000, 55, 2090)
        audio = c["cqt"]
        embedding = get_single_emb("", "2DFT", audio)
        return embedding.numpy()

    return forward_2DFT, 8000

def forward_audio(fwd_fun, model_sr, path):
    audio, sr = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, sr, model_sr)
    embedding = fwd_fun(audio)
    return embedding

def forward_batch(batch):
    fwd_fun, model_sr = get_2DFT()
    embeddings = {}
    for path in tqdm.tqdm(batch):
        embeddings[path] = forward_audio(fwd_fun, model_sr, path)
    return embeddings

def rank_average(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_2DFT()
    query_embedding = forward_audio(fwd_fun, model_sr, query_path)

    # compute similarity
    similarities = {}
    for item_path in item_paths:
        if cache is not None and item_path in cache:
            item_embedding = cache[item_path]
        else:
            item_embedding = forward_audio(fwd_fun, model_sr, item_path)
        sim = np.dot(item_embedding, query_embedding) / (norm(item_embedding)*norm(query_embedding))
        if np.isnan(sim):
            print(item_path, item_embedding)
            print(query_path, query_embedding)
            print(norm(item_embedding), norm(query_embedding))
            sim = np.array([0])
        similarities[item_path] = sim.item()
    return similarities

def rank_align(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_2DFT()
    query_embedding = forward_audio(fwd_fun, model_sr, query_path)

    def match_sequences(a, b):
        if len(a) <= len(b):
            a, b = b, a
        hops = len(a) - len(b)

        mses = []
        for i in range(hops+1):
            mses.append(((a[i:i + len(b)] - b) ** 2).mean())
        return np.min(mses) * -1

    # compute similarity
    similarities = {}
    for item_path in item_paths:
        if cache is not None and item_path in cache:
            item_embedding = cache[item_path]
        else:
            item_embedding = forward_audio(fwd_fun, model_sr, item_path)

        sim = match_sequences(item_embedding, query_embedding)
        similarities[item_path] = sim.item()
    return similarities


if __name__ == '__main__':
    e_ = rank_average(['../static/items/Breaking/Bones Breaking.wav'], '../static/items/Breaking/Bones Breaking.wav')
    e = forward_batch(['../static/items/Breaking/Bones Breaking.wav'])
    print(e)