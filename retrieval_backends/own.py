import os
import torch
import torchaudio
import tqdm
import urllib.parse
from torch.nn.functional import cosine_similarity

import numpy as np
from numpy.linalg import norm

from qbv.helpers.get_module import get_module
from qbv.helpers.utils_test import get_single_emb, padding

# GitHub release URL and local directory to store models
model_url = "https://github.com/Jonathan-Greif/QBV/releases/download/v1.0.0/"
model_dir = "resources"

# Pretrained models dictionary
pretrained_models = {
    "coarse_fold3": urllib.parse.urljoin(model_url, "ct_nt_xent_fold3mn10d10s32_01.pt"),
}

os.makedirs(model_dir, exist_ok=True)

def download_model(name, url):
    # Define the file path in the local directory
    file_path = os.path.join(model_dir, name + ".pt")

    if not os.path.exists(file_path):
        print(f"Downloading {name} model...")
        try:
            # Download the file
            urllib.request.urlretrieve(url, file_path)
            print(f"Model {name} downloaded and saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")



def get_own(fold=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    download_model("ct_nt_xent_fold3mn10d10s32_01", pretrained_models[f"coarse_fold{fold}"])
    m_own_ref, m_own_im, _ = get_module("MN", pretrained=False, own=True,
                                             state_dict_module="resources/ct_nt_xent_fold3mn10d10s32_01.pt",
                                             state_dict_pretrained=(None, None), fold=3)
    for m in [m_own_ref, m_own_im]:
        m.to(device)
        m.eval()
    def forward_own(audio, imitation=False):
        with torch.no_grad():
            audio = audio.numpy()[0]
            audio = padding(audio, 32000, 10)
            audio = torch.from_numpy(audio)
            if imitation:
                embedding = get_single_emb(m_own_im, "MN", audio)
            else:
                embedding = get_single_emb(m_own_ref, "MN", audio)
            return embedding.detach().cpu().numpy()

    return forward_own, 32000

def forward_audio(fwd_fun, model_sr, path, imitation=False):
    audio, sr = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, sr, model_sr)
    embedding = fwd_fun(audio, imitation)
    return embedding

def forward_batch(batch):
    fwd_fun, model_sr = get_own()
    embeddings = {}
    for path in tqdm.tqdm(batch):
        embeddings[path] = forward_audio(fwd_fun, model_sr, path)
    return embeddings

def rank_average(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_own()
    query_embedding = forward_audio(fwd_fun, model_sr, query_path, True)

    # compute similarity
    similarities = {}
    for item_path in item_paths:
        if cache is not None and item_path in cache:
            item_embedding = cache[item_path]
        else:
            item_embedding = forward_audio(fwd_fun, model_sr, item_path)
        sim = np.dot(item_embedding, query_embedding) / (norm(item_embedding)*norm(query_embedding))
        similarities[item_path] = sim.item()
    return similarities

def rank_align(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_own()
    query_embedding = forward_audio(fwd_fun, model_sr, query_path, True)

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