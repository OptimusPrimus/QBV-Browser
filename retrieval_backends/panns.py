import torch
import torchaudio
import tqdm
from panns_inference import AudioTagging

import numpy as np
from numpy.linalg import norm

from retrieval_backends.audio_helpers import audio_to_snippets


def get_PANNs():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    at = AudioTagging(checkpoint_path=None, device=device)
    def forward_panns(audio):
        with torch.no_grad():
            audio = audio.numpy()
            return at.inference(audio)[1]
    return forward_panns, 32000

def get_PANNs_snippets(sl, hs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    at = AudioTagging(checkpoint_path=None, device=device)
    def forward_panns(audio):
        with torch.no_grad():
            audio = audio_to_snippets(audio[0], 32000, snippet_length=sl, hop_size=hs)
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

def forward_batch_align(batch):
    fwd_fun, model_sr = get_PANNs_snippets(5,1)
    embeddings = {}
    for path in tqdm.tqdm(batch):
        embeddings[path] = forward_audio(fwd_fun, model_sr, path)
    return embeddings


def rank_align(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_PANNs_snippets(5, 1)
    query_embedding = forward_audio(fwd_fun, model_sr, query_path)

    # compute similarity
    similarities = {}
    for item_path in item_paths:
        if cache is not None and item_path in cache:
            item_embedding = cache[item_path]
        else:
            item_embedding = forward_audio(fwd_fun, model_sr, item_path)
        sim = calculate_max_similarity(query_embedding, item_embedding)
        similarities[item_path] = sim.item()
    return similarities



def calculate_max_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate the highest similarity score between two 2-D arrays (time, representation),
    where the shorter array is slid over the longer array.

    Args:
        array1 (np.ndarray): First 2D array of shape (time1, representation).
        array2 (np.ndarray): Second 2D array of shape (time2, representation).

    Returns:
        float: The highest similarity score.
    """
    # Determine which array is shorter and which is longer
    if array1.shape[0] > array2.shape[0]:
        longer, shorter = array1, array2
    else:
        longer, shorter = array2, array1
    # Normalize the arrays along the representation dimension
    def normalize(array):
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.clip(norm, a_min=1e-8, a_max=None)

    longer = normalize(longer)
    shorter = normalize(shorter)

    # Sliding window comparison
    max_similarity = -np.inf
    for start_idx in range(len(longer) - len(shorter) + 1):
        # Extract a segment from the longer array
        segment = longer[start_idx: start_idx + len(shorter)]

        # Calculate similarity (dot product across all time steps and features)
        similarity = np.sum(segment * shorter)
        max_similarity = max(max_similarity, similarity)

    return max_similarity / len(shorter)


# Example usage
if __name__ == "__main__":
    # Create a dummy audio array (e.g., 5 seconds of 44.1 kHz audio)
    dummy_audio = torch.randn(5 * 44100)
    sampling_rate = 44100
    snippet_length = 1.0  # 1 second snippets
    hop_size = 0.5       # 50% overlap

    snippets = audio_to_snippets(dummy_audio, sampling_rate, snippet_length, hop_size)
    print(f"Generated snippets: {snippets.shape}")  # Shape: [num_snippets, snippet_samples]


if __name__ == '__main__':
    e_ = rank_align(['../static/items/014_House_Fan.wav'], '../static/items/clotho/014_House_Fan.wav')
    e = forward_batch_align(['../static/items/014_House_Fan.wav'])
    print(e)

    e = forward_batch_align(['../static/items/014_House_Fan.wav'])
    print(e)