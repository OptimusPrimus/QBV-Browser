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


def audio_to_snippets(audio_array: torch.Tensor, sampling_rate: int, snippet_length: float, hop_size: float):
    """
    Splits an audio array into a batch of snippets.

    Args:
        audio_array (torch.Tensor): The input audio array (1D tensor).
        sampling_rate (int): Sampling rate of the audio in Hz.
        snippet_length (float): Length of each snippet in seconds.
        hop_size (float): Hop size between snippets in seconds.

    Returns:
        torch.Tensor: A batch of snippets (2D tensor of shape [num_snippets, snippet_samples]).
    """
    # Ensure the input audio array is 1D
    if len(audio_array.shape) != 1:
        raise ValueError("audio_array must be a 1D tensor.")

    # Convert snippet length and hop size from seconds to samples
    snippet_samples = int(snippet_length * sampling_rate)
    hop_samples = int(hop_size * sampling_rate)

    if snippet_samples <= 0 or hop_samples <= 0:
        raise ValueError("snippet_length and hop_size must result in at least one sample.")

    # If the input audio is too short for even one snippet, return the original audio as a single snippet
    if len(audio_array) < snippet_samples:
        return audio_array.unsqueeze(0)  # Add a batch dimension

    # Calculate the total number of snippets
    num_snippets = (len(audio_array) - snippet_samples) // hop_samples + 1

    # Create a list of snippets
    snippets = []
    for i in range(num_snippets):
        start_idx = i * hop_samples
        end_idx = start_idx + snippet_samples
        snippets.append(audio_array[start_idx:end_idx])

    # Stack snippets into a 2D tensor
    snippets_batch = torch.stack(snippets)

    return snippets_batch

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
    e_ = rank_align(['../static/items/014_House_Fan.wav'], '../static/items/014_House_Fan.wav')
    e = forward_batch_align(['../static/items/014_House_Fan.wav'])
    print(e)

    e = forward_batch_align(['../static/items/014_House_Fan.wav'])
    print(e)