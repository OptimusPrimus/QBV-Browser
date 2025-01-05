import torch


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
