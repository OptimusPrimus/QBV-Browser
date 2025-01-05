import torchaudio
import tqdm
from numpy.linalg import norm
import numpy as np
import torch
import torch.nn as nn

def get_VGGish():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = M_VGGish(sr=16000)
    model.to(device)
    model.eval()

    def forward_vggish(audio):
        with torch.no_grad():
            audio = audio.numpy()[0]
            return model(audio).detach().cpu().numpy()

    return forward_vggish, 16000

def forward_audio(fwd_fun, model_sr, path):
    audio, sr  = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, sr, model_sr)
    embedding = fwd_fun(audio)
    return embedding

def forward_batch(batch):
    fwd_fun, model_sr = get_VGGish()
    embeddings = {}
    for path in tqdm.tqdm(batch):
        embeddings[path] = forward_audio(fwd_fun, model_sr, path)
    return embeddings

def rank_average(item_paths, query_path, cache=None):
    fwd_fun, model_sr = get_VGGish()
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


class M_VGGish(nn.Module):
    """
    2s in the style of: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683461
    """

    def __init__(self, sr=16000):
        super(M_VGGish, self).__init__()
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.postprocess = False
        self.l5 = self.model.features[:12]
        self.l6 = self.model.features[:14]

        self.sr = sr  # sampling rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):

        x = self.model._preprocess(x, self.sr).to(self.device)

        x1 = self.l5(x).reshape(len(x), -1)
        x2 = self.l6(x).reshape(len(x), -1)

        x = torch.cat((x1, x2), dim=1)
        x = x.mean(dim=0)[None, :]
        return x

    def get_number_of_2s_segments(self, x):
        n_segments = np.ceil((len(x) / self.sr))
        if n_segments % 2 != 0:
            container = np.zeros(len(x) + self.sr)
            container[:len(x)] = x
            x = container
            n_segments += 1
        return int(n_segments / 2), x


if __name__ == '__main__':
    e_ = rank_average(['../static/items/esc50/1-13572-A-46.wav'], '../static/items/esc50/1-7974-A-49.wav')
    e = forward_batch(['../static/items/esc50/1-7974-A-49.wav'])
    print(e)