# QBV Browser

### Create Environment
```conda env create -f environment.yml```

### Initialize
- Place audios to be retrieved into `static\items` - e.g., `esc50`
- Clone QBV repository `git clone https://github.com/jonathan-greif/qbv.git`

### Run
`python main.py`

### How to Extend
- Add new backend to `retrieval_backends`
  - a `rank(item_paths, query_path, cache=None)` function for ranking
  - a `forward_batch(paths)` function for caching
- add new backend in `retrieval.py` in
  - `cache_item_embeddings()`, 
  - `rank()`, and 
  - `get_retrieval_backends()`.

### TODOs
- [ ] add QVB backends from DCASE paper
- [ ] test mp3 support
- [x] PANNs
- [x] VGGish
- [x] MVGGish
- [x] test GPU support
- [x] test handling of longer audio recordings
- [x] load retrieval for backends individually
- [x] set default backend (VGGish & None)
- [x] speed up loading of all candidate items
- [x] remove duration