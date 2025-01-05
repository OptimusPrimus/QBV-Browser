# QBV Browser

### Initialize
- set up environment 
  - `conda env create -f environment.yml` or 
  - `conda env create -f environment_gpu.yml` or 
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