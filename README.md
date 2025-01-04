# QBV Browser

### Create Environment
```conda env create -f environment.yml```

### Initialize
Place audios to be retrieved - wavs or mp3s (not tested) - into `static\items`

### Run
`python main.py`

### Extend
- Add new backend to `retrieval_backends` 
- In `retrieval.py` update:
  - `cache_item_embeddings()`, 
  - `rank()`, and 
  - `get_retrieval_backends()`.
