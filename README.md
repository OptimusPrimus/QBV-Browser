# QBV Browser

### Create Environment
```conda env create -f environment.yml```

### Initialize
Place audios to be retrieved into `static\items` - tested with ESC50

### Run
`python main.py`

### How to Extend
- Add new backend to `retrieval_backends` 
- In `retrieval.py` update:
  - `cache_item_embeddings()`, 
  - `rank()`, and 
  - `get_retrieval_backends()`.

### TODOs
- [ ] test GPU support
- [ ] test mp3 support
- [ ] add QVB backends from DCASE paper
- [ ] handling of longer audio recordings
