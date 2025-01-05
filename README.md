# QBV Browser

### Create Environment
```conda env create -f environment.yml```

### Initialize
- Place audios to be retrieved into `static\items` - see ESC50 in `esc50`
- Clone QBV repository `git clone https://github.com/jonathan-greif/qbv.git`

### Run
`python main.py`

### How to Extend
- Add new backend to `retrieval_backends` 
- In `retrieval.py` update:
  - `cache_item_embeddings()`, 
  - `rank()`, and 
  - `get_retrieval_backends()`.

### TODOs
- [x] test GPU support
- [ ] test mp3 support
- [x] test handling of longer audio recordings
- [ ] add QVB backends from DCASE paper
- [x] load retrieval for backends individually
- [x] set default backend (VGGish & None)
- [x] speed up loading of all candidate items
- [x] remove duration