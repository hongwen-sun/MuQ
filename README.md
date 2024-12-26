# <img src="images/muq-logo.jpeg" alt="" height="24px"> MuQ & MuQ-MuLan

<div>
  <a href='#'><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"></a>
  <a href='https://arxiv.org/'><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-Upcoming-%23b31b1b?logo=arxiv&link=https%3A%2F%2Farxiv.org%2F"></a>
  <a href='https://huggingface.co/OpenMuQ'><img alt="Static Badge" src="https://img.shields.io/badge/huggingface-OpenMuQ-%23FFD21E?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2FOpenMuQ"></a>
  <a href='https://pytorch.org/'><img alt="Static Badge" src="https://img.shields.io/badge/framework-PyTorch-%23EE4C2C?logo=pytorch"></a>
  <a href='https://pypi.org/project/muq'><img alt="Static Badge" src="https://img.shields.io/badge/pip%20install-muq-green?logo=PyPI&logoColor=white&link=https%3A%2F%2Fpypi.org%2Fproject%2Fmuq"></a>
</div>

This is the official repository for the paper *"**MuQ**: Self-Supervised **Mu**sic Representation Learning
 with Mel Residual Vector **Q**uantization"*.

In this repo, the following models are released:

- **MuQ**: A large music foundation model pre-trained via Self-Supervised Learning (SSL), achieving SOTA in various MIR tasks.
- **MuQ-MuLan**: A music-text joint embedding model trained via contrastive learning, supporting both English and Chinese texts.

## Overview

We develop the **MuQ** for music SSL. MuQ applys our proposed Mel-RVQ as quantitative targets and achieves SOTA performance on many music understanding (or MIR) tasks. 

We also construct the **MuQ-MuLan**, a CLIP-like model trained by contrastive learning, which jointly represents music and text into embeddings.

For more details, please refer to our [paper](https://arxiv.org/).

<div>
  <img src="images/radar.svg" width="45%" alt="Evaluation on MARBLE Benchmark">
  <img src="images/tagging.svg" width="45%" alt="Evaluation on Zero-shot Music Tagging">
</div>

## Usage

To begin with, please use pip to install the official `muq` lib, and ensure that your `python>=3.8`:
```bash
pip3 install muq
```

To extract music audio features using **MuQ**, you can refer to the following code:
```python
import torch, librosa
from muq import MuQ

device = 'cuda'
wav, sr = librosa.load("path/to/music_audio.wav", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 

# This will automatically fetch the checkpoint from huggingface
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

with torch.no_grad():
    output = muq(wavs, output_hidden_states=True)

print('Total number of layers: ', len(output.hidden_states))
print('Feature shape: ', output.last_hidden_state.shape)

```

Using **MuQ-MuLan** to extract the music and text embeddings and calculate the similarity:
```python
import torch, librosa
from muq import MuQMuLan

# This will automatically fetch checkpoints from huggingface
device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

# Extract music embeddings
wav, sr = librosa.load("path/to/music_audio.wav", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 
with torch.no_grad():
    audio_embeds = mulan(wavs = wavs) 

# Extract text embeddings (texts can be in English or Chinese)
texts = ["classical genres, hopeful mood, piano."]
with torch.no_grad():
    text_embeds = mulan(texts = texts)

# Calculate dot product similarity
sim = mulan.calc_similarity(audio_embeds, text_embeds)
print(sim)
```


## Performance

<img src="images/tab-marble.jpg" width="100%" style="max-width: 800px" alt="Table MARBLE Benchmark">
<img src="images/tab-mulan.png" width="50%" style="max-width: 400px; margin: 0 25%" alt="Table Mulan Results">

## Model Checkpoints

| Model Name | Parameters | Data | HuggingFace🤗 |
| ----------- | --- | ---  | ----------- |
| MuQ    | ~300M  | MSD dataset | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)       |
| MuQ-MuLan  | ~700M | music-text pairs | [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)       |

## Citation

Upcoming

## Acknowledgement

We borrow many codes from the following repositories:
- [lucidrains/musiclm-pytorch](https://github.com/lucidrains/musiclm-pytorch)
- [minzwon/musicfm](https://github.com/minzwon/musicfm)


Also, we are especially grateful to the awesome [MARBLE-Benchmark](https://github.com/a43992899/MARBLE-Benchmark).
