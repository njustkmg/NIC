# Noise-Aware Image Captioning with Progressively Exploring Mismatched Words

The code for deploying the NIC framework on AoANet.

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.7.0
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)
- tensorboardX


## Training

### Prepare data

You should preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
### Start training

```bash
$ sh train.sh
```

## Acknowledgements

This repository is based on [AoANet](https://github.com/husthuaan/AoANet) , and you may refer to it for more details about the code.
