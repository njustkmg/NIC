# Noise-Aware Image Captioning with Progressively Exploring Mismatched Words

The code for deploying the NIC framework on AoANet.

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.7.0
- cider ([link](https://github.com/ruotianluo/cider/tree/dbb3960165d86202ed3c417b412a000fc8e717f3))
- coco-caption ([link](https://github.com/ruotianluo/coco-caption/tree/dda03fc714f1fcc8e2696a8db0d469d99b881411))
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
