# PyTorch Simple Super-Resolution

## Model Reference

- [Residual Dense Network for Image Super-Resolution] https://arxiv.org/abs/1802.08797


# Usage

The code will automatically download the [BSDS300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), and extract it to get the training and validation data.

## Training
  
```bash

python3 train.py --upscale_factor 2 --datapath /model/path/to/folder

```

## Inference / Super-resolution

```bash
python3 inference.py --input_image /path/to/img --output_filename /path/to/img --model /path/to/pretrained/model
``
