# PyTorch Simple Super-Resolution

## Model Reference

- [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)


# Usage

The code will automatically download the [BSDS300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), and extract it to get the training and validation data.

## Training
  
```

usage: train.py [-h] --upscale_factor UPSCALE_FACTOR --datapath DATAPATH
                [--model MODEL] [--threads THREADS] [--lr LR]
                [--nEpochs NEPOCHS] [--batchSize BATCHSIZE]
                [--testBatchSize TESTBATCHSIZE] [--isCuda ISCUDA]

Pytorch Image/Video Super-Resolution

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor UPSCALE_FACTOR
                        Super-resolution upscale factor
  --datapath DATAPATH   Path to Original data
  --model MODEL         Choose which SR model to use
  --threads THREADS     Number of thread for DataLoader
  --lr LR               Learning rate
  --nEpochs NEPOCHS     Number of epochs
  --batchSize BATCHSIZE
                        Training batch size
  --testBatchSize TESTBATCHSIZE
                        Test batch size
  --isCuda ISCUDA       Cuda Usage
  
  ```
  
### Example
```
python3 train.py --upscale_factor 2 --datapath /model/path/to/folder
```

## Inference / Super-resolution

```
usage: inference.py [-h] --input_image INPUT_IMAGE --model MODEL
                    [--output_filename OUTPUT_FILENAME] [--cuda]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --input_image INPUT_IMAGE
                        input image to use
  --model MODEL         model file to use
  --output_filename OUTPUT_FILENAME
                        where to save the output image
  --cuda                use cuda
```

### Example
```bash
python3 inference.py --input_image /path/to/img --output_filename /path/to/img --model /path/to/pretrained/model
```
