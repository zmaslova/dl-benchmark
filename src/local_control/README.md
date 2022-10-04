# Script for local run existing inference modes

## Usage:
```bash
python3 local_start.py \
    -m <sync mode> \
    inference_args \
    "-m <model path>" "-w <model weights>" ...
```
## Example:
```bash
python3 local_start.py \
    -m async \
    inference_args \
    "-m resnet-50-pytorch.xml" "-w resnet-50-pytorch.bin" "-r 1" "-b 1" "-i cat_224x224.jpg"
```

All the arguments that need to be passed after `inference_args` can be viewed in the corresponding inference script.
(Inference scripts located in `src/inference` folder)