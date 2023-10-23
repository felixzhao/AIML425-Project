# AIML425-Project
fine tuning RetinaFace pre-trained model

## My Contributions

Throughout the course of this research, I have made the following significant contributions:

- In the file `retinaface.py`, I implemented the functionality to freeze specific layers, enhancing the flexibility of the model during the fine-tuning process.
- Introduced and developed two new files: 
  - `tune_hyper.py`
  - `tune_hyper_val_widerface.py`
  
  These scripts streamline the end-to-end process for hyperparameter tuning and validation on the WIDER FACE dataset.


##Install

Plase following the [guide](https://github.com/felixzhao/AIML425-Project/blob/main/src/README.md).


## How to Run

### Train Base Model

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 --training_dataset './data/widerface/train/label.txt'

### Fine Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python tune_hyper.py --network mobile0.25 --training_dataset './data/widerface/train/label_tune.txt' --pretrained_model './weights/base_model.pth'