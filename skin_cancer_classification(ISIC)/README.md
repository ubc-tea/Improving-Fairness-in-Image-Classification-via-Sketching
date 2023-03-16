# Usage

## Skin ISIC 2018

### Setup

Run the `./skin_cancer_classification(ISIC)/requirements.txt` with the following command to meet requirements.
```
pip install -r requirements.txt
```
The Skin ISIC 2018 dataset can be downloaded [here](https://challenge.isic-archive.com/data/#2018). The images should be stored at `./skin_cancer_classification(ISIC)/data/origin/`.

### Pretained model
The pretained model can be downloaded at [here](https://drive.google.com/file/d/1TamMyz31fAV4T7zLoeHiLCG1kfz1zYk-/view?usp=sharing). The `saved_model` folder should be placed at `./saved_model/`.


### Testing
You could run `./skin_cancer_classification(ISIC)/train_vgg.py` with the following command to test the model.
```
py train_vgg.py --schema test --img_type xxx --privilege_type xxx --fairloss x --model_name xxx.pt
```
You could adjust `--img_type`, `--privilege_type`, `--fairloss`, and `--model_name` in this file when testing the model.

### Training
To train the model, you could run `./skin_cancer_classification(ISIC)/train_vgg.py` with the following command:
```
py train_vgg.py --schema train --img_type xxx --privilege_type xxx --fairloss x --batch_size x --epochs x
```
You could adjust `--img_type`, `--privilege_type`, `--batch_size`, `--epochs`, and `--fairloss` in this file when training the model.

