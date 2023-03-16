# Usage

## CelebA
### Setup

Run the `./face_image_classification(CelebA)/requirements.txt` with the following command to meet requirements.
```
pip install -r requirements.txt
```

The CelebA dataset can be downloaded [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The images should be stored at `./face_image_classification(CelebA)/dataset/img_align_celeba/`

Run `./face_image_classification(CelebA)/train_test_split.py` with the following command:
```
py train_test_split.py --img_type xxx --sensitive_type xxx --num xxx
```
You could adjust `--csv_dir`, `--data_dir`, `--img_dir`, `--img_type`, `--sensitive_type`, and `--num` in this file.

Run `./face_image_classification(CelebA)/Grey.py` with the following command could generate grey scale images after train test split.
```
py Grey.py --sensitive_type xxx
```
You could adjust `--sensitive_type` in this file.


### Pretained model
The pretained model can be downloaded at [here](https://drive.google.com/file/d/1z6suPVTeVDL0ui7UVpchePRbjMTDpSU0/view?usp=sharing).

The structure of the `./face_image_classification(CelebA)/dataset/` folder should be looked like this after preprocessing.

![avatar](../img/CelebA_structure.png)

### Testing
The test results are available at `./face_image_classification(CelebA)/Tesing_results_of_CelebA.ipynb`. You could directly view and run the test results using the existing model.

In addition, you could run `./face_image_classification(CelebA)/train_resnet.py` with the following command to test the model.
```
py train_resnet.py --isTrain 0 --target xxx --img_type xxx --sensitive_type xxx --fairloss x
```
You could adjust `--target`, `--img_type`, `--sensitive_type`, and `--fairloss` in this file when testing the model.


### Training
To train the model, you could run `./face_image_classification(CelebA)/train_resnet.py` with the following command:
```
py train_resnet.py --target xxx --img_type xxx --sensitive_type xxx --fairloss x --batch_size x --max_epochs_stop x --num_epochs x --learning_rate x
```
You could adjust `--target`, `--img_type`, `--sensitive_type`, `--batch_size`, `--max_epochs_stop`, `--num_epochs`, `--fairloss`, and `--learning_rate` in this file when training the model.
