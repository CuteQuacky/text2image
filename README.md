# Text to image using GAN
Ilias Alexandropoulos, mtn2302

Vasiliki Rentoula, mtn2317
## Description

This project is an image generator that uses text descriptions and generates images based on text. 
([Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Training the model](#training-the-model)
- [Testing the model](#testing-the-model)
- [Computing inception score](#Computing-inception-score)

## Installation

```bash
git clone https://github.com/IliasAlex/text2image.git
```
## Usage
```bash
cd text2image
python3 -m venv text2image
source text2image/bin/activate
pip install -r requirements.txt
```

## Data
CUB_200_2011 [Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Also you will need to install the pre-trained text embeddings. 
You can download them from [here](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view?resourcekey=0-sZrhftoEfdvHq6MweAeCjA)

So the file structure should be:
```
/Datasets
    /birds
    /CUB
```

## Training the model
```bash
python3 main.py --data_dir Datasets/
```

`--data_dir`: Path to the datasets folder

`-test`: Path to the test features folder

## Testing the model
The following script will generate images using the trained generator along with the corresponding captions. 
Using the test embeddings.
You can download the pre-trained generators from [here](https://drive.google.com/drive/folders/1el_qwcxf0P3KA4cA0uuqaPVrrmuXBdXb?usp=drive_link)

```bash
python3 test/test-demo.py 
--embedding_path Datasets/birds/test/char-CNN-RNN-embeddings.pickle 
--class_info_path Datasets/birds/test/class_info.pickle 
--filenames_path Datasets/birds/test/filenames.pickle
--captions_dir Datasets/birds/text_c10
--model_path Models/STAGE1G_ResBlock.pth
--output_dir Output-test/
```

## Computing inception score

```bash
python3 test/test-inception.py 
--fake_image_dir Output-test/
--filenames_path Datasets/birds/test/filenames.pickle
```
