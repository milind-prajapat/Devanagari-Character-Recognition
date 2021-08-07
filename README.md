# Devanagari-Character-Recognition
Devanagari Character Recognition Using Image Processing And Deep Learning

This work allows optical character recognition of the Devanagari script, handwritten or printed, written in a paragraph or a line. Image processing techniques enable word and character segmentation, i.e., splitting words from paragraphs and separating characters from a word. The segmented characters are then recognized using a convolutional neural network combined with a deep neural network.

We have also used boosting technique over the neural network for achieving more significant outcomes. We trained five neural networks with different number and type of layers, filters and pool size in the convolutional layer, dropout rate and the number of neurons in the dense layer. Then, while performing recognition, each neural network is first used for prediction separately, and then voting among them is performed. The one which occurred more frequently is chosen as the final prediction, resulting in better precision.

Sample images used for character recognition can be found in the [Words](https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words) directory of the repository.

## Instructions To Use
To perform character recognition, accumulate the images in a directory and then provide the complete path to the directory in [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).

You can then either run the code directly on the visual studio using [Devanagari-Character-Recognition.sln](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Devanagari-Character-Recognition.sln) or can run [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).

## Structure
* [Split_Words.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Words.py) is used for performing word segmentation, helpful in achieving character recognition of paragraphs
* [Split_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Characters.py) is used for performing character segmentation, helpful in achieving character recognition of a word
* [Predict_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Predict_Characters.py) is used for taking the prediction of a character using the boosting technique
* [Split_Dataset.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Dataset.py) is used for splitting the dataset into training and validation sets
* [Model_1.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_1.py), [Model_2.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_2.py), [Model_3.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_3.py), [Model_4.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_4.py), [Model_5.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_5.py) are used for training different convolution neural networks
* [Evaluate_Model.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Evaluate_Model.py) is used for evaluating and comparing the performance of different convolution neural networks

## Character Extraction
**Gif 01.** Word Segmentation 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/64096036/119259092-2efce880-bbea-11eb-942d-b77ed8810993.gif)

**Gif 02.** Character Segmentation

![ezgif-7-c210707f69b3](https://user-images.githubusercontent.com/64096036/119255956-ed654100-bbdb-11eb-88cf-caa7ac835b59.gif)

## Model Performance

**Table 01.** Classification Report on Validation Data 

|  | accuracy_score | precision_score | recall_score | f1_score|
| --- | :---: | :---: | :---: | ---: |
| Model_1    |      0.9856     |      0.9858     |   0.9856  |  0.9856 |
| Model_2    |      0.9888     |      0.9889     |   0.9888  |  0.9888 |
| Model_3    |       0.9889    |       0.9890    |    0.9889 |   0.9889 |
| Model_4    |      0.9892     |      0.9894     |   0.9892   | 0.9892 |
| Model_5    |       0.9836    |       0.9838    |    0.9836  |  0.9836 |
| Boosting   |       0.9932    |       0.9933    |    0.9932  |  0.9932 |

**Table 02.** Classification Report on [Sample Images](https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words)

|  | accuracy_score | precision_score | recall_score | f1_score|
| --- | :---: | :---: | :---: | ---: |
Model_1      |     0.7857     |      0.8733   |     0.7857 |   0.8153
Model_2      |    0.8333      |     0.9415   |     0.8333  |  0.8701
Model_3      |     0.7262      |     0.8333   |     0.7262  |  0.7485
Model_4      |     0.7857     |      0.8175    |    0.7857  |  0.7895
Model_5      |     0.8095     |      0.9200    |    0.8095  |  0.8386
Boosting     |     0.8571      |     0.9444    |    0.8571  |  0.8862

## Features
1. **Word segmentation** enables the character recognition of paragraphs, preserving the order of the words
2. **Character segmentation** enables the character recognition of a word
3. **Image processing** enables the segmentation of slanted words and characters, and *svar* having a *matra*
4. **Data augmentation** using image data generator class, rotated, shifted, sheared and zoomed
5. **Convolution neural network** combined with a deep neural network
6. **Boosting technique** resulting in much more reliable efficiency

## Limitations
1. *vyanjans* having a *matra* cannot be segmented
2. Numerals and *matra* are not included in the dataset and hence can not be determined
3. Words with excess noise like [this](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Words/8.jpg) and highly slanted words might hinder efficiency

## References
* [Handwritten Devanagari Characters](https://drive.google.com/file/d/1kVn8-Cf1RnnePqfxpCnLSt1rxm2eSfh4/view?usp=sharing)
