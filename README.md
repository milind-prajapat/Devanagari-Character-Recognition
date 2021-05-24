# Devanagari-Character-Recognition
Devanagari Character Recognition Using Image Processing And Deep Learning

This work allows optical character recognition of Devanagari script, handwritten or printed, written in a paragraph or a line. Image processing techniques enable word and character segmentation, i.e., splitting words from a paragraph, separating characters from words. The segmented characters are then recognized using a convolutional neural network combined with a deep neural network.

We have also used boosting technique over the neural network for achieving more remarkable outcomes. We trained five neural networks with different number and type of layers, filters and pool size in the convolutional layer, dropout rate and the number of neurons in the dense layer. Then, while performing recognition, each neural network is first used for prediction separately, and then voting among them is performed. One which occurred more frequently is chosen as the final prediction, resulting in better precision.

Sample images used for character recognition can be found in [Words](https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words) directory of the repository.

## Instructions To Use
To perform character recognition, accumulate the images in a directory and then provide the complete path to the directory in [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).

You can then either run the code directly on visual studio using [Devanagari-Character-Recognition.sln](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Devanagari-Character-Recognition.sln) or can run [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).

## Project Structure
* [Split_Words.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Words.py) is used for performing word segmentation, helpful in achieving character recognition from a paragraph
* [Split_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Characters.py) is used for performing character segmentation, helpful in achieving character recognition from a word
* [Predict_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Predict_Characters.py) is used for taking the prediction of a character using boosting technique
* [Split_Dataset.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Dataset.py), [Model_1.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_1.py), [Model_2.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_2.py), [Model_3.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_3.py), [Model_4.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_4.py), [Model_5.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_5.py), [Evaluate_Model.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Evaluate_Model.py) is used for splitting dataset, training models and evaluating their performance using boosting technique

## Character Extraction
**Gif 1.** Word Segmentation 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/64096036/119259092-2efce880-bbea-11eb-942d-b77ed8810993.gif)


**Gif 2.** Character Segmentation

![ezgif-7-c210707f69b3](https://user-images.githubusercontent.com/64096036/119255956-ed654100-bbdb-11eb-88cf-caa7ac835b59.gif)

## Project Features
1. **Word segmentation** enables character recognition of paragraphs, it also preserves the order
2. **Character segmentation** enables character recognition of word
3. **Data augmentation** using image data generator class
4. **Convolution neural network** combined with a deep neural network
5. **Boosting technique** resulting in more reliable efficiency

## Limitations
1. *vyanjans* having a *matra* cannot be segmented
2. Numerals and *matra* are not included in the dataset and hence are not supported
3. Words with excess noise like [this](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Words/8.jpg) or highly slanted words might hinder efficiency

## References
* [Handwritten Devanagari Characters](https://drive.google.com/file/d/1kVn8-Cf1RnnePqfxpCnLSt1rxm2eSfh4/view?usp=sharing)
