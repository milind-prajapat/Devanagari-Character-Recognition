# Devanagari-Character-Recognition
Devanagari character recognition using image processing and deep learning

This work allows optical character recognition of Devanagari script, handwritten or printed, written in a paragraph or a line. Image processing techniques enable word and character segmentation, i.e., splitting words from a section, separating characters from words. The segmented characters are then detected using a convolutional neural network combined with a deep neural network.

We have also used boosting technique over the neural network for achieving more remarkable outcomes. We trained five neural networks with different number and type of layers, filters and pool size in the convolutional layer, dropout rate and the number of neurons in the dense layer. Then, while performing recognition, each neural network is first used for prediction separately, and then a vote is taken. One which occurred more frequently is chosen as the final prediction, resulting in better precision.

Sample images used for character recognition can be found in [Words](https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words) directory of the repository.

## Instructions To Use
To perform character recognition, accumulate the images in a directory and then provide the complete path to the directory in [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).
You can then either run the code directly on visual studio using [Devanagari-Character-Recognition.sln](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Devanagari-Character-Recognition.sln) or can run [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py).

## Project Structure
* [Split_Words.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Words.py) is used for performing word segmentation, helpful in achieving character recognition from a paragraph.
* [Split_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Characters.py) is used for performing character segmentation, helpful in achieving character recognition from a word.
* [Predict_Characters.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Predict_Characters.py) is used for taking the prediction of a character.
* [Split_Dataset.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Dataset.py), [Model_1.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_1.py), [Model_2.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_2.py), [Model_3.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_3.py), [Model_4.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_4.py), [Model_5.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Model_5.py), [Evaluate_Model.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Evaluate_Model.py) is used for splitting dataset, train models and to evaluate their performance using boosting technique.

## Project Features
1. Word segmentation enables character recognition of paragraphs
2. Character segmentation enables character recognition of word
3. Data augmentation using image data generator class
4. Convolution neural network along with deep neural network
5. Boosting technique resulting in more reliable efficiency

## References
* [Dataset](https://drive.google.com/file/d/1ne6XP-Js_JK3PnatCQSJW_hCWQ4JLWkB/view?usp=sharing)
