# Devanagari-Character-Recognition
A project based on image processing and deep learning
In this project, multiple Devanagari words, hand-written in a paragraph are identified. Some samples are given in file [Words]https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words

##Project features
* Image processing, seperating words from paragraph ([Split Words]https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Words.py) and then characters from words([Split Characters]https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Characters.py) are done using OpenCV. 
* Deep learning is used to identify each character in the word.
* Data set used to train the model [Dataset]https://drive.google.com/file/d/1ne6XP-Js_JK3PnatCQSJW_hCWQ4JLWkB/view
* Character recoginition is done using Boosting among CNN models. 5 CNN models have been trained and then voting among their result is done to get the most accurate result possible.
 
