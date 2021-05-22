# Devanagari-Character-Recognition
A project based on image processing and deep learning

In this project, multiple Devanagari words( handwritten or typed), written in a paragraph, can be predicted through images. Fully connected CNN models and boosting are used for better results. 

## Instructions To Use
To predict characters in various images, one can paste the images in a folder and then can change the path in [Main.py](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Main.py) as the name of that folder.
And then run Main.py file.

## Project features
* Image processing, seperating words from paragraph ([Split Words](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Words.py)) and then characters from words([Split Characters](https://github.com/milind-prajapat/Devanagari-Character-Recognition/blob/main/Split_Characters.py)) are done using OpenCV. 
* CNN Models are trained using the dataset and then stored in individual files.
* Character prediction is done using Boosting among the models. 
* 5 CNN models have been trained and then voting among their result is done to get the most accurate result possible.

## References
* [Dataset used to train the model](https://drive.google.com/file/d/1ne6XP-Js_JK3PnatCQSJW_hCWQ4JLWkB/view)
* [Samples Words](https://github.com/milind-prajapat/Devanagari-Character-Recognition/tree/main/Words)
 
