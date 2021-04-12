import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import Split_Words
import Split_Characters
import Predict_Characters

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: 'अ', 37: 'आ', 38: 'इ', 39: 'ई', 40: 'उ', 41: 'ऊ', 42: 'ऋ', 43: 'ए', 44: 'ऐ', 45: 'ओ', 46: 'औ', 47: 'अं', 48: 'अ:'}

def predict(Image):
    Words = Split_Words.Split(Image)
    Word_Characters = Split_Characters.Split(Words)
    Predictions = Predict_Characters.Predict(Word_Characters)

    Answers = []
    for Prediction in Predictions:
        Answers.append([Label_Dict[class_id] for class_id in Prediction])

    return Answers # A list of list    

def test():
    image_paths = ['Words/1.png',
                   'Words/2.png',
                   'Words/3.png',
                   'Words/4.jpeg',
                   'Words/5.jpeg',
                   'Words/6.jpeg']
    
    correct_answers = [[['क','म','ल']],
                       [['प','त','ल']],
                       [['र','व','न']],
                       [['क','म','ल'], ['प','त','ल'], ['र','व','न']],
                       [['क','म','ल'], ['फ','स','ल'], ['म','ह','ल'], ['च','म','क'], ['ल','प','क'], ['प','ट','क'], ['न','ह','र'], ['प','ह','र'], ['ल','ह','र']],
                       [['आ','म'], ['ग','म'], ['औ','र'], ['अं','म'], ['अ:','म']]]
    score = 0
    multiplication_factor = 2 # depends on character set size

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answers = predict(image) # a list of list is returned

        for word_num, answer in enumerate(answers):
            print(''.join(answer))# will be the output string

            n=0
            for j in range(len(answer)):
                if correct_answers[i][word_num][j] == answer[j]:
                    n+=1
                
            if(n == len(correct_answers[i][word_num])):
                score += len(correct_answers[i][word_num]) * multiplication_factor

            else:
                score += n*2
    
    print('The final score of the participant is', score)

if __name__ == "__main__":
    test()
