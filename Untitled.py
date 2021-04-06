import os
import cv2
import random
import shutil

vowels = os.listdir("Vowels")

if os.path.isdir('modified_vowels'):
    shutil.rmtree('modified_vowels')
os.mkdir('modified_vowels')
os.mkdir('modified_vowels\\Test')
os.mkdir('modified_vowels\\Train')

for vowel in vowels:
    if os.path.isdir(f'modified_vowels\\Test\\{vowel}'):
        shutil.rmtree(f'modified_vowels\\Test\\{vowel}')
    os.mkdir(f'modified_vowels\\Test\\{vowel}')
    if os.path.isdir(f'modified_vowels\\Train\\{vowel}'):
        shutil.rmtree(f'modified_vowels\\Train\\{vowel}')
    os.mkdir(f'modified_vowels\\Train\\{vowel}')
    
    List = os.listdir(f"{os.getcwd()}\\Vowels\\{vowel}")
    for _ in range(3):
        random.shuffle(List)

    for img_name in List[:200]:
        shutil.copy(f"Vowels\\{vowel}\\{img_name}", f"modified_vowels\\Test\\{vowel}\\{img_name}")

    for img_name in List[200:]:
        shutil.copy(f"Vowels\\{vowel}\\{img_name}", f"modified_vowels\\Train\\{vowel}\\{img_name}")
