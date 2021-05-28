import os
import shutil
import random
import pandas as pd

from tqdm import tqdm

random.seed(0)

df = pd.read_csv(os.path.join('Dataset', 'Reference.csv'))

if os.path.isdir('Split Dataset'):
    shutil.rmtree('Split Dataset')

os.mkdir('Split Dataset')
os.mkdir(os.path.join('Split Dataset', 'Train'))
os.mkdir(os.path.join('Split Dataset', 'Validation'))

for Class in df['Class']:
    os.mkdir(os.path.join('Split Dataset', 'Train', str(Class)))
    os.mkdir(os.path.join('Split Dataset', 'Validation', str(Class)))

for Class in tqdm(df.iloc[:,0], unit_scale = True, miniters = 1, desc = 'Splitting Dataset '):
    Files = os.listdir(os.path.join('Dataset', str(Class)))
    random.shuffle(Files)

    if len(Files) == 2000:
        for File in Files[:200]:
            shutil.copy(os.path.join('Dataset', str(Class), File), os.path.join('Split Dataset', 'Validation', str(Class), File))
        for File in Files[200:]:
            shutil.copy(os.path.join('Dataset', str(Class), File), os.path.join('Split Dataset', 'Train', str(Class), File))
    elif len(Files) == 1250:
        for File in Files[:125]:
            shutil.copy(os.path.join('Dataset', str(Class), File), os.path.join('Split Dataset', 'Validation', str(Class), File))
        for File in Files[125:]:
            shutil.copy(os.path.join('Dataset', str(Class), File), os.path.join('Split Dataset', 'Train', str(Class), File))

shutil.copy(os.path.join('Dataset', 'Reference.csv'), os.path.join('Split Dataset', 'Reference.csv'))