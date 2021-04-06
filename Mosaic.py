import cv2
import numpy as np
from keras.models import load_model

model = load_model("best_val_loss.hdf5")

Labels = ["ka", "kha", "ga", "gha", "kna",
          "cha", "chha", "ja", "jha", "yna",
          "taamatar", "thaa", "daa", "dhaa", "adna",
          "tabala", "tha", "da", "dha", "na",
          "pa", "pha", "ba", "bha", "ma",
          "yaw", "ra", "la", "waw", "motosaw",
          "petchiryakha", "patalosaw", "ha", "chhya", "tra", "gya",
          "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

file = ""

img = cv2.imread(file)
img = cv2.resize(img, (32,32), interpolation = "nearest")

x = img.reshape(-1, 32, 32, 1) / 255.0

print(Label[np.argmax(model.predict([x]))])