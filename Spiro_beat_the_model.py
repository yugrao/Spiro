from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import random as r
import pickle as p
import numpy as np
from PIL import Image


model = load_model('Spiral_Model-84.38-1020053228.h5')
data = p.load(open("train_model/picarray.pickle", "rb"))
repeats = int(input("How many tests? "))
randpics=[]
for repeat in range(1,repeats+1):
    randpics.append(data[r.randint(0, len(data)-1)])
predictions = model.predict(np.array([pic[0] for pic in randpics]))
#randpics = [[[value*255 for value in row] for row in randpic[0]] for randpic in randpics]]
guesses=[]
guesscorrect = 0

for i in range(len(randpics)):
    print ("\nTEST",i+1,"/",repeats)
    img = Image.fromarray(np.uint8(randpics[i][0][0]*255), mode="L")#(randpics[0][0])*255)
    img.show()
    guess = True if int(input("Parkinsons? ")) else False
    if guess==bool(randpics[i][1]):
        guesscorrect+=1
    guesses.append(guess)
    print ("The answer was",bool(randpics[i][1]))
    print ("The GUESS was",guess)
    boolp = True if round(predictions[i][0]) else False
    print ("The PREDICTION was",boolp,"(",predictions[i][0],")")
    print("\n----")

print("*****************")
print("*****************")
print("Your Accuracy:",round(guesscorrect/repeats*100,2),"%")
print("Model's Accuracy:",round((sum([round(predictions[i][0])==randpics[i][1] for i in range(repeats)])/repeats*100),2),"%")
print("*****************")
print("*****************")
