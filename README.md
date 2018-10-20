# boilermake-parkinsons

## Modules Used
- Keras
- Pandas
- Numpy
- Matplotlib
- Scipy
- Datetime
- Time

## Steps Competed
- Convert points into images
- Get an algorithm to work
- Current Model:
  - 32 Convolutions
  - 10x10 size
  - 40x40 pools
  - 0.2 dropout
  - 128 dense
  - sigmoid activation
  - 20 batch size
  - 50 epochs


## TODOs
- Improve Model
- Host code
- Create front end application
- Connect front end and back end
- Create MongoDB database to improve model


## Thoughts
- Treat data as image vs. time series
- Pixel value from 500x500 to 125x125
- The model initially assumed that everyone had Parkinson's, so it would have a 75% accuracy due to the makeup of our dataset.
