from keras.models import Sequential
from keras.layers import MaxPooling2D  # import the pooling module

model = Sequential()  # Set up the model
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(200, 200, 2)))
# add max pooling layer
# max pooling layer takes the max node of each convolution and returns the node in the layer
# the next layer is essentially halve the size(width & height) of the previous conv. layer
model.summary()
