from keras.models import Sequential
from keras.layers import Conv2D

# Formula: No. of Parameters in a Conv. layer
# depends on the following values:
# K - number of filters in the Conv. layer = filters
# F - the height and width of the Conv. layer = kernel_size
# D_pre - the depth of the previous layer = last value in input shape tuple
###
# We know that there are F * F F D_pre weights per filter
# Therefore the total number of weights in a Conv. layer is K * F * F * D_pre(without biases)
# With biases being taken into consideration, the total number of weights becomes
# K * F * F * D_pre + K - since there is one bias per filter,
# this is results in the number of parameters in a Conv. layer

# Formula: Shape of a Conv.layer
# depends on the following values:
# K - number of filters in the Conv. layer = filters
# F - the height and width of the Conv. layer = kernel_size
# S - the stride of the convolution = strides
# H_pre - the height of the previous layer = first value in input shape tuple
# W_pre - the width of the previous layer = second value in input shape tuple
# depth of Conv. layer is always be equal to K - number of filters
###
# If padding= 'same' then the spatial dimensions of the conv. layer
# are calculated as follows:
# height = ceil(float(H_pre) / float(S))
# width = ceil(float(W_pre) / float(S))
##
# If padding = 'valid' then the spatial spatial dimensions of the conv. layer
# are calculated as follows:
# height = ceil(float(H_pre - F + 1) / float(S))
# width = ceil(float(W_pre - F + 1) / float(S))

# Set up Sequential model
model = Sequential()
# add Conv. layer
model.add(Conv2D(filters=16, kernel_size=2,strides=2,
                 padding='valid', activation='relu', input_shape=(200, 200, 1)))

# print the summary of the model
model.summary()
