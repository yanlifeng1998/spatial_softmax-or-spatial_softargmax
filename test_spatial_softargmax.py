import numpy as np
from keras.models import Model
from keras.layers import Input
from spatial_softargmax import Spatial_SoftArgmax


def test_model(input_shape):
    inputs = Input(shape=input_shape)
    outputs = Spatial_SoftArgmax()(inputs)

    return Model(inputs, outputs)


if __name__ == '__main__':

    H = 256
    W = 256
    K = 4  # data has 4 channels

    model = test_model(input_shape=(H, W, K))
    '''
    Create the data to test the code.
    Each channel has only one pixel value of 100, the rest are 0.
    
    '''
    data = np.zeros((2, H, W, K), dtype=np.float32)  # batch size is 2
    gt1 = []
    gt2 = []
    gt = []

    for c in range(K):
        x = np.random.randint(H)
        y = np.random.randint(W)

        data[0, x, y, c] = 100.
        gt1.append([x, y])

    for c in range(K):
        x = np.random.randint(H)
        y = np.random.randint(W)

        data[1, x, y, c] = 100.
        gt2.append([x, y])

    gt.append(gt1)
    gt.append(gt2)

    # Ground truth
    gt = np.array(gt)

    # The prediction
    pred = model(data)

    # print
    print(gt)
    print(pred)
