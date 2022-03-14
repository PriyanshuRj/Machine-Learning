import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from alexnet import AlexNet

if __name__ == "__main__":
    model = AlexNet(input_shape = (224, 224, 3), classes = 100)
    model.summary()