import keras.models
import tensorflow
from keras.models import model_from_json
import matplotlib.pyplot as plt
import sys

def LoadModel():
    json_file = open("MyModel.json","r")
    json = json_file.read()
    json_file.close()

    model = model_from_json(json)
    model.load_weights("MyModel.h5")

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    graph = tensorflow.get_default_graph()

    return model,graph

def Predict(imgPath):
    x = plt.imread(imgPath)
    x = x.reshape(1,28,28,1)


    with graph.as_default():
        out = model.predict(x)
        return out

model,graph = LoadModel()
print("--- Model Loaded ---")
print("Our Number"+sys.argv[1])
print("Prediction Weight: 0,1,2,3,4,5,6,7,8,9")
print(Predict(sys.argv[1]))