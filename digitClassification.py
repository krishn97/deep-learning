import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import seaborn as sn

class DigitClassification:

    def __init__(self):
        (xTrain, yTrain) , (xTest, yTest) = keras.datasets.mnist.load_data()
        xTrain = xTrain / 255
        xTest = xTest / 255
        self.XtrainFlaten = xTrain.reshape(len(xTrain), 28*28)
        self.XtestFlaten = xTest.reshape(len(xTest), 28*28)
        self.yTrain = yTrain
        self.yTest = yTest

    def buildSingleLayerModel(self):
        model = keras.Sequential([
              keras.layers.Dense(700, input_shape=(784,), activation='sigmoid'),
               keras.layers.Dense(600, activation='sigmoid'),
               keras.layers.Dense(500, activation='sigmoid'),
               keras.layers.Dense(400, activation='sigmoid'),
               keras.layers.Dense(300, activation='sigmoid'),
               keras.layers.Dense(150, activation='sigmoid'),
               keras.layers.Dense(100, activation='sigmoid'),
               keras.layers.Dense(50, activation='sigmoid'),
               keras.layers.Dense(30, activation='sigmoid'),
               keras.layers.Dense(10, activation='sigmoid')
           ])

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
        print(self.XtrainFlaten[0])
        model.fit(self.XtrainFlaten, self.yTrain, epochs=5)

        return model

    def build10LayerModel(self):
        model = keras.Sequential([
               keras.layers.Dense(200, input_shape=(784,), activation='relu'),
               keras.layers.Dense(100, activation='relu'),
               keras.layers.Dense(10, activation='sigmoid')  
           ])

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
        print(self.XtrainFlaten[0])
        model.fit(self.XtrainFlaten, self.yTrain, epochs=5)

        return model

    def evaluateAndPredict(self, model):
        model.evaluate(self.XtestFlaten, self.yTest)
        yPredicted = model.predict(self.XtestFlaten)
        print(np.argmax(yPredicted[0]))
        yPrefictedLabels = [np.argmax(i) for i in yPredicted]
        cm = tf.math.confusion_matrix(labels=self.yTest,predictions=yPrefictedLabels)
        plt.figure(figsize = (10,7))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

def main():
    digitClassification = DigitClassification()
    model = digitClassification.build10LayerModel()
    digitClassification.evaluateAndPredict(model)

if __name__ == "__main__":
    main()