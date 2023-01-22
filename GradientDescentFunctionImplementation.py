import numpy as np 
import tensorflow as tf 
from tensorflow import keras
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import math

class GradientDescent:

    def __init__(self):
        self.df = pd.read_csv("insurance_data.csv")

    def splitTrainAndTestDataSet(self):
        xTrain, xTest, yTrain, yTest = train_test_split(self.df[['age','affordibility']], self.df.bought_insurance, test_size=0.2, random_state=25)

        xTrainScaled = xTrain.copy()
        xTrainScaled['age'] = xTrainScaled['age'] / 100

        xTestScaled = xTest.copy()
        xTestScaled['age'] = xTestScaled['age'] / 100

        self.xTrainData = xTrainScaled
        self.yTrainData = yTrain
        self.xTestData = xTestScaled
        self.yTestData = yTest
        self.xTestScaled = xTestScaled
       

    def buildTensorflowModel(self):
        model = keras.Sequential([
            keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def trainModel(self, model):
        model.fit(self.xTrainData, self.yTrainData, epochs=5000)
        return model
    
    def evaluateModel(self, model):
        model.evaluate(self.xTestData, self.yTestData)
        model.predict(self.xTestScaled)
        coef, intercept = model.get_weights()
        print("cofficients : ", coef, " intercept : ", intercept)
        self.coef = coef
        self.intercept = intercept


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predictionFunction(self, age, affordibility):
        weightedSum = self.coef[0]*age + self.coef[1]*affordibility + self.intercept
        return self.sigmoid(weightedSum)

    def sigmoidNumPy(self, X):
        return 1 / (1 + np.exp(-X))

    def logLossFunction(self, yTrue, yPredicted):
        epsilon = 1e-15
        yPredictedNew = [max(i, epsilon) for i in yPredicted]
        yPredictedNew = [min(i, 1-epsilon) for i in yPredictedNew]
        yPredictedNew = np.array(yPredictedNew)
        return -np.mean(yTrue*np.log(yPredictedNew) + (1-yTrue)*np.log(1-yPredictedNew))

    def gradientDescentAlgo(self, age, affordibility, yTrue, epochs, lossThreshold):
        w1 = 1
        w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weightedSum = w1 * age + w2 * affordibility + bias
            yPredicted = self.sigmoidNumPy(weightedSum)
            loss = self.logLossFunction(yTrue, yPredicted)

            w1d = (1/n)*np.dot(np.transpose(age), (yPredicted - yTrue))
            w2d = (1/n)*np.dot(np.transpose(affordibility), (yPredicted - yTrue))

            biasD = np.mean(yPredicted - yTrue)
            w1 = w1 - rate*w1d
            w2 = w2 - rate*w2d
            bias = bias - rate*biasD 

            print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')   

            if loss <= lossThreshold:
                break

        return w1, w2, bias

    def testGradientDescent(self):
        w1, w2, intercept = self.gradientDescentAlgo(self.xTrainData['age'], self.xTrainData['affordibility'], self.yTrainData, 1000, 0.4631)
        return (w1, w2), intercept

   

    def displayDataSet(self):
        print(self.df.head())
        print("------------------------------")
        print(self.xTrainData.head())
        print("--------------------------------")
        print(self.xTestData.head())


def main():
    gradientDescent = GradientDescent()
    gradientDescent.splitTrainAndTestDataSet()
    model = gradientDescent.buildTensorflowModel()
    model = gradientDescent.trainModel(model)
    coefT, interceptT = model.get_weights()
    gradientDescent.evaluateModel(model)
    coefS, interceptS = gradientDescent.testGradientDescent()
    print("Model coefficients and bias :")
    print("w1 : ", coefT[0], " w2 : ", coefT[1], " bias : ", interceptT)
    print("Custom gradient descent coeffiecient and bias :")
    print("w1 : ", coefS[0], " w2 : ", coefS[1], " bias : ", interceptS)

if __name__ == "__main__":
    main()