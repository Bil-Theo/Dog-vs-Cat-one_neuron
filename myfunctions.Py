import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

def initialisation(X):
    #Cette fonction iniatilise les variables w et b avec des valeurs aleatoires
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return (W,b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1/(1 + np.exp(-Z))
    return A

def cost_function(A, y):
    epsilon = 1e-15
    return 1/len(y)*np.sum(-y*np.log(A + epsilon) - (1 - y)*np.log(1 - A + epsilon))

def gradient(A, X, y):
    Lw = (1/len(y))*( X.T.dot(A - y))
    Lb = (1/len(y))*(np.sum(A-y))
    return (Lw, Lb)

def udpate(Lw, Lb, W, b, rate_learning):
    W = W - rate_learning*Lw
    b = b - rate_learning*Lb
    return (W,b)

def artificial_neuron(X, Y, x, y, nbr_iteration = 1000, rate_learning = 0.01):
    
    W, b = initialisation(X)
    Lg_train = []
    acc_train = []
    Lg_test = []
    acc_test = []
    Rate = 0
    for i in tqdm(range(nbr_iteration)):
        A = model(X, W, b)
        os.system('cls')

        if i % 10 ==0:
            #for train
            Y_pred = prediction(X, W, b)
            Rate = accuracy_score(Y, Y_pred)
            acc_train.append(Rate)
            Lg_train.append(cost_function(A, Y))

            #for test
            A_test = model(x, W, b)
            y_pred = prediction(x, W, b)
            rate = accuracy_score(y, y_pred)
            acc_test.append(rate)
            Lg_test.append(cost_function(A_test, y))

        Lw, Lb = gradient(A, X, Y)  
        W, b = udpate(Lw, Lb, W, b, rate_learning)
    
    
    print(Rate)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Lg_train, label = 'Train Loss')
    plt.plot(Lg_test, label = 'Test Loss')   
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(acc_train, label='Train Acc')
    plt.plot(acc_test, label = 'Test Acc')
    plt.legend()
    plt.show()
    return (W, b)


def prediction(X, W, b):
    A = model(X, W, b)
    return A>=0.5

def save_model(W, b):
    
    
    #for i in range(0, len(W)):
        #S = S + repr(W[i][0])
        #S = S +"\t"
        #S.append(float(W[i][0]))

    data_model = {'coef_neuron': W, 'biais': b}

    with open('deep_model.data', 'wb') as file:
        myPickler  = pickle.Pickler(file)
        myPickler.dump(data_model)
        file.close()
