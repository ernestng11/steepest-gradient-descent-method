import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from scipy import io
import os
import math
numpy.random.seed(42)

cwd = os.chdir('filepath here')
data = io.loadmat('emaildata.mat')
ytrain = data['ytrain']  # shape = (3065,1)
ytest = data['ytest']
xtrain = data['Xtrain']  # shape = (3065,57)
xtest = data['Xtest']
w = np.zeros((57, 1))  # shape = (57,1)
m = xtrain.shape[0]  # m  = 3065
p = xtrain.shape[1]  # p = 57
# w = np.random.randn(p)*np.sqrt(2/m) #shape = (57,1)


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def prediction(w, Data):
    pred = []
    z = np.dot(w.T, Data.T)
    a = sigmoid(z)
    for i in range(0, len(a[0])):
        if (a[0][i] > 0.5):
            pred.append(1)
        elif (a[0][i] <= 0.5):
            pred.append(-1)
    return pred


def updateObj(y, w, X):  # scalar
    z = np.dot(X, w)
    val = -np.multiply(y, z)
    fx = np.sum(np.log(1+np.exp(val)))
    return fx


def updateGrad(y, w, X):  # shape = (1,57)
    z = np.dot(X, w)
    val = np.multiply(y, z)
    denom = -y/(1 + np.exp(val))
    gradient = np.dot(denom.T, X)
    return gradient


w = np.random.randn(57, 1)
count = 1
maxit = 30
tol = 0.1
y_pred = prediction(w, xtest)
grad = updateGrad(ytest, w, xtest)
grad_norm = np.linalg.norm(grad)
obj = updateObj(ytest, w, xtest)
gradlist = []

while grad_norm > tol:
    #Backtracking method (Armijo's Rule)
    a = 1
    while updateObj(ytest, w-a*grad.T, xtest)-updateObj(ytest, w, xtest) >= -0.5*a*np.dot(grad, grad.T):
        a = 0.75*a

    w -= a * grad.T

    grad = updateGrad(ytest, w, xtest)

    grad_norm = np.linalg.norm(grad)
    gradlist.append(grad_norm)
    obj = updateObj(ytest, w, xtest)

    ypred = prediction(w, xtest)
    score = accuracy_score(ytest, ypred)*100
    count += 1
    print("count", count, "gradient", grad_norm, "accuracy", score)

    if grad_norm < tol:
        break

rang = range(1, count)
