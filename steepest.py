import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from scipy import io
import os
from scipy.optimize import minimize
numpy.random.seed(42)

cwd = os.getcwd()
cwd
mydata = io.loadmat('emailvalues.mat')

ytrain = mydata['ytrain'].T
xtrain = np.concatenate((mydata['Xtrain'], np.ones((3065, 1))), axis=1).T


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def prediction(w, Data):
    pred = []
    z = np.dot(w, Data)
    a = sigmoid(z)
    for i in range(0, len(a[0])):
        if (a[0][i] > 0.5):
            pred.append(1)
        elif (a[0][i] <= 0.5):
            pred.append(-1)
    return pred


def updateObj(w, X, y):
    z = np.dot(w, X)
    val = -np.multiply(y, z)
    J = np.sum(np.log(1+np.exp(val)))
    return J


def updateGrad(w, X, y):
    z = np.dot(w, X)
    val = -np.multiply(y, z)
    num = -np.multiply(y, np.exp(val))
    den = 1+np.exp(val)
    f = num/den
    gradJ = np.dot(X, f.T)
    return gradJ


flag = True
w = np.random.randn(1, 58)
X_test = xtrain
y_test = ytrain
count = 0
maxit = 2000
tol = 0.00001
y_pred = prediction(w, X_test)
grad = updateGrad(w, X_test, y_test)
obj = updateObj(w, X_test, y_test)
grad_norm = np.sum(grad**2)**0.5
while(count < maxit and flag):
    count += 1
    old_obj = obj
    a = 0.1
    while(updateObj(w-a*grad.T, X_test, y_test)-updateObj(w, X_test, y_test) >= -0.5*a*np.dot(grad.T, grad)):
        a = 0.8*a

    w -= a * grad.T

    y_pred = prediction(w, X_test)
    grad = updateGrad(w, X_test, y_test)
    obj = updateObj(w, X_test, y_test)
    grad_norm = np.sum(grad**2)**0.5
    diff = old_obj - obj
    print("Epoch", count, "Loss", obj, "Accuracy",
          accuracy_score(y_test[0], y_pred)*100, "Difference", diff, grad_norm)
    if old_obj-obj < tol:
        flag = False


#learning_rate = 1
    # while((updateObj(w - learning_rate*grad.T, X_test, y_test) - updateObj(w, X_test, y_test)) >= 0.5*learning_rate*grad):
    #learning_rate = 0.75*learning_rate


ytest = mydata['ytest'].T
n = np.shape(mydata['Xtest'])[0]
xtest = np.concatenate((mydata['Xtest'], np.ones((n, 1))), axis=1).T
Test_predict = prediction(w, xtest)
print("Test Accuracy", accuracy_score(
    ytest[0], Test_predict)*100)
