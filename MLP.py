import numpy as np
import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class MLP():
  trainX = [] #training input
  trainY = [] #training output
  testX = [] #testing input
  testY = [] #testing output
  Ws = [] #weights
  Zs = [] #hidden layers
  AV #activation function

  def __init__ (self, layerNumber):
    transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #getting the x(2d matrix of 50000*3072, input images), and y(1*3072, output label);
    tmpTrain = trainloader.dataset.data # shape of 50000*32*32*3 need to transform to 50000*3072
    x,y1,y2,y3 = tmpTrain.shape
    self.trainX = np.zeros([x, y1*y2*y3])
    tmpTrainY = trainloader.dataset.targets
    self.trainY = np.zeros([len(tmpTrainY), len(classes)])
    for i in range(len(tmpTrainY)):
      self.trainY[i][tmpTrainY[i]] = 1
    tmpTest = testloader.dataset.data # shape of 10000*32*32*3 need to transform to 10000*3072
    xx, y4, y5, y6 = tmpTest.shape
    self.testY = testloader.dataset.targets
    self.testX = np.zeros([xx, y4*y5*y6])
    for i in range(x):
      self.trainX[i] = tmpTrain[i].flatten()

    self.trainX = self.trainX / np.linalg.norm(self.trainX) 
    for j in range(xx):
      self.testX[j] = tmpTest[j].flatten()

    self.testX = self.testX / np.linalg.norm(self.testX)

    print("the shape of matrix trainX" + str(self.trainX.shape))
    print("the size of trainY " + str(self.trainY.shape))
    print("the shape of matrix testX" + str(self.testX.shape))
    print("the size of testY " + str(len(self.testY)))

    #fill Ws with initial random weight
    #define weights for different layers, put them in a list called Ws, each weight has a shape of size(l-1)*size(l)
    N,D = self.trainX.shape
    start = D
    N,DD =self.trainY.shape
    end = layerNumber[0]
    w = np.random.randn(start, end)*0.1
    self.Ws.append(w)
    for i in range(len(layerNumber)):
      start = layerNumber[i]
      if i+1 == len(layerNumber):
        end = DD
      else:
        end = layerNumber[i+1]
      w = np.random.randn(start, end)*0.1
      print(w)
      self.Ws.append(w)

  #activation functions

  def softmax(self,x):
    N,D = x.shape
    x_exp = np.exp(x - np.max(x, 1)[:, None])
    return x_exp / np.sum(x_exp, axis=-1)[:, None]

  def relu(self,x):
    N, D = x.shape
    ans = np.zeros([N,D])
    for i in range(N):
      for j in range(D):
        if x[i][j] > 0:
          ans[i][j] = x[i][j]
        else:
          ans[i][j] = 0
    return ans

  def sigmoid(self,x):
    N, D = x.shape
    ans = np.zeros([N,D])
    for i in range(N):
      for j in range(D):
        try:
          ans[i][j] = 1/(1+math.exp(-x[i][j]))
        except OverflowError:
          if(x[i][j])>0:
            ans[i][j] = 1
          else:
            ans[i][j] = 0
    return ans

  def getYhead(self,X):
    Y = []
    N,D = X.shape
    self.Zs.clear()
    for layer in layerNumber:
      self.Zs.append(np.zeros([N,layer]))
    Yhead = np.zeros([N, len(classes)])
    tmp = X
    for i in range(len(self.Ws)-1):
      NN,DD = tmp.shape
      tmp = self.relu(np.dot(tmp, self.Ws[i])) #Doing activation function for the Zs in layer
      self.Zs[i] = tmp
    Y = self.softmax(np.dot(tmp, self.Ws[len(self.Ws)-1])) #Doing softmax for final layer since it is multiclass classification
    return Y

  #Cost of one training result
  def cost(self,Y1, Y2):
    total = 0.0
    for i in range(len(Y1)):
      value = (Y1[i]-Y2[i])*(Y1[i]-Y2[i])
      total = total + value
    return total

  #Average cost of all training result
  def averageCost(self,Y1, Y2):
    N,D = Y1.shape
    total = 0.0
    for i in range(N):
      total = total + self.cost(Y1[i],Y2[i])
    return total/N

  def gradient(self,X, Y):
    N,D = X.shape
    Yh = self.getYhead(X)
    print("Average cost so far " + str(self.averageCost(Yh, Y)))
    dY = (Yh - Y)
    dZnext =dY
    WsReversed = self.Ws.copy()
    ZsReversed = self.Zs.copy()
    WsReversed.reverse() #reverse them since we need to calculate the dw from last layer to the first layer
    ZsReversed.reverse()
    dWs =[]
    w = WsReversed[0]
    z = ZsReversed[0]
    dW = np.dot(z.T, dZnext)
    #Probably the problem come from here, I am trying to calculate the dWeight between last hidden layer and final layer
    #now z is the last hidden layer, Yh is the final answer, in the video, derivative of w = al-1*(derivative of sigmoid)*dy
    #al-1 is last hidden layer, Yh*(1-Yh) is the derivative of sigmoid, and dzNext is dy, however if Yh is all zero them dW
    #would be zero and if will keeps like that, and I was like what???
    dZnext = np.dot(dZnext, w.T)
    dWs.append(dW)
    for i in range(1, len(WsReversed)):
      w = WsReversed[i]
      zPrevious = ZsReversed[i-1]
      if i >= len(ZsReversed):
        z = X.copy()
      else:
        z = ZsReversed[i]
      Zn, Zd = zPrevious.shape
      tmp = np.zeros([Zn,Zd])
      for i in range(Zn):
        for j in range(Zd):
          if zPrevious[i][j]>0:
            tmp[i][j] = 1
      dW = np.dot(z.T, dZnext*tmp)
      dZnext = np.dot(dZnext*tmp, w.T)
      dWs.append(dW)
    dWs.reverse()
    return dWs

  def fit(self, lr, decay, eps, maxiterations, bsize, beta):
    N,D = self.trianX.shape
    dW = np.inf*np.ones_like(self.Ws[len(Ws)-1])
    dws = []
    for w in self.Ws:
      dw = np.zeros(w.shape)
      dws.append(dw)
    iter = 0
    while np.linalg.norm(dW) > eps and iter<maxiterations:
      minibatch = np.random.randint(N, size=(bsize))
      g = self.gradient(self.trainX[minibatch,:], self.trainY[minibatch,:])
      for i in range(len(self.Ws)):
        dws[i] = (1-beta)*g[i]+beta*dws[i]
        self.Ws[i] = self.Ws[i]-lr*dws[i]
      dW = g[len(self.Ws)-1]
      lr *= (1. / (1. + decay * iter))
      #print(lr)
      iter = iter+1

  def getBiggestY(self, Y):
    print(Y) 
    N,D = Y.shape
    result = np.zeros([N])
    for i in range(N):
      biggest = 0
      biggestNum = 0
      for j in range(D):
        if Y[i][j]>biggest:
          biggestNum = j
          biggest = Y[i][j]
        result[i] = biggestNum
    print(result)
    return result

  def predict(self):
    print("predicting ...")
    yHead = self.getYhead(self.testX)
    yResult = getBiggestY(self.yHead)
    totalRight = 0.0
    for i in range(len(self.testY)):
      if(yResult[i] == self.testY[i]):
        totalRight = totalRight+1
    return totalRight/len(self.testY)




if __name__ == '__main__':
  layerNumber = [1024,500, 250, 100, 50] #number of hidden layers and number of nodes in each layer
  theMlP = MLP(layerNumber)
  theMLP.fit(0.05 ,0, 0.00001, 10000, 500, .99)
  print(theMLP.predict())