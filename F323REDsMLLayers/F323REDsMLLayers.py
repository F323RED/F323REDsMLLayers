import numpy as np

class AddLayer :
    def Forward(self, x, y):
        return x + y

    def Backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class MulLayer :
    def __init__(self):
        self.x = None
        self.y = None

    def Forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def Backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class ReLULayer :
    def __init__(self):
        self.xMask = None

    def Forward(self, x):
        self.xMask = (x <= 0)
        y = x.copy()
        y[self.xMask] = 0

        return y

    def Backward(self, dout):
        dout[self.xMask] = 0

        return dout

class LeakReLULayer :
    def __init__(self):
        self.xMask = None

    def Forward(self, x):
        self.xMask = (x <= 0)
        y = x.copy()
        y[self.xMask] *= 0.1

        return y

    def Backward(self, dout):
        dout[self.xMask] = 0.1

        return dout

class SigmoidLayer():
    def __init__(self):
        self.y = None

    def Forward(self, x):
        y = 1 / (1 + np.exp(x))
        self.y = y.copy()

        return y

    def Backward(self, dout):
        dx = dout * self.y * (1 - self.y)

        return dx

class AffineLayer:      # Matrix A dot B
    def __init__(self):
        self.x = None
        self.w = None

    def Forward(self, x, w):
        self.x = x.copy()
        self.w = w.copy()
        y = np.dot(x, w)

        return y

    def Backward(self, dout):
        dx = np.dot(dout, self.w.T)
        dw = np.dot(self.x.T, dout)

        return dx, dw

class SoftmaxAndLossLayer:
    def __init__(self):
        self.y = None   # Output of softmax
        self.t = None  
        self.loss = None

    def Forward(self, x, t):
        self.t = t
        self.y = Softmax(x)
        self.loss = CrossEntropyError(self.y, self.t)

        return self.loss

    def Backward(self, dout=1.0):
        batchSize = self.t.shape[0]
        dx = (self.y - self.t) / batchSize

        return dx

def Softmax(x) :
    if x.ndim == 1:
        x = x.reshape(1, x.size)
    
    result_sum_exp = []
    for i in x:
        c = np.max(i)                           # To prevent e^x overflow
        exp_a = np.exp(i - c)
        sum_exp = np.sum(exp_a)
        result_sum_exp.append(exp_a / sum_exp)  # The sum of every elements is 1.
                                
    return np.array(result_sum_exp)     # Represent chance of this answer.

def CrossEntropyError(y, t) :
    # Compatiable with batch process
    if y.ndim == 1 :
        t = t.reshape(1, t.size)        
        y = y.reshape(1, y.size)

    # Less is better.
    return -np.sum(t * np.log(y + 1e-4))  / y.shape[0]


# Debug
#layer1 = AffineLayer()
#layer2 = AddLayer()
#layer3 = SoftmaxAndLossLayer()

#x = np.array([[1.0, 2.0], [-3.0, 4.0]])
#W1 = np.array([[1.0, 1.2, 3.7], [2.0, -0.8, -1.2]])
#b1 = np.array([1.0, 2.0,-1.0])
#t = np.array([1.0, 0.0, 0.0])

#z1 = layer2.Forward(layer1.Forward(x, W1), b1)
#y = layer3.Forward(z1, t)

#dz1 = layer3.Backward()
#db1, dxw = layer2.Backward(dz1)
#dx, dw1 = layer1.Backward(dxw)

#print("y :", y)
#print("dx :", dx)
#print("dw1 :", dw1)
#print("db1 :", db1)