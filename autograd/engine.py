import numpy as np

class Tensor:
    def __init__(self, data, op='', *parents):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data) #матрица данных.shape == градиент.shape 
        self.op = op
        self.parents = parents
        self.backward = lambda: None
    def __add__(self, other):
        ret = Tensor(self.data + other.data, '+', self, other)
        def _backward():
            self.grad += ret.grad  
            other.grad += ret.grad
        ret.backward = _backward
        return ret

    def __mul__(self, other):
        ret = Tensor(self.data * other.data, '*', self, other)
        def _backward():
            self.grad += ret.grad * other.data
            other.grad += ret.grad * self.data
        ret.backward = _backward
        return ret 


a = Tensor(2)
b = Tensor(3)

c = a + b
d = a * c

d.grad = 10

d.backward()

print(a.grad)
print(c.grad)
