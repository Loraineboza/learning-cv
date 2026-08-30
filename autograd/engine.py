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
            for parent in ret.parents:
                parent.grad += ret.grad
                parent.backward()

        ret.backward = _backward
        return ret

    def __mul__(self, other):
        ret = Tensor(self.data * other.data, '*', self, other)
        def _backward():
            for i in range(len(ret.parents)):
                ret.parents[i].grad += ret.grad * ret.parents[0 if i>0 else 1].data
                ret.parents[i].backward()

        ret.backward = _backward
        return ret 


a = Tensor(2)
b = Tensor(3)

c = a + b
d = a * c

d.grad = 10

d.backward()

print(f"a.grad =",a.grad)
print(f"b.grad =",b.grad)
print(f"c.grad =",c.grad)
