import numpy as np

class Tensor:
    def __init__(self, data, op='', *parents):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data) #матрица данных.shape == градиент.shape 
        self.op = op
        self.parents = parents
    def __add__(self, other):
        ret = Tensor(self.data + other.data, '+', self, other)
        return ret

    def __mul__(self, other):
        ret = Tensor(self.data * other.data, '*', self, other)
        return ret 

    def backward(self):
        if self.op == "+":
            for parent in self.parents:
                parent.grad += self.grad
                parent.backward()
        if self.op == "*":
            for i in range(len(self.parents)):
                self.parents[i].grad += self.grad * self.parents[0 if i > 0 else 1].data 
                self.parents[i].backward()

a = Tensor(2)
b = Tensor(3)

c = a * b
d = c * a

d.grad = 10
d.backward()

print("a.grad =", a.grad) #dl/dd = 10; dd/da = c;  
print("b.grad =", b.grad)
print("c.grad =", c.grad)
