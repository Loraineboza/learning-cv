import numpy as np 
from graphviz import Digraph 
from F import MSE, Binary_cross_entropy 
def debug_op(str): 
    print(f"operation is {str}") 

class Box: 
    def __init__(self, data=np.ones(1,), parents=(), op='', name='None'):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self.parents = parents
        self._backward = lambda: None
        self.name = name
        self.op = op
    def __matmul__(self, other):
        if self.data.ndim < 2 or other.data.ndim < 2 and self.data.shape[1] != other.data.shape[0]:
            raise MatricesError("матрицы не совместимы для __matmul__ операции")
        ch = Box(self.data @ other.data, (self, other))
        
        def backward():
            self.data.grad += ch.grad @ other.data
            self.other.grad += self.data @ ch.grad 

        ch._backward = backward
        return ch
    
    def __rmatmul__(self, other):
        return other @ self

    def __add__(self, other):
        ch = Box(self.data + other.data, (self, other), '+')

        def backward():
            self.data.grad += ch.grad +1; other.data.grad += ch.grad +1

        ch._backward = backward()
        return ch

    def __sub__(self, other):
        ch = Box(self.data - other.data, (self, other), '-')

        def backward():
            print("backward of sub")
            self.data.grad += ch.grad -1; other.data.grad += ch.grad -1

        ch._backward = backward

        return ch
    def __neg__(self):
        return Box(-self.data, name=self.name)
    
    def __repr__(self):
        return f"{self.name}:\ndata = \n{self.data}\ngrad = \n{self.grad}\n\n"

    def backward(self):
        for parent in self.parents:
            print(parent)  
            parent._backward()
            parent.backward()
x = Box([3], name='x')
w = Box([2], name='w')
own = x - w
own.backward()
