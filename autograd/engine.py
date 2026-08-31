import numpy as np

class Tensor:
    def __init__(self, data, op='', *parents):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data) #матрица данных.shape == градиент.shape 
        self.op = op
        self.parents = parents
        self._backward = lambda: None
    

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ret = Tensor(self.data + other.data, '+', self, other)

        def backward():
            for parent in ret.parents:
                parent.grad += ret.grad

        ret._backward = backward
        return ret

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ret = Tensor(self.data * other.data, '*', self, other)

        def backward():
            for i in range(len(ret.parents)):
                ret.parents[i].grad += ret.grad * ret.parents[0 if i>0 else 1].data

        ret._backward = backward
        return ret
    
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ret = Tensor(np.matmul(self.data, other.data), '@', self, other)

        def backward():
            for i in range(len(ret.parents)):
                ret.parents[i].grad += ret.grad * ret.parents[1].data.T if i==0 else ret.parents[0].data.T * ret.grad

        ret._backward = backward
        return ret 


    def __pow__(self, other):
        ret = Tensor(self.data ** other, '**', self)

        def backward():
            self.grad += ret.grad * other * (self.data ** (other - 1))

        ret._backward = backward 
        return ret 

    def backward(self):
        graph = build_graph(self)
        self.grad = np.ones_like(self.data)
        for tensor in graph[::-1]: 
            tensor._backward()
    
    def __truediv__(self, other):
        return self * other**-1 #self/other
    def __rtruediv__(self, other):
        return other * self**-1

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1 #-self
    def __rmul__(self, other):
        return self * other

def build_graph(tensor, graph=None): #постройка графа: поиск всех зависимых от <tensor> объектов, которые идут в последовательном неповторяющийся порядке
    if graph is None:
        graph = []
    if tensor not in graph:
        for parent in tensor.parents:
            build_graph(parent, graph)
        graph.append(tensor)
    return graph

X = Tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

W = Tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

Y = X @ W

Y.grad = np.ones_like(Y.data)

Y.backward()

print("Y =")
print(Y.data)

print("X.grad =")
print(X.grad)

print("W.grad =")
print(W.grad)

'''
x.grad = dl/dy * w.T 
то есть если y = x @ w, тогда
dy/dx00 = w00(5)
dy/dx01 = w10(7)
dy/dx10 = w01(6)
dy/dx11 = w11(8)

аналогично для w:
    dy/dw00 = x00
    dy/w10 = x01
    dy/w01 = x10
    dy/w11 = x11
''' 
