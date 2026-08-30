import numpy as np

class Tensor:
    def __init__(self, data, op='', *parents):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data) #матрица данных.shape == градиент.shape 
        self.op = op
        self.parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        ret = Tensor(self.data + other.data, '+', self, other)
        def backward():
            for parent in ret.parents:
                parent.grad += ret.grad

        ret._backward = backward
        return ret

    def __mul__(self, other):
        ret = Tensor(self.data * other.data, '*', self, other)
        def backward():
            for i in range(len(ret.parents)):
                ret.parents[i].grad += ret.grad * ret.parents[0 if i>0 else 1].data

        ret._backward = backward
        return ret
    
    def backward(self):
        graph = build_graph(self)
        self.grad = 1
        for tensor in graph:
            tensor._backward()
def build_graph(tensor, graph=None): #постройка графа: поиск всех зависимых от <tensor> объектов, которые идут в последовательном неповторяющийся порядке
    if graph is None:
        graph = []

    if tensor not in graph:
        graph.append(tensor)

    for parent in tensor.parents:
        build_graph(parent, graph)

    return graph


a = Tensor(2)
b = Tensor(3)

c = a + b
d = c * a

d.backward()

print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
