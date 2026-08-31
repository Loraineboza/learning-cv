import numpy as np

class Tensor:
    def __init__(self, data, op='', *parents):
        self.data = np.array(data) #<Tensor>-объект
        self.grad = np.zeros_like(self.data) #матрица данных.shape (data.shape) == градиент.shape (self.grad.shape) 
        self.op = op #+/-/*/@/**/-x/ etc.
        self.parents = parents #родители-операнды, создавшие ret 
        self._backward = lambda: None #лямба, которая вызывает соответствующую backward ф-ию для каждого <Tensor> объекта, который был 
                                      #создан через определенную мат. операцию. Необходима для подсчета производных с учетом знака  
    

    def __add__(self, other):
        #является ли аргумент other типом <Tensor>. В противном случае подается как скаляр (например степень это константа, поэтому 
        #подается в виде скаляра)
        other = other if isinstance(other, Tensor) else Tensor(other) 
        #создается объект <Tensor> с потомками, операнды которые его создали (поэтому дал название не children, а parents. Но это на свой выбор ;D)) 
        ret = Tensor(self.data + other.data, '+', self, other)

        def backward():
            #операндов, создавшие ret максимум 2. Однако цикл я все равно поставил, чтобы не повторятся для более сложных вычислений производных и градиентов (см. ниже)
            for parent in ret.parents:
                #dl/da = dl/dy * (+/-)1
                parent.grad += ret.grad
        #сохранение лямбды для соответствующего маг. метода. Позже основной (main) backward вызевет каждую _backward лямбду для каждого 
        #соответствующего метода, чтобы подсчитать производные для каждого <Tensor>-объекта 
        ret._backward = backward
        return ret

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ret = Tensor(self.data * other.data, '*', self, other)

        def backward():
            for i in range(len(ret.parents)):
                #dl/da = dl/dy * dy/db. Производная по 1-ому объекту равна произведению между dl/dy и значению 2-ого объекта 
                #та же логика и для 2-ого объекта. Меняются местами лишгь условия
                ret.parents[i].grad += ret.grad * ret.parents[0 if i>0 else 1].data

        ret._backward = backward
        return ret
    
    #<Tensor> @ <Tensor>
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ret = Tensor(np.matmul(self.data, other.data), '@', self, other)

        def backward():
            for i in range(len(ret.parents)):
                #dl/dx = dl/dy @ W.T; dl/dw = X.T @ dl/dy. Применено транспонирование для поиска производных каждого элемента 
                #соответствующих матриц
                ret.parents[i].grad += ret.grad @ ret.parents[1].data.T if i==0 else ret.parents[0].data.T @ ret.grad

        ret._backward = backward
        return ret 

    #<Tensor> ** 2
    def __pow__(self, other):
        #other является скаляром, поэтому не учитывается и не создается в виде отдельного <Tensor>-объекта 
        ret = Tensor(self.data ** other, '**', self)

        def backward():
            #dl/da = dl/dy * 2 * b, где 2 это степень(как пример), а b это ее основание (исходное выражение b^2)
            #степень - константа, поэтому она сдвигается влево и представляется множителем для своего основания
            self.grad += ret.grad * other * (self.data ** (other - 1))

        ret._backward = backward 
        return ret

    #main backward
    def backward(self):
        graph = build_graph(self) #граф из <Tensor> последовательных объектов 
        self.grad = np.ones_like(self.data) #по умолчению градиент dl/dself равен 1. Размерность сохраняется
        for tensor in graph[::-1]: #вызвать backward лямбду у каждого <Tensor> объекта (кроме степени) в графе
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
        #унарный минус. Требуется для использования __add__ вместо минуса: self + (-other)
        return self * -1 #-self

    def __rmul__(self, other):
        return self * other

#постройка графа: поиск всех зависимых <Tensor>-объектов, 
#которые идут в последовательном неповторяющийся порядке. 

#Необходим для правильного порядка вызовов лямбд _backwawrd всех <Tensor>-объектов  
def build_graph(tensor, graph=None):                                      
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
если y = x @ w, тогда
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
