import numpy as np
from graphviz import Digraph
from F import MSE, Binary_cross_entropy

def debug_op(s):
    print(f"operation is {s}")

def _broadcast_grad(grad, shape):
    # приведение градиента к форме операнда с учётом broadcasting
    if grad.shape == shape:
        return grad
    # суммирование по лишним ведущим осям
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    axis = tuple(i for i, (g, s) in enumerate(zip(grad.shape, shape)) if g != s and s == 1)
    if axis:
        grad = grad.sum(axis=axis, keepdims=True)
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad

def _unbroadcast(grad, target_shape, axis, keepdims):
    # обратное преобразование градиента после sum/mean по осям
    if axis is not None and not keepdims:
        shape = list(target_shape)
        for ax in (axis if isinstance(axis, (list, tuple)) else [axis]):
            shape[ax] = 1
        grad = grad.reshape(shape)
        grad = np.broadcast_to(grad, target_shape).copy()
    elif axis is None and not keepdims:
        # скалярный sum/mean: градиент константен по всем элементам
        val = float(grad) if np.ndim(grad) == 0 else grad
        grad = np.full(target_shape, val, dtype=float)
    return grad

class Tensor:
    def __init__(self, data, parents=(), op='', name=None, requires_grad=True):
        self.data = np.array(data, dtype=float)                     # numpy-массив данных
        self.grad = np.zeros_like(self.data, dtype=float)           # градиент той же формы, что и data: grad.shape == data.shape
        self.parents = tuple(parents)                               # кортеж родительских <Tensor>-объектов (операндов), породивших данный узел
        self._backward = lambda: None                               # лямбда-функция, реализующая локальное правило обратного прохода для данной операции; вызывается в основном backward()
        self.name = name                                            # символьное имя узла (для отладки/визуализации)
        self.op = op                                                # символьное обозначение операции: '+', '-', '*', '**', 'neg', 'sum', 'sigmoid' и т.д.
        self.requires_grad = requires_grad                          # флаг: требуется ли вычислять градиент для данного узла

    def zero_grad(self):
        # обнуление градиента узла
        self.grad = np.zeros_like(self.data, dtype=float)

    def _ensure_tensor(self, x):
        # приведение other к <Tensor>, если это скаляр; обеспечивает единообразие интерфейса
        if isinstance(x, Tensor):
            return x
        return Tensor(np.array(x, dtype=float), requires_grad=False)

    def T(self):
        # транспонирование: ret = self.T
        ret = Tensor(self.data.T, (self,), 'T')

        def backward():
            # d(self.T)/dself = transpose(grad)
            if self.requires_grad:
                self.grad += ret.grad.T

        ret._backward = backward
        return ret

    def __add__(self, other):
        # приведение other к <Tensor>, если это скаляр; обеспечивает единообразие интерфейса
        other = self._ensure_tensor(other)
        # создание нового узла графа: ret = self + other
        ret = Tensor(self.data + other.data, (self, other), '+')

        def backward():
            # правило дифференцирования суммы: d(self+other)/dself = 1, d(self+other)/dother = 1
            # градиент от ret просто переносится на операнды с учётом broadcasting
            if self.requires_grad:
                self.grad += _broadcast_grad(ret.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _broadcast_grad(ret.grad, other.data.shape)

        ret._backward = backward
        return ret

    def __radd__(self, other):
        # коммутативность сложения: other + self ≡ self + other
        return self + other

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        # создание узла: ret = self - other
        ret = Tensor(self.data - other.data, (self, other), '-')

        def backward():
            # правило дифференцирования разности: d(self-other)/dself = 1, d(self-other)/dother = -1
            if self.requires_grad:
                self.grad += _broadcast_grad(ret.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _broadcast_grad(-ret.grad, other.data.shape)

        ret._backward = backward
        return ret

    def __rsub__(self, other):
        # реализация через унарный минус и сложение: other - self ≡ (-self) + other
        other = self._ensure_tensor(other)
        return other - self

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        ret = Tensor(self.data * other.data, (self, other), '*')

        def backward():
            # правило произведения: d(self*other)/dself = other, d(self*other)/dother = self
            if self.requires_grad:
                self.grad += _broadcast_grad(ret.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _broadcast_grad(ret.grad * self.data, other.data.shape)

        ret._backward = backward
        return ret

    def __rmul__(self, other):
        # коммутативность умножения: other * self ≡ self * other
        return self * other

    def __pow__(self, other):
        # возведение в степень: other может быть скаляром или <Tensor>; градиент по other не считается
        if isinstance(other, Tensor):
            other_val = other.data
        else:
            other_val = other
        # создание узла: ret = self ** other
        ret = Tensor(self.data ** other_val, (self,), '**')

        def backward():
            # правило дифференцирования степенной функции: d(x^n)/dx = n * x^(n-1)
            if self.requires_grad:
                self.grad += ret.grad * other_val * (self.data ** (other_val - 1))

        ret._backward = backward
        return ret

    def __neg__(self):
        # унарный минус: ret = -self
        ret = Tensor(-self.data, (self,), 'neg')

        def backward():
            # d(-self)/dself = -1
            if self.requires_grad:
                self.grad += -ret.grad

        ret._backward = backward
        return ret

    def __truediv__(self, other):
        # деление через умножение на обратную величину: self / other = self * other**(-1)
        other = self._ensure_tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        # обратное деление: other / self
        other = self._ensure_tensor(other)
        return other / self

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def backward():
            # градиент от sum переносится на все элементы с учётом осей
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape, axis, keepdims)

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        # усреднение элементов: ret = mean(self, axis)
        n = self.data.size if axis is None else int(np.prod([self.data.shape[a] for a in (axis if isinstance(axis, (list, tuple)) else [axis])]))
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')

        def backward():
            # градиент по self: 1/n * градиент от out, с broadcasting обратно до формы self
            if self.requires_grad:
                g = out.grad / n
                self.grad += _unbroadcast(g, self.data.shape, axis, keepdims)

        out._backward = backward
        return out

    def exp(self):
        # экспонента: ret = exp(self)
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def backward():
            # d(exp(x))/dx = exp(x)
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = backward
        return out

    def log(self, eps=1e-9):
        # натуральный логарифм с численной стабилизацией: ret = log(clip(self, eps, +inf))
        safe = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe), (self,), 'log')

        def backward():
            # d(log(x))/dx = 1/x
            if self.requires_grad:
                self.grad += out.grad * (1.0 / safe)

        out._backward = backward
        return out

    def sigmoid(self):
        # численно устойчивая сигмоида
        x = self.data
        out_data = np.empty_like(x, dtype=float)
        pos = x >= 0
        out_data[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out_data[~pos] = ex / (1.0 + ex)
        out = Tensor(out_data, (self,), 'sigmoid')

        def backward():
            # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            if self.requires_grad:
                self.grad += out.grad * out.data * (1 - out.data)

        out._backward = backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')

        def backward():
            # d(tanh(x))/dx = 1 - tanh^2(x)
            if self.requires_grad:
                self.grad += out.grad * (1 - out.data ** 2)

        out._backward = backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def backward():
            # d(ReLU(x))/dx = 1 при x > 0, иначе 0
            if self.requires_grad:
                mask = (self.data > 0).astype(float)
                self.grad += out.grad * mask

        out._backward = backward
        return out

    def backward(self):
        # основной обратный проход: топологическая сортировка графа + применение локальных правил дифференцирования
        # итеративная топологическая сортировка (без рекурсии — не падает на глубоких графах)
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                if id(p) not in visited:
                    stack.append((p, False))
        # инициализация градиента выхода: dL/dself = 1
        self.grad = np.ones_like(self.data, dtype=float)
        # обратный проход по топологически отсортированному графу
        for node in reversed(topo):
            if node.requires_grad:
                node._backward()

    def __repr__(self):
        # строковое представление узла: имя, данные, градиент
        return (f"{self.name}:\n"
                f"data =\n{np.array2string(self.data, precision=4, threshold=20)}\n"
                f"grad =\n{np.array2string(self.grad, precision=4, threshold=20)}\n")
