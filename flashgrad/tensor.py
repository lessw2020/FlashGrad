# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# and tinygrad: https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py
# and https://github.com/jaketae/pygrad/blob/master/pygrad/nn/module.py

import numpy as np

class Tensor:
  
  __slots__="grad", "requires_grad"
  training: bool = False
  has_grad: bool = False
  
  def __init__(self, data, children=(), op="", requires_grad: bool = False, device=None):
    self.data = data
    self.grad = None
    self.requires_grad = requires_grad
    
    # backprop
    self._backward = lambda: None
    self._prev = set(children)
    self._op = op
    
    
    if data.__class__ is list:
      data = np.array(data)
    elif data.__class__ is np.ndarray:
      data = cast(np.ndarray, data)
     
    self.data = data
    return
 
  def __repr__(self):
    return f"FlashTensor {self.data=} with {self.grad=}")
  
  def __add__(self, other):
    if other not isinstance(other, Tensor):
      other = Tensor(other)

    out = Tensor(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad+=out.grad
      other.grad+=out.grad
    out._backward = _backward
    
    return out
  
  def __mul__(self, other):
    if other not isinstance(other, Tensor):
      other = Tensor(other)
    out = Tensor(self.data * other.data, (self, other), "*")
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
     out._backward = _backward
    
    return out
  def __pow__(self,other):
    assert isinstance(other, (int, float)), "only int or float powers supported"
    out = Tensor(self.data**other, (self,),f"**{other}")
    
    def _backward():
      self.grad +=(other* self.data**(other-1))* out.grad
     out._backward = _backward
    
    return out
  
  def relu(self):
    out = Tensor(0 if self.data <0 else self.data, (self,), 'ReLU')
    
    def _backward():
      self.grad += (out.data > 0)* out.grad
      
    out._backward = _backward
    return out
  
    
  def backward(self):
    if self.data.shape > 1:
      print(f"Backward called on non-scalar. Aborting..."
     
    topo = []
    visited = set()
    def build_topo(v):
            if v not in visited:
              visited.add(v)
              for child in v._prev:
                build_topo(child)
              topo.append(v)
     build_topo(self)
     
     # one Tensor at a time, apply chain rule to get gradient
     self.grad = 1
     for v in reversed(topo):
            v._backward()
     
  
    
 
