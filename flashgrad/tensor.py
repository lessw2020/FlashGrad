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
              topp.append(v)
     build_topo(self)
     
     # one Tensor at a time, apply chain rule to get gradient
     self.grad = 1
     for v in reversed(topo):
            v._backward()
     
  
    
 
