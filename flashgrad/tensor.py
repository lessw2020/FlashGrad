# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# and tinygrad: https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py
# and https://github.com/jaketae/pygrad/blob/master/pygrad/nn/module.py

import numpy as np

class Tensor:
  
  __slots__="grad", "requires_grad"
  training: bool = False
  has_grad: bool = False
  
  def __init__(self, data, requires_grad: bool = False, device=None):
    self.data = data
    self.grad = None
    self.requires_grad = requires_grad
    
    if data.__class__ is list:
      data = np.array(data)
    elif data.__class__ is np.ndarray:
      data = cast(np.ndarray, data)
     
    self.data = data
    return
 
  def __repr__(self):
    return f"FlashTensor {self.data=} with {self.grad=}")
   

     
    
    
 
