"""
@author: jpzxshi
"""
import numpy as np
import torch
from .module import Algorithm
from .fnn import FNN
from ..integrator.hamiltonian import SV
from ..utils import lazy_property, grad

class HNN(Algorithm):
    '''Hamiltonian neural networks.
    '''
    def __init__(self, H_size, activation='tanh', initializer='orthogonal', integrator='midpoint'):
        super(HNN, self).__init__()
        self.H_size = H_size
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        
        self.ms = self.__init_modules()
    
    # 训练时的前向传播
    def criterion(self, x0h, x1):
        return self.__integrator_loss(x0h[0], x1, x0h[1])
    
    # 测试时的前向传播
    def predict(self, x0, h, steps=1, keepinitx=False, returnnp=False):
        x0 = self._to_tensor(x0)
        N = max(int(h * 10), 1) # 忽略这个N吧。h是time step，steps是步数，然后忘记N
        solver = SV(self.ms['H'], None, iterations=10, order=4, N=N)
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()  # 返回的是长度为len(steps)的数组
        return res.cpu().detach().numpy() if returnnp else res
    
    @lazy_property  # 当你第一次访问某个被 @lazy_property 修饰的方法时，会自动计算并缓存结果，之后再次访问该属性时直接返回缓存值，而不会重复计算
    def J(self):
        d = int(self.H_size[0] / 2)
        res = np.eye(self.H_size[0], k=d) - np.eye(self.H_size[0], k=-d)
        return torch.tensor(res, dtype=self.dtype, device=self.device)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = FNN(self.H_size, self.activation, self.initializer)
        return modules
    
    def __integrator_loss(self, x0, x1, h):
        if self.integrator == 'midpoint':
            mid = ((x0 + x1) / 2).requires_grad_(True)
            gradH = grad(self.ms['H'](mid), mid)
            return torch.nn.MSELoss()((x1 - x0) / h, gradH @ self.J)
        else:
            raise NotImplementedError