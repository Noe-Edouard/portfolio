import numpy as np
from numpy import ndarray
import gc
from skimage.filters import frangi as frangi_skimage
from typing import Callable, Optional, Sequence, Literal

from core.utils.gpu import is_gpu_available_available

class Enhancer:
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and is_gpu_available_available()
        self.selector = {
            'frangi': self.frangi,
        }

    def select_enhancement_function(self, method: Literal['frangi']) -> Callable[..., ndarray]:
        if method not in self.selector:
            raise ValueError(f"Unknown enhancement method: {method}. Valid methods : {list(self.selector.keys())}")
        
        return self.selector[method]
    
    def frangi(
        self,
        image: ndarray,
        hessian_function: Callable[..., list[ndarray]] = None,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0},
        scales: Optional[Sequence[int]] = range(0, 10, 2),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        black_ridges: Optional[bool] = True,
        skimage = False,
    ) -> ndarray:
        
        # Check for GPU compatibility if using a custom Hessian function
        if hessian_function and hasattr(hessian_function.__self__, 'use_gpu'):
            if hessian_function.__self__.use_gpu != self.use_gpu:
                raise ValueError(f'GPU must be used for both Derivator and Enhancer or none. Current values: Derivator({hessian_function.__self__.use_gpu}, Enhancer({self.use_gpu}))')
            
        if skimage:
            if self.use_gpu:
                raise ValueError('Skimage function can only be used for non GPU processing.')
            else:
                return frangi_skimage(image, sigmas=scales, alpha=alpha, beta=beta, gamma=gamma, black_ridges = black_ridges)

        if self.use_gpu:
            import cupy as cp
            # from cucim.skimage.feature import hessian_matrix as gpu_hessian_matrix
            from core.utils.gpu import gpu_hessian_matrix_eigvals
            xp = cp
            eigvals_function = gpu_hessian_matrix_eigvals
            pass
        else:
            from skimage.feature import hessian_matrix_eigvals as cpu_hessian_eigvals
            xp = np
            # hessian_function = hessian_function if hessian_function else cpu_hessian_matrix
            eigvals_function = cpu_hessian_eigvals
        
        from skimage.feature import hessian_matrix as cpu_hessian_matrix
        hessian_function = hessian_function if hessian_function else hessian_matrix
        
        if not black_ridges:
            image = -image
        
        image = xp.asarray(image)
        image = image.astype(xp.float32, copy=False)
        filtered_image = xp.zeros_like(image)
        
        for scale in scales:
            hessian = hessian_function(image, sigma=scale, **hessian_params)
            
            eigvals = eigvals_function(hessian)

            # All subsequent operations are fine since eigvals is now the correct type
            eigvals = xp.take_along_axis(eigvals, xp.abs(eigvals).argsort(0), axis=0)
            
            
            if image.ndim == 2:
                lambda1 = eigvals[0]
                lambda2 = xp.maximum(eigvals[1], 1e-10)
                r_a = xp.inf
                r_b = xp.abs(lambda1) / lambda2

            else:
                lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]
                lambda2_pos = xp.maximum(lambda2, 1e-10)
                lambda3_pos = xp.maximum(lambda3, 1e-10)
                
                r_a = lambda2_pos / lambda3_pos
                r_b = xp.abs(lambda1) / xp.sqrt(lambda2_pos * lambda3_pos)
            
            s = xp.sqrt((eigvals**2).sum(axis=0))

            if gamma is None:
                gamma = s.max() / 2 if s.max() != 0 else 1
            
            vesselness = 1.0 - xp.exp(-(r_a**2) / (2 * alpha**2))
            vesselness *= xp.exp(-(r_b**2) / (2 * beta**2))
            vesselness *= (1.0 - xp.exp(-(s**2) / (2 * gamma**2)))
            
            filtered_image = xp.maximum(filtered_image, vesselness)
            
            del hessian, eigvals, vesselness, s
            gc.collect()
            
        return xp.asnumpy(filtered_image) if self.use_gpu else filtered_image

