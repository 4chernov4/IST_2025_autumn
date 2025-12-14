import numpy as np
from numpy.linalg import LinAlgError, norm, pinv
from collections import defaultdict
from datetime import datetime
from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve


class LineSearchTool:
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method')

    @classmethod
    def from_dict(cls, options):
        return cls(**options)

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c

        alpha = previous_alpha if previous_alpha is not None else getattr(self, 'alpha_0', 1.0)

        if self._method == 'Armijo':
            phi0 = float(oracle.func_directional(x_k, d_k, 0))
            dphi0 = float(oracle.grad_directional(x_k, d_k, 0))
            while float(oracle.func_directional(x_k, d_k, alpha)) > phi0 + self.c1 * alpha * dphi0:
                alpha /= 2
            return alpha

        # Wolfe line search
        f = oracle.func
        g = oracle.grad
        alpha_ls, _, _, _, _, _ = line_search(
            f, g, x_k, d_k,
            gfk=g(x_k),
            old_fval=f(x_k),
            c1=self.c1,
            c2=self.c2
        )
        if alpha_ls is not None:
            return alpha_ls

        # fallback Armijo
        phi0 = float(oracle.func_directional(x_k, d_k, 0))
        dphi0 = float(oracle.grad_directional(x_k, d_k, 0))
        while float(oracle.func_directional(x_k, d_k, alpha)) > phi0 + self.c1 * alpha * dphi0:
            alpha /= 2
        return alpha


def get_line_search_tool(options=None):
    if isinstance(options, LineSearchTool):
        return options
    if isinstance(options, dict):
        return LineSearchTool.from_dict(options)
    return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    ls = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    alpha_prev = None

    if display:
        print("Gradient descent started")
        print(f"Initial point: {x_0}")

    for k in range(max_iter + 1):
        f_val = float(oracle.func(x))
        grad = oracle.grad(x)
        grad_norm = float(norm(grad))

        if trace:
            history['time'].append(k)
            history['func'].append(f_val)
            history['grad_norm'].append(grad_norm)
            if x.size <= 2:
                history['x'].append(x.copy())

        if grad_norm <= tolerance:
            return x, 'success', history

        if k == max_iter:
            break

        d = -grad
        alpha = ls.line_search(oracle, x, d, alpha_prev)
        if alpha is None or alpha <= 0 or np.isnan(alpha):
            return x, 'computational_error', None

        alpha_prev = alpha
        x = x + alpha * d

    return x, 'iterations_exceeded', None



def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    ls = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    alpha_prev = None

    if display:
        print("Newton's method started")
        print(f"Initial point: {x_0}")

    for k in range(max_iter + 1):
        f_val = float(oracle.func(x))
        grad = oracle.grad(x)
        grad_norm = float(norm(grad))

        if trace:
            history['time'].append(k)
            history['func'].append(f_val)
            history['grad_norm'].append(grad_norm)
            if x.size <= 2:
                history['x'].append(x.copy())

        if grad_norm <= tolerance:
            return x, 'success', history

        if k == max_iter:
            break

        try:
            hess = oracle.hess(x)
            d = -np.linalg.solve(hess, grad)
        except Exception:
            return x, 'computational_error', None

        alpha = ls.line_search(oracle, x, d, alpha_prev)
        if alpha is None or alpha <= 0 or np.isnan(alpha):
            return x, 'computational_error', None

        alpha_prev = alpha
        x = x + alpha * d

    return x, 'iterations_exceeded', None
