import numpy as np
import scipy
from scipy.special import expit, logsumexp


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        log_loss_terms = logsumexp(np.column_stack([np.zeros_like(z), z]), axis=1)
        log_loss = np.mean(log_loss_terms)
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        return log_loss + reg_term

    def grad(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        sigmoid = expit(z)
        grad_log = self.matvec_ATx(-self.b * sigmoid) / m
        grad_reg = self.regcoef * x
        return grad_log + grad_reg

    def hess(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        sigmoid = expit(z)
        diag_s = sigmoid * (1 - sigmoid)
        hess_log = self.matmat_ATsA(diag_s) / m
        n = len(x)
        hess_reg = self.regcoef * np.eye(n)
        return hess_log + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._reset_cache()

    def _reset_cache(self):
        self._x_cache = None
        self._d_cache = None
        self._Ax_cache = None
        self._Ad_cache = None

    def func(self, x):
        self._reset_cache()
        return super().func(x)

    def grad(self, x):
        self._reset_cache()
        return super().grad(x)

    def hess(self, x):
        self._reset_cache()
        return super().hess(x)

    def func_directional(self, x, d, alpha):
        if (self._x_cache is None or self._d_cache is None or 
            not np.array_equal(self._x_cache, x) or not np.array_equal(self._d_cache, d)):
            self._x_cache = x.copy()
            self._d_cache = d.copy()
            self._Ax_cache = self.matvec_Ax(x)
            self._Ad_cache = self.matvec_Ax(d)
        
        Ax_alpha = self._Ax_cache + alpha * self._Ad_cache
        m = len(self.b)
        z = -self.b * Ax_alpha
        
        log_loss_terms = logsumexp(np.column_stack([np.zeros_like(z), z]), axis=1)
        log_loss = np.mean(log_loss_terms)
        
        x_alpha = x + alpha * d
        reg_term = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        
        return log_loss + reg_term

    def grad_directional(self, x, d, alpha):
        if (self._x_cache is None or self._d_cache is None or 
            not np.array_equal(self._x_cache, x) or not np.array_equal(self._d_cache, d)):
            self._x_cache = x.copy()
            self._d_cache = d.copy()
            self._Ax_cache = self.matvec_Ax(x)
            self._Ad_cache = self.matvec_Ax(d)
        
        Ax_alpha = self._Ax_cache + alpha * self._Ad_cache
        m = len(self.b)
        z = -self.b * Ax_alpha
        
        sigmoid = expit(z)
        grad_dot = np.dot(-self.b * sigmoid, self._Ad_cache) / m
        x_alpha = x + alpha * d
        reg_term = self.regcoef * np.dot(x_alpha, d)
        
        return grad_dot + reg_term


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    if scipy.sparse.issparse(A):
        def matvec_Ax(x):
            return A.dot(x)
        
        def matvec_ATx(x):
            return A.T.dot(x)
        
        def matmat_ATsA(s):
            S = scipy.sparse.diags(s)
            return A.T.dot(S.dot(A))
    else:
        def matvec_Ax(x):
            return A.dot(x)
        
        def matvec_ATx(x):
            return A.T.dot(x)
        
        def matmat_ATsA(s):
            return (A.T * s).dot(A)
    
    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        grad[i] = (func(x + eps * e_i) - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f_i = func(x + eps * e_i)
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1
            f_j = func(x + eps * e_j)
            f_ij = func(x + eps * e_i + eps * e_j)
            
            hess[i, j] = (f_ij - f_i - f_j + f_x) / (eps * eps)
    
    return hess