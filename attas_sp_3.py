"""Tests with the ATTAS aircraft short-period mode estimation.

Decision variables: x, wn, vn.
Steady-state KF and marginalization gains obtained with equality constraints.
Cholesky square roots used.

"""


import collections
import importlib
import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io
import scipy.linalg
import sympy
import sym2num.model
from ceacoest import optim
from ceacoest.modelling import symoptim

import ad


### Enable 64-bit in jax ###
jax.config.update("jax_enable_x64", True)


# Reload modules to simplify testing
for m in [ad]:
    importlib.reload(m)


class Model(symoptim.Model):
    def __init__(self, nx, nu, ny):
        super().__init__()
        
        self.nx = nx
        """Number of states."""
        
        self.nu = nu
        """Number of inputs."""
        
        self.ny = ny
        """Number of outputs."""
        
        nxy = nx + ny
        
        # Define decision variables
        v = self.variables
        v['x'] = [f'x{i}' for i in range(nx)]
        v['wn'] = [f'wn{i}' for i in range(nx)]
        v['vn'] = [f'vn{i}' for i in range(ny)]
        v['xnext'] = [f'xnext{i}' for i in range(nx)]
        v['xprev'] = [f'xprev{i}' for i in range(nx)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})
        
        # Define auxiliary variables
        v['N'] = 'N'
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['uprev'] = [f'uprev{i}' for i in range(nu)]
        
        # Register optimization functions
        self.add_constraint('dynamics')
        self.add_constraint('measurements')
        self.add_objective('wnmerit')
        self.add_objective('vnmerit')
        self.add_objective('logdet_Q')
        self.add_objective('logdet_R')
    
    def dynamics(self, xnext, xprev, uprev, wn, A, B, sQ_tril):
        """Residuals of model dynamics."""
        sQ = tril_mat(sQ_tril)
        return xnext - (A @ xprev + B @ uprev + sQ @ wn)
    
    def measurements(self, y, x, u, vn, C, D, ybias, sR_tril):
        """Residuals of model measurements."""
        sR = tril_mat(sR_tril)
        return y - (C @ x + D @ u + ybias + sR @ vn)
        
    def wnmerit(self, wn):
        """Merit of normalized process noise."""
        return -0.5 * (wn ** 2).sum()

    def vnmerit(self, vn):
        """Merit of normalized measurement noise."""
        return -0.5 * (vn ** 2).sum()

    def logdet_Q(self, sQ_tril, N):
        """Merit of Q log-determinant."""
        sQ = tril_mat(sQ_tril)
        return  -(N - 1) * sum(sympy.log(d) for d in sQ.diagonal())

    def logdet_R(self, sR_tril, N):
        """Merit of R log-determinant."""
        sR = tril_mat(sR_tril)
        return  -N * sum(sympy.log(d) for d in sR.diagonal())
        
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'ntx': len(self.variables['sQ_tril']),
               'nty': len(self.variables['sR_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


def logdet_marg(A, C, sQ_tril, sR_tril, N, use_jax=False):
    # Assemble the input matrices
    sQ = tril_mat(sQ_tril)
    sR = tril_mat(sR_tril)
    Q = sQ @ sQ.T
    R = sR @ sR.T
    
    if use_jax:
        from jax import numpy as np
        Pp = ad.dare(A.T, C.T, Q, R)
    else:
        import numpy as np
        Pp = scipy.linalg.solve_discrete_are(A.T, C.T, Q, R)
    
    nx = len(A)
    ny = len(C)
    z = np.zeros_like(C.T)
    sPp = np.linalg.cholesky(Pp)    
    corr_mat = np.block([[sR, C @ sPp],
                          [z,   sPp]])
    q, r = np.linalg.qr(corr_mat.T)
    s = np.sign(r.diagonal())
    sPc = (r.T * s)[ny:, ny:]
    
    z = np.zeros_like(A)
    pred_mat = np.block([[A @ sPc, sQ],
                         [sPc,     z]])
    q, r = np.linalg.qr(pred_mat.T)
    s = np.sign(r.diagonal())
    sPr = (r.T * s)[nx:, nx:]
    
    eps = 1e-40
    log_det_sPc = np.sum(np.log(np.abs(sPc.diagonal()) + eps))
    log_det_sPr = np.sum(np.log(np.abs(sPr.diagonal()) + eps))
    return (N-1) * log_det_sPr + log_det_sPc


logdet_marg_grad = jax.grad(logdet_marg, range(4))
logdet_marg_d2A = jax.jacfwd(lambda *a, **k: logdet_marg_grad(*a, **k)[0:], 0)
logdet_marg_d2C = jax.jacfwd(lambda *a, **k: logdet_marg_grad(*a, **k)[1:], 1)
logdet_marg_d2sQ = jax.jacfwd(lambda *a, **k: logdet_marg_grad(*a, **k)[2:], 2)
logdet_marg_d2sR = jax.jacfwd(lambda *a, **k: logdet_marg_grad(*a, **k)[3:], 3)


class LogDetMargObjective:
    def __init__(self):
        f_sig = inspect.signature(logdet_marg)
        params = list(f_sig.parameters.values())[:-1]
        self.__signature__ = inspect.Signature(params)
    
    @staticmethod
    def __call__(*args, **kwargs):
        return logdet_marg(*args, **kwargs)

    @staticmethod
    def grad(*args, **kwargs):
        # Ensure jax is used
        args = args[:5]
        kwargs['use_jax'] = True

        # Compute gradients
        d_A, d_C, d_sQ_tril, d_sR_tril = logdet_marg_grad(*args, **kwargs)
        
        # Build output dictionary
        grad = collections.OrderedDict()
        grad['A'] = d_A
        grad['C'] = d_C
        grad['sQ_tril'] = d_sQ_tril
        grad['sR_tril'] = d_sR_tril
        return grad

    @classmethod
    def hess_nnz(cls, dec_shapes, out_shape):
        hess_ind = cls.hess_ind(dec_shapes, out_shape)
        return sum(v[0].size for v in hess_ind.values())
    
    @staticmethod
    def hess_ind(dec_shapes, out_shape):
        # Retrieve variable dimensions
        ny, nx = dec_shapes['C']
        ntx, = dec_shapes['sQ_tril']
        nty, = dec_shapes['sR_tril']
        nA = nx ** 2
        nC = ny * nx
        
        # Build output dictionary
        hess = collections.OrderedDict()
        hess['A', 'A'] = np.tril_indices(nA)
        hess['C', 'A'] = [np.repeat(np.arange(nC), nA),
                          np.tile(np.arange(nA), nC)]
        hess['sQ_tril', 'A'] = [np.repeat(np.arange(ntx), nA),
                                np.tile(np.arange(nA), ntx)]
        hess['sR_tril', 'A'] = [np.repeat(np.arange(nty), nA),
                                np.tile(np.arange(nA), nty)]
        hess['C', 'C'] = np.tril_indices(ny*nx)
        hess['sQ_tril', 'C'] = [np.repeat(np.arange(ntx), nC),
                                np.tile(np.arange(nC), ntx)]
        hess['sR_tril', 'C'] = [np.repeat(np.arange(nty), nC),
                                np.tile(np.arange(nC), nty)]
        hess['sQ_tril', 'sQ_tril'] = np.tril_indices(ntx)
        hess['sR_tril', 'sQ_tril'] = [np.repeat(np.arange(nty), ntx),
                                      np.tile(np.arange(ntx), nty)]
        hess['sR_tril', 'sR_tril'] = np.tril_indices(nty)
        
        # Convert to ndarray and return
        for k, v in hess.items():
            hess[k] = np.asarray(v)
        return hess
    
    @staticmethod
    def hess_val(*args, **kwargs):
        # Ensure jax is used
        args = args[:5]
        kwargs['use_jax'] = True
        
        # Compute Hessian
        d_A_A, d_C_A, d_sQ_A, d_sR_A, = logdet_marg_d2A(*args,**kwargs)
        d_C_C, d_sQ_C, d_sR_C, = logdet_marg_d2C(*args,**kwargs)
        d_sQ_sQ, d_sR_sQ, = logdet_marg_d2sQ(*args,**kwargs)
        d_sR_sR, = logdet_marg_d2sR(*args,**kwargs)
        
        # Get shapes
        C = kwargs.get('C') or args[1]
        sQ_tril = kwargs.get('sQ_tril') or args[2]
        sR_tril = kwargs.get('sR_tril') or args[3]
        ny, nx = C.shape
        ntx = sQ_tril.size
        nty = sR_tril.size
        
        # Build output dictionary
        hess = collections.OrderedDict()
        hess['A', 'A'] = d_A_A.reshape(nx**2, nx**2)[np.tril_indices(nx**2)]
        hess['C', 'A'] = d_C_A
        hess['sQ_tril', 'A'] = d_sQ_A
        hess['sR_tril', 'A'] = d_sR_A
        hess['C', 'C'] = d_C_C.reshape(ny*nx, ny*nx)[np.tril_indices(ny*nx)]
        hess['sQ_tril', 'C'] = d_sQ_C
        hess['sR_tril', 'C'] = d_sR_C
        hess['sQ_tril', 'sQ_tril'] = d_sQ_sQ[np.tril_indices(ntx)]
        hess['sR_tril', 'sQ_tril'] = d_sR_sQ
        hess['sR_tril', 'sR_tril'] = d_sR_sR[np.tril_indices(nty)]
        return hess


class Problem(optim.Problem):
    
    def __init__(self, model, y, u):
        super().__init__()
        
        self.model = model
        """Underlying model."""
        
        self.y = np.asarray(y)
        """Measurements."""
        
        self.u = np.asarray(u)
        """Inputs."""

        self.uprev = self.u[:-1]
        """Previous inputs."""
        
        N = len(y)
        self.N = N
        """Number of measurement instants."""
        
        nx = model.nx
        nu = model.nu
        ny = model.ny
        ntx = model.ntx
        nty = model.nty
        nt2x = nx * (2*nx + 1)
        ntxy = (nx + ny) * (nx + ny + 1) // 2
        
        assert y.ndim == 2
        assert y.shape[1] == ny
        assert u.shape == (N, nu)
        assert N > 1
        
        # Register decision variables
        self.add_decision('ybias', ny)
        self.add_decision('sQ_tril', nty)
        self.add_decision('sR_tril', nty)
        self.add_decision('A', (nx, nx))
        self.add_decision('B', (nx, nu))
        self.add_decision('C', (ny, nx))
        self.add_decision('D', (ny, nu))
        self.add_decision('vn', (N, ny))
        self.add_decision('wn', (N - 1, nx))
        x = self.add_decision('x', (N, nx))
        
        # Define and register dependent variables
        xprev = optim.Decision((N-1, nx), x.offset)
        xnext = optim.Decision((N-1, nx), x.offset + nx)
        self.add_dependent_variable('xprev', xprev)
        self.add_dependent_variable('xnext', xnext)
    
        # Register problem functions
        self.add_constraint(model.dynamics, (N - 1, nx))
        self.add_constraint(model.measurements, (N, ny))
        self.add_objective(model.wnmerit, N - 1)
        self.add_objective(model.vnmerit, N)
        self.add_objective(model.logdet_Q, ())
        self.add_objective(model.logdet_R, ())
        self.add_objective(LogDetMargObjective(), ())
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'u': self.u, 'uprev': self.uprev, 'N': self.N,
                **super().variables(dvec)}


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i>=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(elem):
    import numpy as np
    if isinstance(elem, jnp.ndarray) and not isinstance(elem, np.ndarray):
        np = jnp
    
    ntril = len(elem)
    n = int(round(0.5*(np.sqrt(8*ntril + 1) - 1)))
    mat = np.zeros((n, n), dtype=elem.dtype)
    for ind, val in zip(tril_ind(n), elem):
        if np is not jnp:
            mat[ind] = val
        else:
            mat = mat.at[ind].set(val)
    return mat


def load_data():
    # Retrieve data
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1'][30:-30]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r

    # Shift and rescale
    yshift = np.r_[-0.003, -0.04]
    yscale = np.r_[10.0, 20.0]
    ushift = np.r_[-0.04]
    uscale = np.r_[25.0]
    
    y = (y + yshift) * yscale
    u = (u + ushift) * uscale
    
    # Add artificial noise
    np.random.seed(0)
    N = len(y)
    y_peak_to_peak = y.max(0) - y.min(0)
    #y[:, :] += y_peak_to_peak[:] * 3e-2 * np.random.randn(N, 2)
    
    return t, u, y, yshift, yscale, ushift, uscale


def load_data2(yshift, yscale, ushift, uscale):
    # Retrieve data
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv2.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv2'][50:-100]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r
    
    # Shift and rescale
    y = (y + yshift) * yscale
    u = (u + ushift) * uscale    
    return t, u, y


if __name__ == '__main__':
    nx = 2
    nu = 1
    ny = 2
    
    # Load experiment data
    t, u, y, yshift, yscale, ushift, uscale = load_data()
    t2, u2, y2 = load_data2(yshift, yscale, ushift, uscale)
    symmodel = Model(nx=nx, nu=nu, ny=ny)
    model = symmodel.compile_class()()
    model.dt = t[1] - t[0]
    problem = Problem(model, y, u)
    
    N = len(y)
    
    # Compute initial guess
    np.random.seed(0)
    sQ0 = np.eye(nx) * 1 + np.diag(np.random.rand(nx)) * 0.1
    sR0 = np.eye(ny) * 1e-1 + np.diag(np.random.rand(nx)) * 0.1
    Q0 = sQ0 @ sQ0.T
    R0 = sR0 @ sR0.T
    A0 = np.eye(nx)
    C = np.eye(nx)
    sQ0_tril = sQ0[np.tril_indices(nx)]
    sR0_tril = sR0[np.tril_indices(ny)]
        
    vn0 = np.zeros((N, ny))
    x0 = y - vn0 @ sR0.T
    wn0 = (x0[1:] - x0[:-1] @ A0.T) @ np.linalg.inv(sQ0.T)
    
    # Set initial guess
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = A0
    var0['B'][:] = np.zeros((nx, nu))
    var0['C'][:] = C
    var0['D'][:] = 0
    var0['x'][:] = x0
    var0['vn'][:] = vn0
    var0['wn'][:] = wn0
    var0['sQ_tril'][:] = sQ0_tril
    var0['sR_tril'][:] = sR0_tril
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['C'][:] = C
    var_U['C'][:] = C
    var_L['D'][:] = 0
    var_U['D'][:] = 0
    #var_L['sQ_tril'][~tril_diag(nx)] = 0
    #var_U['sQ_tril'][~tril_diag(nx)] = 0
    #var_L['sR_tril'][~tril_diag(ny)] = 0
    #var_U['sR_tril'][~tril_diag(ny)] = 0
    var_L['sQ_tril'][tril_diag(nx)] = 1e-10
    var_L['sR_tril'][tril_diag(ny)] = 1e-4
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    var_constr_L = problem.unpack_constraints(constr_L)
    var_constr_U = problem.unpack_constraints(constr_U)
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['measurements'][:] = 1
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['sR_tril'][:] = 1e2
    var_scale['sQ_tril'][:] = 1e2
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_scaling_on_demand', 'no')
        nlp.add_str_option('linear_system_scaling', 'mc19')
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 100.0)
        nlp.add_num_option('tol', 1e-8)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    wn = opt['wn']
    vn = opt['vn']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    ybias = opt['ybias']
    sQ = tril_mat(opt['sQ_tril'])
    sR = tril_mat(opt['sR_tril'])
    
    yopt = xopt @ C.T + u @ D.T + ybias
    Q = sQ @ sQ.T
    R = sR @ sR.T
    
    Ac = scipy.linalg.logm(A) / model.dt
