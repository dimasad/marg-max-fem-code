"""Tests with the ATTAS aircraft short-period mode estimation.

Decision variables: x, wn, vn.
Steady-state KF and marginalization gains obtained with equality constraints.
Cholesky square roots used.

"""


import importlib
import os

import numpy as np
import scipy.io
import scipy.linalg
import sympy
import sym2num.model
from ceacoest import optim
from ceacoest.modelling import symoptim


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
        v['Kn'] = [[f'Kn{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['Mn'] = [[f'Mn{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        v['sPp_tril'] = [f'sPp{i}_{j}' for i,j in tril_ind(nx)]
        v['sPc_tril'] = [f'sPc{i}_{j}' for i,j in tril_ind(nx)]
        v['sPr_tril'] = [f'sPr{i}_{j}' for i,j in tril_ind(nx)]
        po = [[f'pred_orth{i}_{j}' for j in range(2*nx)] for i in range(2*nx)]
        co = [[f'corr_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx+ny)]
        v['pred_orth'] = po
        v['corr_orth'] = co
        self.decision.update({k for k in v if k != 'self'})
        
        # Define auxiliary variables
        v['N'] = 'N'
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['uprev'] = [f'uprev{i}' for i in range(nu)]
        
        # Register optimization functions
        self.add_constraint('dynamics')
        self.add_constraint('measurements')
        self.add_constraint('pred_orthogonality')
        self.add_constraint('corr_orthogonality')
        self.add_constraint('pred_cov')
        self.add_constraint('corr_cov')
        self.add_objective('wnmerit')
        self.add_objective('vnmerit')
        self.add_objective('logdet_Q')
        self.add_objective('logdet_R')
        self.add_objective('logdet_Pp')
        self.add_objective('logdet_marg')
    
    def dynamics(self, xnext, xprev, uprev, wn, A, B, sQ_tril):
        """Residuals of model dynamics."""
        sQ = tril_mat(sQ_tril)
        return xnext - (A @ xprev + B @ uprev + sQ @ wn)
    
    def measurements(self, y, x, u, vn, C, D, ybias, sR_tril):
        """Residuals of model measurements."""
        sR = tril_mat(sR_tril)
        return y - (C @ x + D @ u + ybias + sR @ vn)
    
    def pred_orthogonality(self, pred_orth):
        resid = 0.5 * (pred_orth @ pred_orth.T - np.eye(2 * self.nx))
        return [resid[i] for i in tril_ind(2 * self.nx)]
    
    def corr_orthogonality(self, corr_orth):
        resid = 0.5 * (corr_orth @ corr_orth.T - np.eye(self.nx + self.ny))
        return [resid[i] for i in tril_ind(self.nx + self.ny)]
    
    def pred_cov(self, A, sPp_tril, sPc_tril, sPr_tril, sQ_tril, Mn, pred_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sPr = tril_mat(sPr_tril)
        sQ = tril_mat(sQ_tril)
        
        zeros = np.zeros(A.shape)
        M1 = np.block([[sPp, zeros], 
                       [Mn,  sPr]])
        M2 = np.block([[A @ sPc, sQ], 
                       [sPc,     zeros]])
        return M1 @ pred_orth - M2
    
    def corr_cov(self, C, sR_tril, sRp_tril, sPp_tril, sPc_tril, Kn, corr_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sRp = tril_mat(sRp_tril)
        sR = tril_mat(sR_tril)
        
        zeros = np.zeros((self.nx, self.ny))
        M1 = np.block([[sRp,  zeros.T],
                       [Kn, sPc]])
        M2 = np.block([[sR,    C @ sPp], 
                       [zeros, sPp]])
        return M1 @ corr_orth - M2
    
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

    def logdet_Pp(self, sPp_tril):
        """Merit of initial state covariance log-determinant."""
        sPp = tril_mat(sPp_tril)
        return  -sum(sympy.log(d) for d in sPp.diagonal())

    def logdet_R(self, sR_tril, N):
        """Merit of R log-determinant."""
        sR = tril_mat(sR_tril)
        return  -N * sum(sympy.log(d) for d in sR.diagonal())
    
    def logdet_marg(self, sPc_tril, sPr_tril, N):
        """Merit correction of state marginalization."""
        sPc = tril_mat(sPc_tril)
        sPr = tril_mat(sPr_tril)
        log_det_sPc = sum(sympy.log(d) for d in sPc.diagonal())
        log_det_sPr = sum(sympy.log(d) for d in sPr.diagonal())
        return  (N-1) * log_det_sPr + log_det_sPc
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'ntx': len(self.variables['sQ_tril']),
               'nty': len(self.variables['sR_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


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
        self.add_decision('sRp_tril', nty)
        self.add_decision('sPp_tril', ntx)
        self.add_decision('sPc_tril', ntx)
        self.add_decision('sPr_tril', ntx)
        self.add_decision('A', (nx, nx))
        self.add_decision('B', (nx, nu))
        self.add_decision('C', (ny, nx))
        self.add_decision('D', (ny, nu))
        self.add_decision('Kn', (nx, ny))
        self.add_decision('Mn', (nx, ny))
        self.add_decision('vn', (N, ny))
        self.add_decision('wn', (N - 1, nx))
        self.add_decision('pred_orth', (2*nx, 2*nx))
        self.add_decision('corr_orth', (nx + ny, nx + ny))
        x = self.add_decision('x', (N, nx))
        
        # Define and register dependent variables
        xprev = optim.Decision((N-1, nx), x.offset)
        xnext = optim.Decision((N-1, nx), x.offset + nx)
        self.add_dependent_variable('xprev', xprev)
        self.add_dependent_variable('xnext', xnext)
    
        # Register problem functions
        self.add_constraint(model.dynamics, (N - 1, nx))
        self.add_constraint(model.measurements, (N, ny))
        self.add_constraint(model.pred_orthogonality, nt2x)
        self.add_constraint(model.corr_orthogonality, ntxy)
        self.add_constraint(model.pred_cov, (2*nx, 2*nx))
        self.add_constraint(model.corr_cov, (nx + ny, nx + ny))
        self.add_objective(model.wnmerit, N - 1)
        self.add_objective(model.vnmerit, N)
        self.add_objective(model.logdet_Q, ())
        self.add_objective(model.logdet_Pp, ())
        self.add_objective(model.logdet_R, ())
        self.add_objective(model.logdet_marg, ())
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'u': self.u, 'uprev': self.uprev, 'N': self.N,
                **super().variables(dvec)}


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i>=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(elem):
    ntril = len(elem)
    n = int(round(0.5*(np.sqrt(8*ntril + 1) - 1)))
    mat = np.zeros((n, n), dtype=elem.dtype)
    for ind, val in zip(tril_ind(n), elem):
        mat[ind] = val
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
    #y[:, :] += y_peak_to_peak[:] * 1e-3 * np.random.randn(N, 2)
    
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
    sQ0 = np.eye(nx) * 1
    sR0 = np.eye(ny) * 1e-1
    Q0 = sQ0 @ sQ0.T
    R0 = sR0 @ sR0.T
    A0 = np.eye(nx)
    C = np.eye(nx)
    
    vn0 = np.zeros((N, ny))
    x0 = y - vn0 @ sR0.T
    wn0 = (x0[1:] - x0[:-1] @ A0.T) @ np.linalg.inv(sQ0.T)
    
    Pp0 = scipy.linalg.solve_discrete_are(A0.T, C.T, Q0, R0)
    Rp0 = C @ Pp0 @ C.T + R0
    iRp0 = np.linalg.inv(Rp0)
    K0 = Pp0 @ C.T @ iRp0
    Pc0 = (np.eye(nx) - K0 @ C) @ Pp0
    sPp0 = np.linalg.cholesky(Pp0)
    sPc0 = np.linalg.cholesky(Pc0)
    
    z = np.zeros_like(A0)
    pred_mat = np.block([[A0 @ sPc0, sQ0],
                         [sPc0,      z]])
    q, r = np.linalg.qr(pred_mat.T)
    s = np.sign(r.diagonal())
    pred_orth0 = (q * s).T
    sPr0 = (r.T * s)[nx:, nx:]
    Mn0 = (r.T * s)[nx:, :nx]

    z = np.zeros_like(K0)
    corr_mat = np.block([[sR0, C @ sPp0],
                         [z,   sPp0]])
    q, r = np.linalg.qr(corr_mat.T)
    s = np.sign(r.diagonal())
    corr_orth0 = (q * s).T
    sRp0 = (r.T * s)[:ny, :ny]
    Kn0 = (r.T * s)[ny:, :ny]
    
    # Set initial guess
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = A0
    var0['B'][:] = np.zeros((nx, nu))
    var0['C'][:] = C
    var0['D'][:] = 0
    var0['Kn'][:] = Kn0
    var0['Mn'][:] = Mn0
    var0['x'][:] = x0
    var0['vn'][:] = vn0
    var0['wn'][:] = wn0
    var0['sQ_tril'][:] = sQ0[np.tril_indices(nx)]
    var0['sR_tril'][:] = sR0[np.tril_indices(ny)]
    var0['sRp_tril'][:] = sRp0[np.tril_indices(ny)]
    var0['sPp_tril'][:] = sPp0[np.tril_indices(nx)]
    var0['sPc_tril'][:] = sPc0[np.tril_indices(nx)]
    var0['sPr_tril'][:] = sPr0[np.tril_indices(nx)]
    var0['pred_orth'][:] = pred_orth0
    var0['corr_orth'][:] = corr_orth0

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
    var_L['sPc_tril'][tril_diag(nx)] = 1e-10
    var_L['sPp_tril'][tril_diag(nx)] = 1e-10
    var_L['sPr_tril'][tril_diag(nx)] = 1e-10
    var_L['sRp_tril'][tril_diag(ny)] = 0
    
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
    var_constr_scale['pred_cov'][:] = 1e2
    var_constr_scale['corr_cov'][:] = 1e2
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['sR_tril'][:] = 1e2
    var_scale['sQ_tril'][:] = 1e2
    var_scale['sRp_tril'][:] = 1e2
    var_scale['sPp_tril'][:] = 1e2
    var_scale['sPc_tril'][:] = 1e2
    var_scale['sPr_tril'][:] = 1e2
    
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
    Kn = opt['Kn']
    Mn = opt['Mn']
    ybias = opt['ybias']
    pred_orth = opt['pred_orth']
    corr_orth = opt['corr_orth']
    sRp = tril_mat(opt['sRp_tril'])
    sPp = tril_mat(opt['sPp_tril'])
    sPc = tril_mat(opt['sPc_tril'])
    sPr = tril_mat(opt['sPr_tril'])
    sQ = tril_mat(opt['sQ_tril'])
    sR = tril_mat(opt['sR_tril'])
    
    yopt = xopt @ C.T + u @ D.T + ybias
    Pc = sPc @ sPc.T
    Pp = sPp @ sPp.T
    Pr = sPr @ sPr.T
    Rp = sRp @ sRp.T
    Q = sQ @ sQ.T
    R = sR @ sR.T

    Ac = scipy.linalg.logm(A) / model.dt
