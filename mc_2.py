"""Monte Carlo experiment estimation.

Decision variables: x, wn, vn.
Steady-state KF and marginalization gains obtained with equality constraints.
Cholesky square roots used.
Balanced realization.

"""


import argparse
import importlib
import os
import pathlib

import numpy as np
import scipy.io
import scipy.linalg
import sympy
import sym2num.model
from ceacoest import optim
from ceacoest.modelling import symoptim


class Model(symoptim.Model):
    
    generated_name = 'Model'
    version = '1'

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
        v['sW_diag'] = [f'sW{i}' for i in range(nx)]
        po = [[f'pred_orth{i}_{j}' for j in range(2*nx)] for i in range(2*nx)]
        co = [[f'corr_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx+ny)]
        ko = [[f'ctrl_orth{i}_{j}' for j in range(nx+nu)] for i in range(nx)]
        oo = [[f'obs_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx)]
        v['pred_orth'] = po
        v['corr_orth'] = co
        v['ctrl_orth'] = ko
        v['obs_orth'] = oo
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
        self.add_constraint('ctrl_gram')
        self.add_constraint('obs_gram')
        self.add_constraint('ctrl_orthogonality')
        self.add_constraint('obs_orthogonality')
        self.add_objective('wnmerit')
        self.add_objective('vnmerit')
        self.add_objective('logdet_Q')
        self.add_objective('logdet_R')
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
    
    def ctrl_gram(self, sW_diag, A, B, ctrl_orth):
        bmat = np.block([A * sW_diag, B])
        return sW_diag[:, None] * ctrl_orth - bmat
    
    def obs_gram(self, sW_diag, A, C, obs_orth):
        bmat = np.block([A.T * sW_diag, C.T])
        return sW_diag[:, None] * obs_orth - bmat
    
    def ctrl_orthogonality(self, ctrl_orth):
        resid = 0.5 * (ctrl_orth @ ctrl_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def obs_orthogonality(self, obs_orth):
        resid = 0.5 * (obs_orth @ obs_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
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
    
    def logdet_marg(self, sPc_tril, sPr_tril, N):
        """Merit correction of state marginalization."""
        sPc = tril_mat(sPc_tril)
        sPr = tril_mat(sPr_tril)
        log_det_sPc = sum(sympy.log(d) for d in sPc.diagonal())
        log_det_sPr = sum(sympy.log(d) for d in sPr.diagonal())
        return  (N - 1) * log_det_sPr + log_det_sPc
    
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
        self.add_decision('sQ_tril', ntx)
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
        self.add_decision('Mn', (nx, nx))
        self.add_decision('vn', (N, ny))
        self.add_decision('wn', (N - 1, nx))
        self.add_decision('pred_orth', (2*nx, 2*nx))
        self.add_decision('corr_orth', (nx + ny, nx + ny))
        self.add_decision('sW_diag', nx)
        self.add_decision('ctrl_orth', (nx, nx + nu))
        self.add_decision('obs_orth', (nx, nx + ny))
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
        self.add_constraint(model.ctrl_gram, (nx, nx + nu))
        self.add_constraint(model.obs_gram, (nx, nx + ny))
        self.add_constraint(model.ctrl_orthogonality, ntx)
        self.add_constraint(model.obs_orthogonality, ntx)
        self.add_objective(model.wnmerit, N - 1)
        self.add_objective(model.vnmerit, N)
        self.add_objective(model.logdet_Q, ())
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


def estimate(model, datafile, matlab_est):
    uv, yv, ue, ye = load_data(datafile)
    problem = Problem(model, ye, ue)
    
    N = len(ye)
    nx = model.nx
    nu = model.nu
    ny = model.ny
    
    # Load initial guess
    guess = matlab_est['guess'][0,0]
    A0 = guess['A']
    B0 = guess['B']
    C0 = guess['C']
    D0 = guess['D']
    W0 = np.diag(guess['gram'].ravel())

    x0, e0, yp0 = predict(guess, ye, ue)
    w0 = x0[1:] - x0[:-1] @ A0.T - ue[:-1] @ B0.T
    v0 = ye - x0 @ C0.T - ue @ D0.T

    R0 = 1 / N * e0.T @ e0 
    Q0 = 1 / N * w0.T @ w0 + 0.1 * np.eye(nx)    
    sR0 = np.linalg.cholesky(R0)
    sQ0 = np.linalg.cholesky(Q0)
    sW0 = np.sqrt(W0)
    
    wn0 = w0 @ np.linalg.inv(sQ0.T)
    vn0 = v0 @ np.linalg.inv(sR0.T)
    
    Pp0 = scipy.linalg.solve_discrete_are(A0.T, C0.T, Q0, R0)
    Rp0 = C0 @ Pp0 @ C0.T + R0
    iRp0 = np.linalg.inv(Rp0)
    K0 = Pp0 @ C0.T @ iRp0
    Pc0 = (np.eye(nx) - K0 @ C0) @ Pp0
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
    corr_mat = np.block([[sR0, C0 @ sPp0],
                         [z,   sPp0]])
    q, r = np.linalg.qr(corr_mat.T)
    s = np.sign(r.diagonal())
    corr_orth0 = (q * s).T
    sRp0 = (r.T * s)[:ny, :ny]
    Kn0 = (r.T * s)[ny:, :ny]

    assert np.all(np.isfinite(W0))    
    ctrl_bmat = np.c_[A0 @ sW0, B0]
    q, r = np.linalg.qr(ctrl_bmat.T)
    ctrl_orth0 = (q * np.diag(np.sign(r))).T
    
    obs_bmat = np.c_[A0.T @ sW0, C0.T]
    q, r = np.linalg.qr(obs_bmat.T)
    obs_orth0 = (q * np.diag(np.sign(r))).T
    
    # Set initial guess
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = A0
    var0['B'][:] = B0
    var0['C'][:] = C0
    var0['D'][:] = D0
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
    var0['sW_diag'][:] = np.diag(sW0)
    var0['ctrl_orth'][:] = ctrl_orth0
    var0['obs_orth'][:] = obs_orth0
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['ybias'][:] = 0
    var_U['ybias'][:] = 0
    var_L['sQ_tril'][tril_diag(nx)] = 1e-10
    var_L['sR_tril'][tril_diag(ny)] = 1e-4
    var_L['sPc_tril'][tril_diag(nx)] = 1e-10
    var_L['sPp_tril'][tril_diag(nx)] = 1e-10
    var_L['sPr_tril'][tril_diag(nx)] = 1e-10
    var_L['sRp_tril'][tril_diag(ny)] = 0
    var_L['sW_diag'][:] = 0
    
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
    var_constr_scale['pred_cov'][:] = 1e1
    var_constr_scale['corr_cov'][:] = 1e1
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['sR_tril'][:] = 1e1
    var_scale['sQ_tril'][:] = 1e1
    var_scale['sRp_tril'][:] = 1e1
    var_scale['sPp_tril'][:] = 1e1
    var_scale['sPc_tril'][:] = 1e1
    var_scale['sPr_tril'][:] = 1e1
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 100.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 500)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)

    sRp = tril_mat(opt['sRp_tril'])
    isRp = np.linalg.inv(sRp)
    Kn = opt['Kn']
    A = opt['A']
    
    opt['L'] = A @ Kn @ isRp
    opt['status'] = info['status']
    return opt


def load_data(datafile):
    data = scipy.io.loadmat(datafile)
    
    # Retrieve data
    u = data['u']
    y = data['y']
    
    N = len(y)
    ue = u[N//2:]
    ye = y[N//2:]
    uv = u[:N//2]
    yv = y[:N//2]
    return uv, yv, ue, ye


def load_matlab_estimates(datafile):
    estfile = datafile.parent / ('estim_' + datafile.name)
    return scipy.io.loadmat(estfile)


def predict(mdl, y, u):
    A = mdl['A']
    B = mdl['B']
    C = mdl['C']
    D = mdl['D']
    L = mdl['L']
    
    nx = len(A)
    N = len(y)
    try:
        x0 = mdl['x0'].ravel()
    except (KeyError, ValueError):
        x0 = np.zeros(nx)
    
    x = np.tile(x0, (N, 1))
    yp = np.empty_like(y)
    e = np.empty_like(y)
    
    for k in range(N):
        yp[k] = C @ x[k] + D @ u[k]
        e[k] = y[k] - yp[k]
        if k+1 < N:
            x[k+1] = A @ x[k] + B @ u[k] + L @ e[k]
    return x, e, yp


def get_model(config):
    nx = config['nx']
    nu = config['nu']
    ny = config['ny']
    
    script_name = pathlib.Path(__file__).stem
    modname = f'generated_{script_name}_v{Model.version}_nx{nx}_nu{nu}_ny{ny}'
    try:
        mod = importlib.import_module(modname)        
        cls = getattr(mod, Model.generated_name)
        return cls()
    except ImportError:
        symmodel = Model(nx=nx, nu=nu, ny=ny)
        with open(f'{modname}.py', mode='w') as f:
            print(symmodel.print_code(), file=f)
        return get_model(config)


def cmdline_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'edir', nargs='?', default='data/mc', 
        help='experiment directory',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = cmdline_args()
    edir = pathlib.Path(args.edir)
    config = scipy.io.loadmat(edir / 'config.mat', squeeze_me=True)
    suffix = pathlib.Path(__file__).stem[3:]
        
    model = get_model(config)
    datafiles = sorted(edir.glob('exp*.mat'))
    
    msefile = edir / f'val_mse_{suffix}.txt'
    open(msefile, 'w').close()
    
    for datafile in datafiles:
        i = int(datafile.stem[3:])
        print('_' * 80)
        print('Experiment #', i, sep='')
        print('=' * 80)
        
        uv, yv, ue, ye = load_data(datafile)
        matlab_est = load_matlab_estimates(datafile)
        
        opt = estimate(model, datafile, matlab_est)
        
        savekeys = {
            'A', 'B', 'C', 'D', 'L', 'sW_diag', 'Kn',
            'sRp_tril', 'sQ_tril', 'sR_tril', 'sPp_tril', 'status',
        }
        optsave = {k:v for k,v in opt.items() if k in savekeys}
        np.savez(edir / (f'ml_{suffix}' + datafile.stem), **optsave)
        
        xopt, eopt, ypopt = predict(opt, yv, uv)
        esys = [predict(mdl, yv, uv)[1] for mdl in matlab_est['sys'].flat]
        
        yvdev = yv - np.mean(yv, 0)
        mse = [np.mean(e**2) for e in (yv, eopt, *esys)]
        with open(msefile, 'a') as f:
            print(i, *mse, sep=', ', file=f)

