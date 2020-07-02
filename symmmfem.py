"""Symbolic marginalization by maximization filter-error method models."""


import numpy as np

import sympy
from ceacoest.modelling import symoptim


class BaseModel(symoptim.Model):
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
