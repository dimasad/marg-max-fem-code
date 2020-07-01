"""Marginalization by maximization filter-error method problems."""


import numpy as np

from ceacoest import optim


class BaseProblem(optim.Problem):
    
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
