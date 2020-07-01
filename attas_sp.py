"""Tests with the ATTAS aircraft short-period mode estimation."""


import importlib
import os

import numpy as np
import scipy.io
import scipy.linalg
import sympy
import sym2num.model

import mmfem
import symmmfem


# Reload modules for testing
for m in (mmfem, symmmfem):
    importlib.reload(m)


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
    #y[:, 0] += y_peak_to_peak[0] * 1e-3 * np.random.randn(N)
    
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


tril_diag = symmmfem.tril_diag
tril_mat = symmmfem.tril_mat


if __name__ == '__main__':
    nx = 2
    nu = 1
    ny = 2
    
    # Load experiment data
    t, u, y, yshift, yscale, ushift, uscale = load_data()
    t2, u2, y2 = load_data2(yshift, yscale, ushift, uscale)
    symmodel = symmmfem.BaseModel(nx=nx, nu=nu, ny=ny)
    model = symmodel.compile_class()()
    model.dt = t[1] - t[0]
    problem = mmfem.BaseProblem(model, y, u)
    
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
    var_L['sQ_tril'][tril_diag(nx)] = 1e-7
    var_L['sR_tril'][tril_diag(ny)] = 1e-7
    var_L['sPc_tril'][tril_diag(nx)] = 1e-7
    var_L['sPp_tril'][tril_diag(nx)] = 1e-7
    var_L['sPr_tril'][tril_diag(nx)] = 1e-7
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
    var_constr_scale['pred_cov'][:] = 10
    var_constr_scale['corr_cov'][:] = 10
    
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
