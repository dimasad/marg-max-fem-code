"""Tests with the ATTAS aircraft short-period mode estimation."""


import importlib
import os

import numpy as np
import scipy.io
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
    ushift = np.r_[0.04]
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

    
