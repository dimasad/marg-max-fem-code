"""Automatic differentiation utilities."""


import jax
import jax.numpy as jnp
import numpy as np


### Enable 64-bit in jax ###
jax.config.update("jax_enable_x64", True)


def dare(a, b, q, r):
    # Build the symplectic matrix
    a_inv = jnp.linalg.inv(a)
    b_r_inv_bT = b @ jnp.linalg.solve(r, b.T)
    z = jnp.block([[a + b_r_inv_bT @ a_inv.T @ q, -b_r_inv_bT @ a_inv.T],
                   [-a_inv.T @ q,                  a_inv.T]])
    
    # Obtain eigenvectors for its stable subspace
    lamb, v = eig(z)
    stab = jnp.abs(lamb) < 1.0
    n = len(a)
    assert jnp.sum(stab) == n
    
    # Build solution
    u1 = v[:n, stab]
    u2 = v[n:, stab]
    x = jnp.linalg.solve(u1.T, u2.T).T
    return 0.5 * jnp.real(x + x.conj().T)


@jax.custom_jvp
def eig(A):
    return jnp.linalg.eig(A)


@eig.defjvp
def eig_jvp(primals, tangents):
    A, = primals
    A_dot, = tangents
    
    D, U = eig(A)
    F = jnp.zeros_like(U)
    for i, j in np.ndindex(*F.shape):
        if i != j:
            assert jnp.abs(D[j] - D[i]) > 1e-16
            F = F.at[i,j].set(1 / (D[j] - D[i]))
    
    Uinv_A_dot_U = jnp.linalg.solve(U, A_dot) @ U
    D_dot = jnp.diag(Uinv_A_dot_U)
    U_dot = U @ (F * Uinv_A_dot_U)
    
    return (D, U), (D_dot, U_dot)

