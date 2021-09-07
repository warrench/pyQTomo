import numpy as np

def PSDconstraint(t):
    """Function used for enforcing positive semidefinite property of
     density matrix in cholesky decomposition

    Ensures that Tr(rho) = 1, which is equivalent to sum(t_i**2) = 1.

    Parameters
    ----------
    t : array
        t_i parameters that go into Cholesky decomposition

    Returns
    -------
    float
        Function that is 0 when PSD constraint satisfied

    """

    return np.array((t[:]**2).sum()-1)


