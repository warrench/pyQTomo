import numpy as np

def OpfromChol_nQB(t, QPT=False):
    """Function that takes guess of Cholesky decomposition of a Hermitian
    operator, and return the d^n x d^n operator

    The Cholesky decomposition for an n-qubit operator is:
    op = (T.dag() * T)/(Tr(T.dag() * T))
    where T is upper triangular matrix of the form
    T = [[  t_0                ,          0         ,          ...        ,  0],
         [  t_2^n + it_2^n+1   ,         t_1        ,          ...        ,  0],
         [        ...          ,         ...        ,          ...        ,  0],
         [  t_4^n-2 + it_4^n-1 , t_4^n-4 + it_4^n-3 , t_4^n-6 + it_4^n-5  , t_2^n-1]

    For Quantum State Tomography (QST), the returned operator is a density
    matrix of size 2^n * 2^n

    For Quantum Process Tomography (QPT), the returned operator is the Chi
    Process Matrix of size 4^n x 4^n

    Parameters
    -----------
    t : array
        length = (d)^2n, containing (real) values of the t's where for
        QST d=2 and for QPT d=4

    Returns
    ---------
    op_t: array
        size = (d^n x d^n) containing the operator that has been reconstructed
        from the Cholesky decomposition. Either the density matrix for
        QST, or Chi matrix for QPT
    """
    if QPT:
        n = int(np.log2(np.sqrt(np.sqrt(len(t)))))
        d = 4
    else:
        n = int(np.log2(np.sqrt(len(t))))
        d = 2
    T = np.zeros((d**n, d**n)) + 0j
    main_diag = np.diag(t[0:int(d**n)])

    even = t[int(d**n)::2]
    odd = t[int(d**n+1)::2]

    off = even + 1j*odd
    T += main_diag

    for i in range(1, int(d**n)):
        diag = np.diag(off[int((i-1)*d**n-(i-1)*i//2):int((i)*d**n-i*(i+1)//2)], k=-i)
        T += diag
    T = np.matrix(T)
    norm = np.array(T.H.dot(T).trace()).flatten()[0].real
    op_t = (T.H.dot(T))/norm
    op_t = np.array(op_t)
    return op_t