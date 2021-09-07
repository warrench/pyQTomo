import numpy as np
import itertools as it
from scipy.optimize import minimize
from collections import Counter
import qutip as qt


import pyQTomo.utils.pulse_schemes as ps
from pyQTomo.utils.cholesky import OpfromChol_nQB
from pyQTomo.utils.constraints import PSDconstraint


"""
Code based off of an implementation of single and two-qubit implementation by
Morten Kjaergaard (mkjaergaard@nbi.ku.dk) during his postdoctoral research at
Engineering Quantum Systems group at MIT (2016-2020)

This has been generalized to n-qubit quantum state tomography by Christopher
Warren (warrenc@chalmers.se) during his PhD research at Quantum Technology
Laboratory at Chalmers University of Technology (2019-2024)
"""


#===============================================================================
#================================ MLE Wrapper ==================================
#===============================================================================

def MLE_QST(probvectors, betas, pulse_scheme, n, verbose=False):
    """Maximum likelihood estimation for n qubit state tomography

    Detailed description of the MLE process:

    The expectation value of m_<I..I>, ... , m_<A..A> for a 2^n x 2^n matrix
    constructed as the tensor product of two paulis, can be found via:
    [m_<I..I>, ... , m_<A...A>] = betas^{-1}*[p_0...0^A, ... , p_1...1^A]
    and the objective function L to be minimized to enforce the PSD is given by:
    L = sum_{A}(m_<A> - Tr(A*rho_T) )**2
    where rho_T is the cholesky decomposition of a 2^n x 2^n matrix, and A is the
    tensor product of n pauli matrices. E.g. for 3 qubit tomography
    A = [IIX,
         IIY,
         IIZ,
         ...,
         ZZZ]
    where the tensor product is implicity between each Pauli. The corresponding
    pulse sequence to get the above list of A's is:
    Pulses = [XXX,
              XXY,
              XXZ,
              XYX,
              ...,
              ZZZ]
    In total there are 3^n measurement settings corresponding to all combinations
    of [X, Y, Z]

    If this pulse scheme is generated from the pulse scheme and ran in that order
    in Labber it will produce the proper output. Not tested for other pulse schemes

    Parameters
    ----------
    probvectors : array
        Array of array of probabilities from the Average State Vector of Labber
        for each of the n qubit states.
    betas: Array
        Array of arrays of betas for each qubit. Each beta corresponds to a matrix
        of the form
                    beta[0][0] = beta_I for Q1_|0>
                    beta[0][1] = beta_Z for Q1_|0>
                    beta[1][0] = beta_I for Q1_|1>
                    beta[1][1] = beta_Z for Q1_|1>
    pulse_scheme : Array of strings
        The pulse scheme used to generate the data

    n: int
        The number of qubits to find the density matrix of

    verbose: bool
        If true will return not just the minimal t values, but also more results
        from the minimizer

    Returns
    -------
    Array of t's
        Array of the 4^n t's that minimize the objective function

    """
    t_guess = np.ones(int(4**n))/4**n
    consts = ({'type': 'eq',
               'method': 'BFGS',
               'fun': lambda t: PSDconstraint(t)})

    measuredEvals = []
    Paulis = []
    for j, pulse in enumerate(pulse_scheme):
        sub_Paulis, seqs = getPaulisfromPulses_QST(pulse)
        Paulis.append(sub_Paulis)
        measuredEvals.append(getEvalsfromProbs_nQB(seqs,
                                                    probvectors[j],
                                                    betas))

    measuredEvals_flattened = np.array(list(it.chain(*measuredEvals))
                                        ).flatten()
    Paulis_flattened = np.array(list(it.chain(*Paulis)))

    if n > 1:
        Evals_reduced, Paulis_reduced = pop_redundant(measuredEvals_flattened,
                                                      Paulis_flattened)
    else:
        Evals_reduced = measuredEvals_flattened
        Paulis_reduced = Paulis_flattened


    result = minimize(MLE_Functional_QST,
                      t_guess,
                      args=(Evals_reduced, Paulis_reduced),
                      constraints=consts,
                      method='SLSQP',
                      tol=1e-15)
    if verbose:
        return result

    return result['x']

#===============================================================================
#======================= Maximum-Likelihood Functionals ========================
#===============================================================================


def MLE_Functional_QST(t, measuredEvals, Paulis):
    """The functional minimized for n-qubit density matrix

    Parameters
    ----------
    t : array
        List of 4^n floats, corresponding to each t_i
    measuredEvals : array
        Array of 4^n-1 floats, corresponding to measured expectation values of
        various Pauli operators
    Paulis : array
        Array of 4^n-1 2^n x 2^n Pauli matrices, corresponding to the measured
        expectation values

    Returns
    -------
    L : Float
        The number to minimized in the maximum likelihood minimizer
    """

    rho_reconstructed = OpfromChol_nQB(t)
    expect = np.einsum('ij,ljk', rho_reconstructed, Paulis) # perform matrix multiplication over list of paulis
    tr_expect = np.einsum('iij', expect).real # take the real component of the trace

    L = np.sum((measuredEvals - tr_expect)**2)

    return L

#===============================================================================
#================================ QST Functions ================================
#===============================================================================

def getPaulisfromPulses_QST(pulse):
    """Returns the 2^n x 2^n matrix from the tensor product between n Paulis

    Parameters
    ----------
    pulse : (tuple)

    Returns
    -------
    2^n x 2^n matrix
    """
    PaulisFromPulses = []
    pulse_set = set(pulse)

    contains_dup = len(pulse) != len(pulse_set)

    if contains_dup:
        # n_paulis = Counter(pulse).items()
        pulse_idxs = {}
        for key, val in Counter(pulse).items():
            pulse_idxs[key] = [i for i, x in enumerate(pulse) if x==key]

        #Create a container to store the boken down pulse sequences
        seqs = []
        for key, val in pulse_idxs.items():
            repl_seq = ['I', key]
            new_seqs = list(it.product(repl_seq, repeat=len(val)))
            # Pop out the first entry which is <I...I>
            new_seqs = new_seqs[1:]
            seqs.append(new_seqs)
        seq_list = list(it.product(*seqs))

        #Reuse container name to hold the reconstructed pauli operators
        seqs = []
        for sequence in seq_list:
            temp = [None]*len(pulse)
            for i, (key, idx_mask) in enumerate(pulse_idxs.items()):
                for k, idx in enumerate(idx_mask):
                    temp[idx] = sequence[i][k]
            seqs.append(temp)
        PaulisFromPulses = ps.pulseToPauli_nQb(seqs)


    else:
        PaulisFromPulses= ps.pulseToPauli_nQb(pulse, dupe=False)
        seqs = pulse

    return PaulisFromPulses, seqs


def getEvalsfromProbs_nQB(pulses, probvector, betas):
    """Converts p0..0, p1..0, p0...1 and p1..1 probabilities into measurements of
    the Z paulis
     m_<II>, m_<ZI>, m_<IZ>, m_<ZZ> through solving the equation:
    probvector = betas * [m_<II>, m_<ZI>, m_<IZ>, m_<ZZ>]

    Parameters
    ----------
    probvector : 2^n x 1 array
        vector of the form [p_0...0, p_1...0, p_0...1, p_1...1]
    betas : n x (2x2) array
        beta parameters for each qubit

    Returns
    -------
    float or list of floats
        Returns either the float corresponding to the expectation value,
        or returns a list of three expectation values. This is identical to
        the structure of returns from getPaulisfromPulses_QST, see docstring
        for more information.

    """
    Beta = [qt.Qobj(beta) for beta in betas]
    Beta = np.array(qt.tensor(Beta))
    #Not a huge fan of matrix inversion maybe change this to a lst-sq fit
    ExpectVals = np.linalg.inv(Beta).dot(np.transpose(probvector)).flatten()
    # ExpectVals = np.array(ExpectVals).flatten()
    ExpectMask = list(it.product(range(2), repeat=len(betas)))
    #flip to the Labber convention for the average state vector
    for i, entry in enumerate(ExpectMask):
        ExpectMask[i] = list(entry)[::-1]
    ExpectMask = [''.join(str(i) for i in val) for val in ExpectMask]
    state_dict = {val: i for i, val in enumerate(ExpectMask)}

    pulse_mask = []

    for pulse in pulses:
        s = ''
        for val in pulse:
            if val=='I':
                s += '0'
            else:
                s += '1'
        pulse_mask.append(s)

    # if len(pulses)>1:
    if 'I' in list(it.chain(*pulses)):
        expectation_values = []
        for i, pulse in enumerate(pulses):
            key = pulse_mask[i]
            expectation_values.append(np.real(ExpectVals[state_dict[key]]))
    else:
        expectation_values = [np.real(ExpectVals[-1])]
    return expectation_values

def pop_redundant(expects, paulis):
    """Function to pop out redundant expectation values from the QST fitting.
    We measure expectation values of the form XXY and ZZY, in the case of n=3.
    Each of these would be decomposed and would have a redundant value of the
    form IIY. (I.e  XXY -> IIY, IXY, XIY, XXY  and ZZY -> IIY, IZY, ZIY, ZZY)
    We only need to keep non-redundant entries to fit over

    Parameters
    -----------
    expects: array
        List of expectation values generated from corresponding to the pulse
        sequence and measured probabilities

    paulis: array
        Array of all the n-Pauli matrices generated as a result of the pulse
        sequence.

    Returns
    --------
    unique_expects: array
        The array of expectation values without the redundant entries due to
        the decomposition of the pulse sequence. Matched to the index
        corresponding to the first time the reundant n-Pauli matrix appears

    unique_paulis: array
        The array of unique pauli matrices that are generated from the pulse
        sequence in the order that they appear first

    """
    unique_paulis, indexes = np.unique(paulis, return_index=True, axis=0)
    unique_expects = np.zeros(len(unique_paulis))

    for i, idx in enumerate(indexes):
        unique_expects[i] = expects[idx]

    return unique_expects, unique_paulis
