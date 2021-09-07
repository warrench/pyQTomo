"""
 ----------------------------------------------------
 - Common pulse schemes for n qubit tomo      -
 ----------------------------------------------------

Functions and definitions used for determining which pulse scheme was used
to do the tomography and convert those pulses into corresponding Pauli matrices
"""

import numpy as np
import itertools
import qutip as qt

# Initialize Pauli matrices in numpy format:
I = qt.qeye(2)
sz = qt.sigmaz()
sx = qt.sigmax()
sy = qt.sigmay()

# Create dictionary used for mapping post-pulsing into axis
dictPulseToPauli = {'Z': sz, 'Y': sy, 'X': sx, '-Y': -sy, '-X': -sx, 'I': I}
dictPulseToState = {'0': qt.basis(2,0),
                   '1': qt.basis(2,1),
                   '+': (1/np.sqrt(2))*(qt.basis(2,0) + qt.basis(2,1)),
                   '-': (1/np.sqrt(2))*(qt.basis(2,0) - qt.basis(2,1)),
                   '+i': (1/np.sqrt(2))*(qt.basis(2,0) + 1j*qt.basis(2,1)),
                   '-i': (1/np.sqrt(2))*(qt.basis(2,0) - 1j*qt.basis(2,1))}

def nQubit_Meas(n):
    """
    Generate a list of measurement operators correponding to the
    [X,Y,Z]^n Pauli group

    Input:
        n (int): Number of qubits to perform tomography over

    Returns:
        (list) list of measurement operators corresponding to all combinations
                of the n qubit Pauli group for QST or QPT
    """
    #Pulse in [X,Y,Z] order
    seq_single = ['X', 'Y', 'Z']
    return list(itertools.product(seq_single, repeat=n))

def nQubit_Prep(n, overdetermined=False):
    """
    Generate a list of preparation states corresponding to the
    axes of the Bloch sphere. For QPT it is enough to take
    all of the +ve eigenstates of the Bloch sphere
    and a single -ve eigenstate. For this reason we take
    [0, 1, +, +i] as our preparation basis. There is also
    the option to generate a list of all Bloch inputs

    Input:
        n (int): Number of qubits to perform tomography over

        overdetermined (boolean): Whether to prepare all input states

    Returns
        (list): List of all preparations for QPT protocol
    """
    seq_single = ['0', '1', '+', '+i']
    if overdetermined:
        seq_single = ['0', '1', '+', '-', '+i', '-i']
    return list(itertools.product(seq_single, repeat=n))

def pulseToPrep_nQb(pulse):
    """
    Helper function to map a list of pulses to
    its particular qutip value

    Input:
        pulse (list): List of pulses which contain the
        dictionary keys of a sequence

    Returns:
        seq (list): List of preparations mapped to the specific
                    basis state in qutip
    """
    seq = [dictPulseToState[p] for p in pulse]
    return seq

def pulseToPauli_nQb(pulses, dupe=True):
    """
    Helper function to map a list of Pauli operators
    to their particular matrix representation

    Returns:
        Paulis_nQB (list): List of pauli operators tensored
    """
    Paulis_nQB = []
    if dupe:
        for i, pulse in enumerate(pulses):
            seq = [dictPulseToPauli[p] for p in pulse]
            Paulis_nQB.append(np.matrix(qt.tensor(seq)))
    else:
        seq = [dictPulseToPauli[p] for p in pulses]
        Paulis_nQB.append(np.matrix(qt.tensor(seq)))

    return Paulis_nQB