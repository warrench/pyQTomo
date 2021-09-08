# pyQTomo
A suite of tomographic reconstruction techniques for quantum information based on python.

Code based off of an implementation of single and two-qubit implementation by
Morten Kjaergaard (mkjaergaard@nbi.ku.dk) during his postdoctoral research at
Engineering Quantum Systems group at MIT (2016-2020)

This has been generalized to n-qubit quantum state tomography by Christopher
Warren (warrenc@chalmers.se) during his PhD research at Quantum Technology
Laboratory at Chalmers University of Technology (2019-2024)


# TODO List
1. Implement fitting routines for Quantum Process Tomography
2. Include error mitigation strategies for pre-processing data

Suggestions welcome

# Installation

1. Download the repository
2. perform a python setup using setup.py
`python setup.py install` or `python setup.py develop`
3. To use the Labber portion of the code you must install Labber separately. This does
not require a licsense, just access to the python API for parsing the log files. This can
be worked around by directly accessing the hdf5 file, however the API makes data handling
much simpler.
