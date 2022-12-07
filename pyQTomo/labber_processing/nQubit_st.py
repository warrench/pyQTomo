import numpy as np
import itertools as it
import pyQTomo.tomo_functions.statetomography as st
import pyQTomo.utils.pulse_schemes as ps
import pyQTomo.utils.fitting_functions as models
from pyQTomo.utils.cholesky import OpfromChol_nQB
import matplotlib.pyplot as plt
import Labber
import scipy.optimize as opt


class nQubitStateTomography(object):
    """
    Class that implements n-qubit state tomography
    """

    def __init__(self, datafile, nQubit, betasfile=None, CF_file=None, QPT=False, index=None):
        self.datafile = datafile
        self.betafile = betasfile
        self.CF_file = CF_file
        self.nQubit = nQubit
        self.QPT = QPT

        self.dataLog = Labber.LogFile(self.datafile)
        if self.betafile is not None:
            self.betaLog = Labber.LogFile(self.betafile)
        if self.CF_file is not None:
            self.CF_log = Labber.LogFile(self.CF_file)

        self.LogChannel = None

        for channel in self.dataLog.getLogChannels():
            if 'Average state vector' in channel['name']:
                self.LogChannel = channel['name']

        self.data = self.dataLog.getData(self.LogChannel)
        # Figure out if it is QPT data or QST data

        if self.QPT:
            #Then it is QPT data in shape (nPrep, nMeas, prob)
            self.data = np.reshape(self.data, (int(4**nQubit),
                                              int(3**nQubit),
                                              -1))
        else:
            print(self.data.shape)
            if self.data.shape[0] > int(3**self.nQubit):
                self.data = np.reshape(self.data, (int(self.data.shape[0]//int(3**nQubit)), int(3**nQubit), -1))
                if index is not None:
                    self.data = self.data[index, :, :]
                else:
                    self.data = self.data[0, :, :]
            else:
                pass

        self.betas = None
        self.CF_matrix = None
        self.pulse_scheme = ps.nQubit_Meas(self.nQubit)

    def getBetas(self, verbose=False):
        betas = [np.zeros((2,2)) for j in range(self.nQubit)]
        betas_fit_results = [[None]*2 for j in range(self.nQubit)]

        if (self.betas is None) and (self.betafile is not None) and (self.CF_file is None):
            channels = []
            for channel in self.betaLog.getLogChannels():
                if 'Population' in channel['name']:
                    channels.append(channel['name'])
            grouped_channels = [list(i) for j, i in it.groupby(channels,
                                lambda x: x.split(' - ')[-1].split(' ')[1])]
            xname = self.betaLog.getStepChannels()[0]['name']
            xdata = self.betaLog.getStepChannels()[0]['values']

            fitModel = models.CosineModel()

            for i in range(self.nQubit):
                chan = grouped_channels[i]
                # print(chan)
                for j in range(2):
                    ydata = self.betaLog.getData(chan[j]).flatten()
                    params = fitModel.guess(ydata, xdata, freq=0.5, phi=-np.pi*j)
                    res = fitModel.fit(ydata, params, x=xdata)
                    if verbose:
                        # print(res.best_values)
                        res.plot_fit()
                        plt.show()
                    betas_fit_results[i][j] = res
                    betas[i][j][0] = res.best_values['constant']
                    betas[i][j][1] = ((-1)**j) * res.best_values['amplitude']
                self.betas = betas
                self.betas_fit_results = betas_fit_results
        else:
            beta = 0.5*np.ones((2,2))
            beta[1,1] = -0.5
            betas = [beta for j in range(self.nQubit)]
            self.betas = betas
            self.betas_fit_results = 'Default Beta Behaviour'
        if verbose:
            # pass
            return self.betas, self.betas_fit_results
        else:
            return self.betas

    def getCF_matrix(self):
        cf_channel = None
        for channel in self.dataLog.getLogChannels():
            if 'Average state vector' in channel['name']:
                cf_channel = channel['name']
        cf_mat = self.CF_log.getData(cf_channel)
        cf_mat = np.reshape(cf_mat, (-1, int(2**self.nQubit), int(2**self.nQubit)))
        cf_mat = np.mean(cf_mat, axis=0).transpose()
        self.CF_matrix = cf_mat
        return self.CF_matrix


    def getDMs(self, QPT_idx=0, bootstrap=False, n=1000):
        if self.QPT:
            tomo_data = self.data[QPT_idx, :, :]
        else:
            tomo_data = self.data
            if self.CF_matrix is not None:
                tomo_data = self._apply_mitigation(tomo_data, self.CF_matrix)

        
        if bootstrap:
            new_data = self.bootstrap(shots=n, QPT_idx=0)
            if self.CF_matrix is not None:
                new_data = self._apply_mitigation(new_data, self.CF_matrix)
            t = st.MLE_QST(new_data, self.betas, self.pulse_scheme, self.nQubit)
        else:
            t = st.MLE_QST(tomo_data, self.betas, self.pulse_scheme, self.nQubit)
        rho = OpfromChol_nQB(t)
        return rho

    def bootstrap(self, shots=1000, QPT_idx=0):
        if self.QPT:
            return NotImplementedError
        else:
            tomo_data = self.data
            shape = tomo_data.shape
            new_data = np.zeros(shape)
            for i in range(shape[0]):
                pvals = self.data[i, :]
                new_data[i, :] = np.random.multinomial(shots, pvals)/shots
        return new_data

    def _apply_mitigation(self, data, calib, method='ls'):
        data_shape = data.shape
        data_mitigated=np.zeros(data_shape)
        if method=='inv':
            inv_calib = np.linalg.inv(calib)
            data_mitigated = np.dot(inv_calib, data)
        elif method=='ls':
            for i in range(data_shape[0]):
                def fun(x):
                    return np.sum((data[i, :] - np.dot(calib, x))**2)
                x0 = np.random.rand(data_shape[-1])
                x0 = x0 / np.sum(x0)
                cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
                bnds = tuple((0,1) for x in x0)
                res = opt.minimize(fun, x0, method='SLSQP',
                                constraints=cons,
                                bounds=bnds, tol=1e-6)
                data_mitigated[i, :] = res.x
        else:
            data_mitigated = data
        return data_mitigated


