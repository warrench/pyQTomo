import numpy as np
import itertools as it
import pyQTomo.tomo_functions.statetomography as st
import pyQTomo.utils.pulse_schemes as ps
import pyQTomo.utils.fitting_functions as models
from pyQTomo.utils.cholesky import OpfromChol_nQB
import Labber


class nQubitStateTomography(object):
    """
    Class that implements n-qubit state tomography
    """

    def __init__(self, datafile, betafile, nQubit, QPT=False):
        self.datafile = datafile
        self.betafile = betafile
        self.nQubit = nQubit
        self.QPT = QPT

        self.dataLog = Labber.LogFile(self.datafile)
        self.betaLog = Labber.LogFile(self.betafile)

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
            #It is QST data
            pass

        self.betas = None
        self.pulse_scheme = ps.nQubit_Meas(self.nQubit)

    def getBetas(self, verbose=False):
        betas = [np.zeros((2,2)) for j in range(self.nQubit)]
        betas_fit_results = [[None]*2 for j in range(self.nQubit)]

        if self.betas is None:
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
                for j in range(2):
                    ydata = self.betaLog.getData(chan[0]).flatten()
                    params = fitModel.guess(ydata, xdata)
                    res = fitModel.fit(ydata, params, x=xdata)
                    betas_fit_results[i][j] = res
                    betas[i][j][0] = res.best_values['constant']
                    betas[i][j][1] = ((-1)**j) * res.best_values['amplitude']
                self.betas = betas
                self.betas_fit_results = betas_fit_results
        if verbose:
            return self.betas, self.betas_fit_results
        else:
            return self.betas

    def getDMs(self, QPT_idx=0):
        if self.QPT:
            tomo_data = self.data[QPT_idx, :, :]
        else:
            tomo_data = self.data

        t = st.MLE_QST(tomo_data, self.betas, self.pulse_scheme, self.nQubit)
        rho = OpfromChol_nQB(t)
        return rho
