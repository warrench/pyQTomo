'''
 ----------------------------------------------------
 -      Fitting models used for analysis            -
 ----------------------------------------------------

Originally developed by Andreas Bengtsson (andreas.bengtsson@chalmers.se)
then adapted to the ExQA state tomograpy suite
'''
import lmfit
import numpy as np


class CosineModel(lmfit.Model):
    """Class for fitting cosine-model with offset using lmfit models"""

    def __init__(self, *args, **kwargs):
        # Inherit from the lmfit cosine model
        super().__init__(cosine, *args, **kwargs)  # note that

    def guess(self, data, x, **kwargs):
        """Tweaked method for guessing parameters of initial fit to cosine

        Parameters
        ----------
        data : array
            Data to be fitted (stored in an array)
        x : array
            x-values to be fitted over
        **kwargs :
            Keyword arguments used for lmfit

        Returns
        -------
        type
            Returns updated guess of fitting parameters

        """
        amp_guess = abs(max(data) - min(data)) / 2
        freq_guess, ph_guess = fft_freq_phase_guess(data, x)
        params = self.make_params()

        def pset(param, value, min=-np.inf, max=np.inf):
            """Method for updating parameters in lmfit model

            Parameters
            ----------
            param : string
                Name of parameter to be updated
            value : float
                Value to set param to
            min : float
                minimum value
            max : float
                maximum value

            Returns
            -------
            type: lmfit.models
                Returns updated parameter values in the lmfit class

            """
            params["%s%s" % (self.prefix, param)].set(value=value,
                                                      min=min,
                                                      max=max)

        pset('amplitude', value=amp_guess, min=0)
        pset('frequency', value=freq_guess, min=0)
        pset('phase', value=ph_guess, min=-2 * np.pi, max=2 * np.pi)
        pset('constant', value=np.mean(data))
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


def fft_freq_phase_guess(data, t):
    """Function for generating a decent guess at freq and phase

    Parameters
    ----------
    data : array
        Data from which to guess frequency and phase
    t : array
        t's to be used for guessing phase

    Returns
    -------
    floats
        Returns frequency guess and phase guess

    """
    w = np.fft.fft(data)[:len(data) // 2]
    f = np.fft.fftfreq(len(data), t[1] - t[0])[:len(w)]
    w[0] = 0  # Remove DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_w = np.abs(w)
    freq_guess = abs(f[abs_w == max(abs_w)][0])
    ph_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess)[0]
    return freq_guess, ph_guess


def cosine(x, amplitude, frequency, phase, constant):
    """Method that generates cosine-functino with offset.

    Parameters
    ----------
    x : float or array
        x-value(s) at which to generate function
    amplitude : float
        Amplitude of cosine
    frequency : float
        Frequency of cosine
    phase : float
        Phase of cosine
    constant : float
        Constant offset of cosine

    Returns
    -------
    float or array:
        single-value or numpy array of cosine-function

    """
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + constant
