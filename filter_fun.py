import scipy.signal as signal
import numpy as np


derivative_coeff = [2 / 8, 1 / 8, 0, -1 / 8, -2 / 8]

def derivative_filter_data(data):
    filtered_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        filtered_data[:,i] = derivative_filter(data[:,i])
    return filtered_data

def derivative_filter(data):
    filtered_data = signal.lfilter(derivative_coeff, 1, data)
    return filtered_data

def apply_filter(b, a, data):
    filtered_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        filtered_data[:,i] = signal.lfilter(b, a, data[:,i])
    return filtered_data

def get_filter_taps(low_pass = None, high_pass = None, numtaps = 40, filtertype = 'fir'):
    if(filtertype == 'fir'):
        filter_fun = signal.firwin
    elif(filtertype == 'butter'):
        filter_fun = signal.butter
    filter_taps = None
    a = [1]
    if low_pass is not None and high_pass is not None:
        if(filtertype == 'fir'): filter_taps = filter_fun(numtaps, [low_pass, high_pass], window='hamming' , pass_zero=False, fs = 1)
        elif(filtertype == 'butter'): filter_taps, a = filter_fun(numtaps, [low_pass, high_pass], btype='bandpass', fs = 1)
    elif low_pass is not None:
        if(filtertype == 'fir'): filter_taps = filter_fun(numtaps, low_pass, window='hamming', fs = 1)
        elif(filtertype == 'butter'): filter_taps, a = filter_fun(numtaps, low_pass, btype='lowpass', fs = 1)
    elif high_pass is not None:
        if(filtertype == 'fir'): filter_taps = filter_fun(numtaps, high_pass, window='hamming' , pass_zero=False, fs = 1)
        elif(filtertype == 'butter'): filter_taps, a = filter_fun(numtaps, high_pass, btype='highpass', fs = 1)
    return filter_taps, a

def custom_filters(data, low_pass = None, high_pass = None, numtaps = 40, filtertype = 'fir'): 
    filtered_data = np.zeros(data.shape)
    filter_taps, a = get_filter_taps(low_pass, high_pass, numtaps, filtertype)
    if filter_taps is not None:
        filtered_data = apply_filter(filter_taps, a, data)
    else:
        print("Filter taps are none")
    return filtered_data