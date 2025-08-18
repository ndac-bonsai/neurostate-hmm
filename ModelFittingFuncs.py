
import numpy as np
import pickle
import ssm
from sklearn.decomposition import PCA

def extract_features(data_filename,low_freq_bound, high_freq_bound, FFT_lower_bound, FFT_upper_bound, buffer_size, obsdim):
    

    # Extract data from .bin file
    train_data = np.fromfile(data_filename, dtype=np.float32)

    # Reshape data into seq_len x buffer length
    train_data = np.reshape(train_data[:int(len(train_data)/buffer_size) * buffer_size], (-1, buffer_size))
    sequences, _ = train_data.shape

    # Z-score data
    mean = np.mean(train_data, axis=0, keepdims=True)
    stdev = np.std(train_data, axis=0, keepdims=True)
    train_z_scores = (train_data - mean)/stdev

    '''
    Consider only frequencies within the provided range [FFT_lower_bound, FFT_upper_bound]. Each index of the FFT vector does not necessarily
    respond to a specific frequency. The FFT is binned with index 0 starting at 0 Hz and the final index ending at the sampling frequency/2.
    To only grab a certain frequency range, a scaling factor is calculated to convert a frequency into its corresponding index in 
    the FFT vector. The frequency bounds are then converted into corresponding FFT vector indices by dividing by the scaling factor
    '''
    scaling_factor = (FFT_upper_bound-FFT_lower_bound)/(buffer_size-1)
    freqs = np.arange(0,buffer_size,1) * scaling_factor
    train_z_scores = train_z_scores[:,(freqs >= low_freq_bound) & (freqs <= high_freq_bound)]
    

    # Fit PCA and calculate features
    pca = PCA(n_components=obsdim)
    train_features = pca.fit(train_z_scores)
    pca_components = pca.components_.T
    pca_components = pca_components / np.linalg.norm(pca_components,ord=1,axis=0)
    train_features = train_z_scores.dot(pca_components)
    pca.components_ = pca_components.T

    return pca, train_features, mean, stdev

def fit_model(numStates,numSubstates,obsdim,features,N_em_iters):
    if numSubstates == 1:
        model, lls = fit_HMM(numStates,obsdim,features,N_em_iters)
    elif numSubstates > 1:
        model, lls = fit_HSMM(numStates,numSubstates,obsdim,features,N_em_iters)
    else:
        raise Exception("Number of substates must be positive integer")
    
    return model, lls


def fit_HSMM(numStates,numSubstates,obsdim,features,N_em_iters):
    print("Fitting Gaussian HSMM with EM")
    transition_kwargs = {'r_min': numSubstates, 'r_max': numSubstates}
    hsmm = ssm.HSMM(numStates,obsdim,observations="gaussian",transition_kwargs=transition_kwargs)
    hsmm.initialize(features)
    hsmm_em_lls = hsmm.fit(features, method="em", initialize=False, num_iters=N_em_iters)
    return hsmm, hsmm_em_lls

def fit_HMM(numStates,obsdim,features,N_em_iters):
    print("Fitting Gaussian HMM with EM")
    hmm = ssm.HMM(numStates,obsdim,observations="gaussian")
    hmm.initialize(features)
    hmm_em_lls = hmm.fit(features,method="em", initialize=False, num_iters=N_em_iters)
    return hmm, hmm_em_lls


