
import numpy as np
import pickle
import ssm
from ssm.messages import forward_pass
from ssm.util import replicate, collapse

# NOTE: this function accesses global Python variables: seq_len, feature_sequence, iteration
def decode_state(input_buffer, FFT_lower_bound, FFT_upper_bound, low_freq_bound, high_freq_bound, buffer_size):

    # Convert the input buffer into a numpy array
    input_buffer = [point for point in input_buffer]
    input_buffer = np.asarray(input_buffer)
    
    # Create a stack of buffers (observations) of length seq_len to feed into model
    feature_sequence[0:seq_len-1] = feature_sequence[1:, :] # Remove oldest observation
    feature_sequence[-1, :] = input_buffer # insert newest observation
    formatted_feature_sequence = feature_sequence[-iteration:,:] # Create array of length seq_len (or <seq_len if there have been fewer than seq_len observations)

    # Preprocess data with same steps as for model fitting/training
    z_scores = (formatted_feature_sequence - mean)/stdev
    scaling_factor = (FFT_upper_bound-FFT_lower_bound)/(buffer_size-1)
    freqs = np.arange(0,buffer_size,1) * scaling_factor
    z_scores = z_scores[:,(freqs >= low_freq_bound) & (freqs <= high_freq_bound)]

    pca_components = pca_features.components_.T
    features = z_scores.dot(pca_components)

    # Predict hidden state from sequence of features
    if isinstance(model,ssm.HSMM):
        belief_state = filter(model, features) # HSMM filtering was slightly patched
    elif isinstance(model,ssm.HMM):
        belief_state = model.filter(features) # HMM filtering using original ssm code
    
    # Save predictions
    current_belief = belief_state[-1,:]

    return current_belief

def load_model(model_fn,numStates,numSubstates,obsdim):

    with open(model_fn,'rb') as f:
        model,pca_features,mean,stdev,lls = pickle.load(f)
    print("Model loaded:")

    model_numStates = model.transitions.K
    model_obsdim = model.transitions.D
    if isinstance(model,ssm.HSMM):
        print("This is an HSMM model.")
        model_numSubstates = model.transitions.r_min
    elif isinstance(model,ssm.HMM):
        print("This is an HMM model")
        model_numSubstates = 1
    else:
        raise Exception("Model type must be ssm.HMM or ssm.HSMM")
    
    if numStates != model_numStates:
        raise Exception(f"Number of states does not match: Bonsai = {numStates}, Pickle file = {model_numStates}")

    if numSubstates != model_numSubstates:
        raise Exception(f"Number of substates does not match: Bonsai = {numSubstates}, Pickle file = {model_numSubstates}")

    if obsdim != model_obsdim:
        raise Exception(f"Observation dimension does not match: Bonsai = {obsdim}, Pickle file = {model_obsdim}")

    
    return model,pca_features,mean,stdev


def filter(model, data, input=None, mask=None, tag=None):
        m = model.state_map
        pi0 = model.init_state_distn.initial_state_distn
        Ps = model.transitions.transition_matrices(data, input, mask, tag)
        log_likes = model.observations.log_likelihoods(data, input, mask, tag)
        pzp1 = hmm_filter(replicate(pi0, m), Ps, replicate(log_likes, m))
        return collapse(pzp1, m)

# redefining ssm's function with a single added line of code
def hmm_filter(pi0, Ps, ll):
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Check if using heterogenous transition matrices
    hetero = (Ps.shape[0] == T-1)

    # Predict forward with the transition matrix
    pz_tt = np.empty((T-1, K))
    pz_tp1t = np.empty((T-1, K))
    for t in range(T-1):
        m = np.max(alphas[t])
        pz_tt[t] = np.exp(alphas[t] - m)
        pz_tt[t] /= np.sum(pz_tt[t])
        pz_tp1t[t] = pz_tt[t].dot(Ps[hetero*t])

    # Include the initial state distribution
    # Numba's version of vstack requires all arrays passed to vstack
    # to have the same number of dimensions.
    pi0 = np.expand_dims(pi0, axis=0)
    pi0 = pi0 / np.sum(pi0) # line added for PATCH
    pz_tp1t = np.vstack((pi0, pz_tp1t))

    # Numba implementation of np.sum does not allow axis keyword arg,
    # and does not support np.allclose, so we loop over the time range
    # to verify that each sums to 1.
    for t in range(T):
        assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8
    

    return pz_tp1t
