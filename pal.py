"""
Arjun Aeri 2020 

PAL: Preprocessing And Loading module 
This file contains functions to load and preprocess the EMG data
"""

import numpy as np
import os
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, GlobalAveragePooling1D
from scipy.signal import butter, lfilter, iirnotch

ddir = './EMG_data_for_gestures-master/'

ns = 200

def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

def get_xy(process_input=feature_normalize):
  tdx = np.load('tdataf.npy')
  tdy = np.load('tdatafl.npy')

  #tdx = process_input(tdx)
  tdy = np_utils.to_categorical(tdy, 7)

  return (tdx, tdy)

""" 
gen_numpy_datlab goes through the data folder and 
saves the data (x) and the labels (y) in a binary numpy format.
This is useful since we read the data from text CSV once,
and from then on can load the data from the much faster binary 
.npy representation.  

ddir: Parent directory of dataset
"""
def gen_numpy_datlab(ddir):
    ofd = 'tdataf'

    folders = os.listdir(ddir)

    data = []
    labels = []
    rdat = None 

    for fold in folders: 
      filez = os.listdir(ddir + fold)

      for fil in filez:
        delimiter = "\t"
        if rdat is None:
            rdat = np.loadtxt(ddir + fold + "/" + fil, delimiter = delimiter, skiprows=1)
        else:
            rdat = np.concatenate((rdat, np.loadtxt(ddir + fold + "/" + fil, delimiter = delimiter, skiprows=1)))

    for ln in range(0, rdat.shape[0] - ns, ns):
      cl = rdat[ln][-1]
      if cl > 0 and rdat[ln + ns][-1] == cl:
        data.append(np.array(rdat[ln:ln + ns])[:,1:9]) # select the 8 channels, disregard timestep data
        labels.append(rdat[ln][-1] - 1)                # shift labels to be index from 0-n 

    np.save(ofd, data)
    np.save(ofd + 'l', labels)

fs = 200

ns = 200

"""
filter_emg 
Filters data with:
             a bandpass filter from 2 to 45 hz (focus on expected muscle frequencies),
             a notch filter of 60 hz (remove interference from AC)
This method is not used now. But it can be added to the preprocessing function list.
"""
def filter_emg(dataset):
  bpf = butter(4, [2/(fs/2), 45/(0.5*fs)], btype='band')
  nf = iirnotch(60/(fs/2),30)
  return lfilter(nf[0], nf[1], lfilter(bpf[0],bpf[1], dataset)) 