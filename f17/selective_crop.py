import scipy.io as sio
import numpy as np
import h5py
import os
from scipy import misc



print("Starting to crop selectively..")
# dataset = sio.loadmat(path)
strain_file_path = "/Users/abhinandandubey/Desktop/Mesh/523-FULLDATA/20161129 - Strain 201 (Chl1 deletion) - Position 0.mat"
dataset = h5py.File(strain_file_path, 'r')
print dataset["cDIC"].shape

