# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:06:08 2023

@author: iliodis
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os
import tensorflow as tf


### CHANGE THAT PATH IF NEEDED
# path0 = Path("/Users/ioannisliodis/Desktop/Repositories/ANN-surrogate-model")
path0 = Path()


path1 = path0.joinpath('f1_models')
path2 = path0.joinpath('f2_models')

nameseos =  ['aprldponline_nan.coldrmfcrust',
             'bhblp.cold',
             'dd2all.cold',
             'eosAU.166',
             'eosUU.166',
             'eos_bsk20.166',
             'exLS220_smooth.cold',
             'exLS375_smooth.cold',
             'gs1ga_smooth.cold',
             'gs2ga_smooth.cold',
             'ppapr3.cold400',
             'ppeng.cold400',
             'ppgnh3.cold',
             'pph4.cold',
             'ppmpa1.cold400',
             'ppsly4.cold',
             'ppwff2.cold',
             'sfho.cold',
             'tm1.cold',
             'tma_smooth.cold']

### f1
f1_models  = []
f1_EOSs    = []
f1_scalers_targ = []
f1_scalers_pred = []

# folder_list = glob.glob(str(path0.joinpath("*")))
f1_folder_list = []
for n in range(20):
    f1_folder_list.append(str(path1.joinpath(nameseos[n])))



for f1_folder in f1_folder_list:
    # file_list = glob.glob(str(Path(folder).joinpath("*")))
    f1_file_list = []
    for i in range(4):
        f1_file_list.append(str(Path(f1_folder).joinpath(os.listdir(f1_folder)[i])))
    
    for f1_file in f1_file_list:
        if f1_file.endswith(".json"):
            f1_file_json = f1_file
        elif f1_file.endswith(".h5"):
            f1_file_h5 = f1_file
    
    # Read the standard scalers
    f1_scalersdat = pd.read_json(f1_file_json, orient = 'table')
    
    # Create the scaler
    from sklearn.preprocessing import StandardScaler
    f1_scalerpred = StandardScaler()
    f1_scalerpred.mean_  = f1_scalersdat.pred_sc_mean[0]
    f1_scalerpred.scale_ = f1_scalersdat.pred_sc_scale[0]
    
    f1_scalertarg = StandardScaler()
    f1_scalertarg.mean_  = f1_scalersdat.targ_sc_mean[0]
    f1_scalertarg.scale_ = f1_scalersdat.targ_sc_scale[0]
    
    
    
    # Read the model
    f1_reconstructed_model = tf.keras.models.load_model(f1_file_h5, compile=False)
    
    f1_models.append(f1_reconstructed_model)
    f1_EOSs.append(os.path.split(f1_folder)[1])
    f1_scalers_targ.append(f1_scalertarg)
    f1_scalers_pred.append(f1_scalerpred)



### f2
f2_models  = []
f2_EOSs    = []
f2_scalers_targ = []
f2_scalers_pred = []


# folder_list = glob.glob(str(path0.joinpath("*")))
f2_folder_list = []
for n in range(20):
    f2_folder_list.append(str(path2.joinpath(nameseos[n])))



for f2_folder in f2_folder_list:
    # file_list = glob.glob(str(Path(folder).joinpath("*")))
    f2_file_list = []
    for i in range(3):
        f2_file_list.append(str(Path(f2_folder).joinpath(os.listdir(f2_folder)[i])))
    
    for f2_file in f2_file_list:
        if f2_file.endswith(".json"):
            f2_file_json = f2_file
        elif f2_file.endswith(".h5"):
            f2_file_h5 = f2_file
    
    # Read the standard scalers
    f2_scalersdat = pd.read_json(f2_file_json, orient = 'table')
    
    # Create the scaler
    # from sklearn.preprocessing import StandardScaler
    f2_scalerpred = StandardScaler()
    f2_scalerpred.mean_  = f2_scalersdat.pred_sc_mean[0]
    f2_scalerpred.scale_ = f2_scalersdat.pred_sc_scale[0]
    
    f2_scalertarg = StandardScaler()
    f2_scalertarg.mean_  = f2_scalersdat.targ_sc_mean[0]
    f2_scalertarg.scale_ = f2_scalersdat.targ_sc_scale[0]
    
    
    
    # Read the model
    f2_reconstructed_model = tf.keras.models.load_model(f2_file_h5, compile=False)
    
    f2_models.append(f2_reconstructed_model)
    f2_EOSs.append(os.path.split(f2_folder)[1])
    f2_scalers_targ.append(f2_scalertarg)
    f2_scalers_pred.append(f2_scalerpred)





def f1(eos,p_c,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    p_c : float
        Central pressure.
    a : float
        Coupling constant.

    Returns
    -------
    M : float
        Mass.
    R : float
        Radius.

    """
    
    where = f1_EOSs.index(eos)
    
    model = f1_models[where]
    scaler_pred = f1_scalers_pred[where]
    scaler_targ = f1_scalers_targ[where]
    
    p_c = np.log10(p_c)
    
    X = np.array([[p_c,a]])
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model.predict(X, verbose = 0)
    
    pred_outs = 10**scaler_targ.inverse_transform(pred_outs)
    
    M = pred_outs[0,0]
    
    R = pred_outs[0,1]
    
    return M,R


def f1_fast(eos,p_c,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    p_c : float
        Central pressure.
    a : float
        Coupling constant.

    Returns
    -------
    M : float
        Mass.
    R : float
        Radius.

    """
    
    where = f1_EOSs.index(eos)
    
    model = f1_models[where]
    scaler_pred = f1_scalers_pred[where]
    scaler_targ = f1_scalers_targ[where]
    
    p_c = np.log10(p_c)
    
    X = np.array([[p_c,a]])
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model(X).numpy()
    
    pred_outs = 10**scaler_targ.inverse_transform(pred_outs)
    
    M = pred_outs[0,0]
    
    R = pred_outs[0,1]
    
    return M,R


def f1_batch(eos,p_c,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    p_c : numpy array
        Central pressure.
    a   : numpy array
        Coupling constant.

    Returns
    -------
    M : numpy array
        Mass.
    R : numpy array
        Radius.

    """
    
    where = f1_EOSs.index(eos)
    
    model = f1_models[where]
    scaler_pred = f1_scalers_pred[where]
    scaler_targ = f1_scalers_targ[where]
    
    p_c = np.log10(p_c)
    
    X = np.array([p_c,a]).T
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model.predict(X, verbose = 0)
    
    pred_outs = 10**scaler_targ.inverse_transform(pred_outs)
    
    M = pred_outs.T[0]
    
    R = pred_outs.T[1]
    
    return M,R



def f2(eos,M,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    M : float
        Mass.
    a : float
        Coupling constant.

    Returns
    -------
    R : float
        Radius.

    """
    
    where = f2_EOSs.index(eos)
    
    model = f2_models[where]
    scaler_pred = f2_scalers_pred[where]
    scaler_targ = f2_scalers_targ[where]
    
    
    X = np.array([[M,a]])
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model.predict(X, verbose = 0)
    
    pred_outs = scaler_targ.inverse_transform(pred_outs)
    
    
    R = pred_outs[0,0]
    
    return R


def f2_fast(eos,M,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    M : float
        Mass.
    a : float
        Coupling constant.

    Returns
    -------
    R : float
        Radius.

    """
    
    where = f2_EOSs.index(eos)
    
    model = f2_models[where]
    scaler_pred = f2_scalers_pred[where]
    scaler_targ = f2_scalers_targ[where]
    
    
    X = np.array([[M,a]])
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model(X).numpy()
    
    pred_outs = scaler_targ.inverse_transform(pred_outs)
    
    
    R = pred_outs[0,0]
    
    return R


def f2_batch(eos,M,a):
    """
    

    Parameters
    ----------
    eos : str
        Equation of State.
    M : numpy array
        Mass.
    a : float
        Coupling constant.

    Returns
    -------
    R : numpy array
        Radius.

    """
    
    where = f2_EOSs.index(eos)
    
    model = f2_models[where]
    scaler_pred = f2_scalers_pred[where]
    scaler_targ = f2_scalers_targ[where]
    
    
    X = np.array([M,a]).T
    
    #  Scale the data
    X = scaler_pred.transform(X)
    
    
    pred_outs = model.predict(X, verbose = 0)
    
    pred_outs = scaler_targ.inverse_transform(pred_outs)
    
    
    R = pred_outs.T[0]
    
    return R




