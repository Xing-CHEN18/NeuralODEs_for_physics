import numpy as np
import os
import time
import argparse

import torch
import torch.nn as nn
#import torch.tensor as Tensor
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint as odeint
import torch.nn.functional as F
#import torchaudio
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
from scipy import interpolate
import os.path
import itertools
from scipy import linalg
from scipy.stats import norm, cauchy,laplace,logistic, gennorm
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter
import pywt
import scipy.io as sio
import sys


def get_data_Mumax_pos(path, data_length: int, disgard, steps, disk_radius, Default_dtype = torch.float64):
    
    readin = (np.genfromtxt(path)[:,:])
    readin = ((readin-disk_radius)/10/1)[disgard:]
    
    data = readin[:data_length-1*(steps-1)]
    for i in range(1, steps):
        temp = readin[i:data_length-(steps-1)+i]
        data = np.concatenate((data, temp), axis = -1)
    
    true_y0 = torch.tensor(np.array(data[0]), dtype=Default_dtype) 
    true_y = torch.tensor(data, dtype=Default_dtype) 
    true_y0 = torch.unsqueeze(true_y0, 0)
    true_y = torch.unsqueeze(true_y, 1)
    
    return true_y0, true_y

def get_data_Mumax_mz(path, data_length: int, discard, steps, sigma, ds, Default_dtype = torch.float64, fix_0 = False):
    
    
    readin = (np.genfromtxt(path)[::ds,3:4])
    if fix_0: # remove the bias of the inital state
        bias0 = (np.genfromtxt('../../Mumax3_simulations/1skyrmion_input_PMA_DMI_train/table.txt')[::ds,3])[0]
    else:
        bias0 = readin[0]
    readin = ((readin - bias0)*10)[discard:]
    #print(readin.shape)
    np.random.seed(6)
    noise = np.random.normal(0, sigma, readin.shape)
    readin = readin + noise
    
    data = readin[:data_length-1*(steps-1)]
    
    for i in range(1, steps):
        temp = readin[i:data_length-(steps-1)+i]
        data = np.concatenate((data, temp), axis = -1)
    
    true_y0 = torch.tensor(np.array(data[0]), dtype=Default_dtype) 
    true_y = torch.tensor(data, dtype=Default_dtype) 
    true_y0 = torch.unsqueeze(true_y0, 0)
    true_y = torch.unsqueeze(true_y, 1)
    
    return true_y0, true_y

def get_data_Mumax_mz_DMI(t, path, data_length, discard, steps, ds, Default_dtype = torch.float64):
    
    bias0 = (np.genfromtxt('/home/xing/Mumax/Model_1skyr/sin600p_Kuamp[-0505]_Damp[-0404]_sp50_f4.out/table.txt')[::ds,3])
    mz = ((np.genfromtxt(path)[::ds,3]))
    mz = ((mz-bias0[0])*10)[discard:discard+data_length]
    readin = ((np.genfromtxt(path)[::ds,5:6])*10**3)[discard:]
    #print(readin.shape)
     
    data = readin[:data_length-1*(steps-1)]
    
    for i in range(1, steps):
        temp = readin[i:data_length-(steps-1)+i]
        data = np.concatenate((data, temp), axis = -1)
    
    true_y0 = torch.tensor(np.array(data[0]), dtype=Default_dtype) 
    true_y = torch.tensor(data, dtype=Default_dtype) 
    true_y0 = torch.unsqueeze(true_y0, 0)
    true_y = torch.unsqueeze(true_y, 1)
    
    tck_mz = interpolate.interp1d(t.numpy(), mz, kind='cubic')
    
    return true_y0, true_y, tck_mz, mz

def get_data_mat(path, data_length, discard, steps, ds, sp, itr, Default_dtype = torch.float64):
    DATA = sio.loadmat(path)
    spokenDB_2D = DATA['spokenDB_2D']*10
    data = []
    for i in range(0,9):
        data.append(spokenDB_2D[sp,itr,i].flatten('F'))
    
    output_data = (np.hstack(data).reshape((-1,1)))[::ds][discard:]
    data = output_data[:data_length-1*(steps-1)]
    
    for i in range(1, steps):
        temp = output_data[i:data_length-(steps-1)+i]
        data = np.concatenate((data, temp), axis = -1)
        
    true_y0 = torch.tensor(np.array(data[0]), dtype=Default_dtype) 
    true_y = torch.tensor(data, dtype=Default_dtype) 
    true_y0 = torch.unsqueeze(true_y0, 0)
    true_y = torch.unsqueeze(true_y, 1)
    
    return true_y0, true_y


def get_data_txt(path, data_length: int, discard, steps, ds, Default_dtype = torch.float64, start = 0, stop = 1):
    #-----------for signal from oscillator, start = 2, stop = 3
    data = (np.genfromtxt(path))[::ds]
    #print(data.shape)
    data = data *10
    output_data = data[discard:,start:stop]
    
    data = output_data[:data_length-1*(steps-1)]
    
    for i in range(1, steps):
        temp = output_data[i:data_length-(steps-1)+i]
        data = np.concatenate((data, temp), axis = -1)
    #print(data.shape)
    true_y0 = torch.tensor(np.array(data[0]), dtype=Default_dtype) 
    true_y = torch.tensor(data, dtype=Default_dtype) 
    true_y0 = torch.unsqueeze(true_y0, 0)
    true_y = torch.unsqueeze(true_y, 1)
    
    return true_y0, true_y

def get_add_txt(path, data_size: int, discard, ds, device, Default_dtype = torch.float64, start = 1, stop = 2, dt = (25/2)/1000):
    #-----------for signal from oscillator, start = 1, stop = 2
    data = (np.genfromtxt(path))[::ds]
    input_data = data[discard:discard+data_size,start:stop]*10
    #print(input_data.shape)
    ext = torch.tensor(input_data, dtype=Default_dtype) 
    ext = torch.unsqueeze(ext, 1) 
    
    #dt = (25/2)/1000
    t = torch.arange(0., (data_size)*dt, dt)
    #ts = torch.arange(0., (data_size)*dt, dt/5)
    
    ext_time = torch.unsqueeze(t, 1)
    ext_time = torch.unsqueeze(ext_time, 1)
    
    ext = torch.cat((ext, ext_time), dim = -1)
    
    if device.type == 'cpu':
        tck = interpolate.splrep(t.numpy(), ext[:,0,0].cpu().numpy())
        #sequence = interpolate.splev(ts.cpu().numpy(), tck)
        
    else:
        coeffs = natural_cubic_spline_coeffs(t, ext[:,:,0])
        tck = NaturalCubicSpline(coeffs)
    
    return t, ext, tck

def get_add_mat(path, data_size, discard, ds, sp, itr, Default_dtype = torch.float64, dt = (25/2)/1000):
    DATA = sio.loadmat(path)
    spokenDB_2D = DATA['spokenDB_2D_in']
    data = []
    for i in range(0,9):
        data.append(spokenDB_2D[sp,itr,i].flatten('F'))
    
    data = (np.hstack(data).reshape((-1,1)))[::ds]
    input_data = data[discard:discard+data_size]*10
    ext = torch.tensor(input_data, dtype=Default_dtype) 
    #ext = torch.unsqueeze(ext, 1) 
    ext = torch.unsqueeze(ext, 1) 
    #dt = (25/2)/1000
    t = torch.arange(0., (data_size)*dt, dt)
    #ts = torch.arange(0., (data_size)*dt, dt/5)
    
    ext_time = torch.unsqueeze(t, 1)
    ext_time = torch.unsqueeze(ext_time, 1)
    
    ext = torch.cat((ext, ext_time), dim = -1)
    if device.type == 'cpu':
        #tck = interpolate.splrep(t.numpy(), ext[:,0,0].cpu().numpy())
        #seq = interpolate.splev((tt.data.numpy()), tck)
        tck = interpolate.interp1d(t.numpy(), ext[:,0,0].cpu().numpy(), kind='linear') #‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’
        
    else:
        coeffs = natural_cubic_spline_coeffs(t, ext[:,:,0])
        tck = NaturalCubicSpline(coeffs)
    
    return t[:-10], ext[:-10], tck


def get_add_sin_input(data_size: int, disgard, sample_p, Default_dtype = torch.float64, dt = 0.0125):
    
    np.random.seed(6)
    t = torch.arange(0., (data_size)*dt, dt)
    #print(t.dtype, t.is_cuda)
    pi = torch.acos(torch.zeros(1)).item() * 2 
    
    sequence = (torch.tensor((0+(np.random.rand(6001))*2-1)[np.int_(disgard/sample_p*1) + 0:], dtype=Default_dtype))
    period = sample_p*dt*1
    num = (torch.floor(t/period)).long()
    resp = sequence[num]
    
    ext = (resp)*torch.sin(2*pi/(period*1)*t)+0
    ext = torch.unsqueeze(ext, 1)
    ext = torch.unsqueeze(ext, 1)
    ext_time = torch.unsqueeze(t, 1)
    ext_time = torch.unsqueeze(ext_time, 1)
   
    ext = torch.cat((ext, ext_time), dim = -1)
    
    tck = interpolate.splrep(t.cpu().numpy(), ext[:,0,0].cpu().numpy())
    
    return t, ext, tck

def get_processed_mg(Neural_number, scale = 1, mg_size = 10000):
    mg = np.loadtxt('../../Mackey_Glass_series_prediction_skyrmion/mg.txt')[:mg_size]
    mg = (mg/scale).reshape(1,mg_size)
    np.random.seed(3)
    W_mask = np.random.randn(Neural_number, 1)
    input_processed = np.dot(W_mask, mg)
    sequence = np.reshape(input_processed, (-1,1), 'F').squeeze()
    sequence = np.append(sequence,0)
    
    return sequence

def get_test_Ku_ext_DMI(data_size, dt, steps_par):
    i = np.arange(7.5,8.6,0.05)  #range of Ku: from 7.5  to 8.55 J/m^3
    ku = 0+(i-8)/0.5 #normalize the range to give to Neural ODEs
    ts = torch.arange(0., (data_size+100)*dt, dt)    # interpolation time a bit large to avoid error to solve in Neural ODEs
    period = steps_par*dt     #for each Ku, the time duration is 800 steps, see the file in 'Mumax3_simulations/1skyrmion_input_PMA_DMI_test/Ku_ext_DMI/table.txt'
    num = (torch.floor(ts/period)).long()
    resp = ku[num]
    tck_Ku = interpolate.interp1d(ts.numpy(), resp, kind='linear') # to get the interpolation  of Ku input
    ext_Ku = torch.unsqueeze(torch.tensor(resp), 1)

    dmi = np.array(30*[0.25,0,0,0]) # external input of DMI: a pulse with 1/4 of a period
    num = (torch.floor(ts/(period/4))).long() 
    resp = dmi[num]
    tck_D = interpolate.interp1d(ts.numpy(), resp, kind='linear') # to get the interpolation  of DMI pulse
    ext_D = torch.unsqueeze(torch.tensor(resp), 1)
    
    ext = torch.stack((ext_Ku, ext_D), dim = 2)
    
    return tck_Ku, tck_D, ext

def get_test_DMI_ext_Ku(data_size, dt, steps_par):
    i = np.arange(2.6,3.5,0.05)
    dmi = 0+(i-3)/0.4
    ts = torch.arange(0., (data_size+100)*dt, dt)
    period = steps_par*dt  
    num = (torch.floor(ts/period)).long()
    resp = dmi[num]
    tck_D = interpolate.interp1d(ts.numpy(), resp, kind='linear')
    ext_D = torch.unsqueeze(torch.tensor(resp), 1)
    
    ku = np.array(30*[0.8,0,0,0])
    num = (torch.floor(ts/(period/4))).long()
    resp = ku[num]
    tck_Ku = interpolate.interp1d(ts.numpy(), resp, kind='linear')
    ext_Ku = torch.unsqueeze(torch.tensor(resp), 1)
    
    ext = torch.stack((ext_Ku, ext_D), dim = 2)
    
    return tck_Ku, tck_D, ext



def get_frequency(series, num_pars, steps_par):
    dt = 5e-12
    freqs = []
    pulse = np.int_(steps_par*(1/4)) #pulse duration
    for i in range(num_pars):
        NFFT = np.int_(2**(np.ceil(np.log2(abs(steps_par-pulse))))) #perform fft on the response after the pulse of 1/4 period:
        freq = np.absolute(np.fft.rfft(series[i*steps_par+pulse:(i+1)*steps_par]-series[i*steps_par+pulse:(i+1)*steps_par].mean(), NFFT))
        freq = freq/np.max(freq)
        F1 = np.arange(0,NFFT/2+1)/NFFT*(1/dt)
        freq_max_indx = F1[np.argmax(freq)]/1e9  #in GHz
        #print(freq_tp)
        freqs.append(freq_max_indx)
        
    return freqs