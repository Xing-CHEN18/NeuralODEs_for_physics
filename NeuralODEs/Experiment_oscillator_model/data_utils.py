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
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
from scipy import interpolate
from scipy import linalg
from scipy.stats import norm, cauchy,laplace,logistic, gennorm
from scipy.optimize import curve_fit
import scipy.io as sio
#import os.path
#import itertools
#import pandas as pd
#from scipy.signal import savgol_filter
#import pywt
#import sys


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

def get_add_mat(path, data_size, discard, ds, sp, itr, device, Default_dtype = torch.float64, dt = (25/2)/1000):
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

def add_noise_to_states(spoken_node_2D, var, seed_num):
    speaker = [1,2,5,6,7]
    sp_indx = np.arange(0,5)
    uttrs_idx = np.arange(0,10)
    dig_idx = np.arange(0,10)
    np.random.seed(seed_num)
    spoken_node_2D_noise = np.empty((5,10,10), dtype=object)
    for sp in sp_indx:
        for uttr in uttrs_idx:
            for di in dig_idx:
                mat = spoken_node_2D[sp,uttr,di]
                ntr = np.random.normal(0.0, var, mat.shape)
                #ntr = truncated_cauchy_rvs(loc=0, scale=var, a=-0.2, b=0.2, size=mat.shape)
                #ntr =   np.clip(cauchy.rvs(loc=0.0, scale=var, size=mat.shape), -clip, clip) 
                #ntr = gennorm.rvs(beta = beta, scale=var,size=mat.shape)
                spoken_node_2D_noise[sp,uttr,di] =  mat + ntr
                
    return spoken_node_2D_noise


def test_all_speakers(ode_func, spokenDB_2D_in, spokenDB_2D, dt, dim, Default_dtype = torch.float64):
    

    speaker = [1,2,5,6,7]
    sp_indx = np.arange(0,5)
    uttrs_idx = np.arange(0,10)
    dig_idx = np.arange(0,10)

    spoken_node_2D = np.empty((5,10,10), dtype=object)
    spoken_node_time = np.empty((5,10,10), dtype=object)
    spoken_node_loss = np.empty((5,10,10), dtype=object)


    for sp in sp_indx:
        for uttr in uttrs_idx:
            for di in dig_idx:
                readin = np.append(spokenDB_2D[sp,uttr,di].flatten('F'), np.zeros(dim-1))
                readin = np.expand_dims(readin, axis=1)
                lg = len(readin)

                data = readin[:lg-(dim-1)]
                for i in range(1,dim):
                    temp = readin[i:lg-(dim-1-i)]
                    data = np.concatenate((data, temp), axis = -1)
                
                true_y = torch.tensor(data, dtype=Default_dtype)
                true_y = torch.unsqueeze(true_y, 1)
                true_y0 = torch.tensor(np.array(true_y[0]), dtype=Default_dtype)
            
                input_data = np.append(spokenDB_2D_in[sp,uttr,di].flatten('F'),np.zeros(dim))
                #t = torch.arange(0, len(input_data)*dt, dt)
                t = torch.linspace(0, len(input_data)*dt-dt, len(input_data))
                tck = interpolate.interp1d(t.numpy(),input_data, kind='linear')
                t = t[:lg-(dim-1)]

                ext = torch.tensor(input_data[:lg-(dim-1)].reshape((-1,1)), dtype=Default_dtype)
                ext = torch.unsqueeze(ext, 1) 
                ext_time = torch.unsqueeze(t, 1)
                ext_time = torch.unsqueeze(ext_time, 1)
                ext = torch.cat((ext, ext_time), dim = -1)

                ode_func.sequence = tck
            
                with torch.no_grad():
                    c0 = time.perf_counter()
                    pred_y = odeint(ode_func, torch.cat((true_y0, ext[0,:,1:]), dim = -1), t, method = 'rk4' )
                    loss = F.mse_loss(pred_y[:,:,0:1],true_y[:,:,0:1])
                    print('Test set: MSE(exp-Node) = {:.6f}'.format(loss.item()))
                    c1 = time.perf_counter()
                    print(str(c1-c0)+ "seconds.")
                    #spoken_ode_2D.append(pred_y[:,0,0].numpy().reshape((400,-1), order = 'F'))
                    #spoken_ode_time.append(c1-c0)
                    spoken_node_2D[sp,uttr,di] = pred_y[:,0,0].numpy().reshape((400,-1), order = 'F')
                    spoken_node_time[sp,uttr,di] = c1-c0
                    spoken_node_loss[sp,uttr,di] = loss.item()
    
    
    return spoken_node_2D, spoken_node_time, spoken_node_loss