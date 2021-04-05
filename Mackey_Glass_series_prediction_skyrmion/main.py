import numpy as np
import argparse
from MG_prediction import * 
from plot import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='1skyrmion_voltage', help='folder of the training set')
parser.add_argument('--keep_step', type=int, default=2, help='duration of each pre-processed input in the reservoir')
parser.add_argument('--mu', type=float, default=1e-4, help='regularization factor of ridge regression') #time interval
parser.add_argument('--h_max', type=int, default=80, help='duration of each pre-processed input in the reservoir')
parser.add_argument('--plot_error','-pe', action='store_false', help='plot the prediction results')

parser.add_argument('-f')
args = parser.parse_args()

previous_steps = np.arange(0,4,4)
Neural_number = 50
H = np.arange(0,args.h_max,1)


#---load Mumax3 output for prediction
data_from_reservoir = (np.loadtxt('../Mumax3_simulations/'+args.name+'_mackey_glass/table.txt').T)[3]
data_from_reservoir =(( data_from_reservoir-data_from_reservoir[0])*10)[::2]
Tr_errors_mu, Ts_errors_mu, Y_tr_mu, Y_ts_mu, tr, ts = RC(data_from_reservoir, Neural_number, 
        previous_steps, H, keep_step = args.keep_step, Disgard = 300, Total_num = 10000, is_kfold = False,mu = args.mu)

#---load NeuralODEs output for prediction
data_from_reservoir = np.loadtxt('../NeuralODEs/Mumax_skyrmion_model/output/mg_'+args.name+'.txt')[::1]
Tr_errors, Ts_errors, Y_tr, Y_ts, tr, ts = RC(data_from_reservoir, Neural_number, previous_steps, 
                   H, keep_step = args.keep_step, Disgard = 300, Total_num = 10000, is_kfold = False,mu = args.mu)

#---plot the NRMSE & H
if args.plot_error:
    fig = plt.figure(figsize=(6.0, 4.4))
    fig1 = fig.add_subplot(111)
    plt.plot(H,(np.log10(np.array(Ts_errors_mu[0]))).T, 'g', label='Mumax')
    plt.plot(H,(np.log10(np.array(Ts_errors[0]))).T, 'g--', label='NODE')
    plot(fig1, font_size = 20, fr_thk =  2, xy_tick_thk = 2,is_xlabel = True, xlabel = 'Horizontal step',is_ylabel = True, 
   ylabel = 'NRMSE (log)',is_lim = True, xlim = [0,80], ylim = [-4,0.1],is_title = False, is_legend = True, 
     legend_size = 18, leg_loc = 'lower right', bitx = 0, bity = 1)
    plt.savefig('./output/NRMSE_'+args.name+'.png')
else:
    #---plot the prediction results of Mumax
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    fig1 = fig.add_subplot(111)
    tr_x = np.arange(0,len(tr.T))
    ts_x = tr_x[-1]+np.arange(0,len(ts.T))
    plt.plot(np.concatenate((tr_x, ts_x)), np.concatenate((tr.T, ts.T)),  'b', label='true')
    plt.plot(tr_x, Y_tr_mu.T, color = 'orange', label='train')
    plt.plot(ts_x, Y_ts_mu.T, 'g--', label='test')
    plt.plot(np.concatenate((tr_x, ts_x)), np.abs(np.concatenate((tr.T, ts.T))- np.concatenate((Y_tr.T, Y_ts.T))), 'r', label='prediction error')
    plot(fig1, font_size = 20, fr_thk =  2, xy_tick_thk = 2, xlabel = 'Time (a.u.)', ylabel = 'Amplitude (a.u.)',is_lim = True, xlim = [4000,6000], ylim = [0, 1.4],is_title = True, title = 'Mumax',is_legend = True, legend_size = 15, leg_loc = 'lower center', bitx = 0, bity = 1)
    plt.savefig('./output/pred_'+args.name+'_Mumax.png')

    #---plot the prediction results of NODE
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    fig1 = fig.add_subplot(111)
    plt.plot(np.concatenate((tr_x, ts_x)), np.concatenate((tr.T, ts.T)),  'b', label='true')
    plt.plot(tr_x, Y_tr.T, color = 'orange', label='train')
    plt.plot(ts_x, Y_ts.T, 'g--', label='test')
    plt.plot(np.concatenate((tr_x, ts_x)), np.abs(np.concatenate((tr.T, ts.T))- np.concatenate((Y_tr.T, Y_ts.T))), 'r', label='prediction error')
    plot(fig1, font_size = 20, fr_thk =  2, xy_tick_thk = 2, xlabel = 'Time (a.u.)', ylabel = 'Amplitude (a.u.)',is_lim = True, xlim = [4000,6000], ylim = [0, 1.4],is_title = True, title = 'NODE',is_legend = True, legend_size = 15, leg_loc = 'lower center', bitx = 0, bity = 1)
    plt.savefig('./output/pred_'+args.name+'_NODE.png')
