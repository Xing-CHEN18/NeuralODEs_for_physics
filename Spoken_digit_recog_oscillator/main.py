import numpy as np
import time
from func_utlis import *
from plot import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--enable_train','-ts',  action='store_false', help='enable the training process')
parser.add_argument('--method', type=str, default='co', help='folder of the training set') #'co' and 'spc'
parser.add_argument('--steps', type=int, default=2, metavar = 'k', help='the number of shifted steps(new dimensions k)')
parser.add_argument('--seed_num', type=int, default=8, help='seed number of random noise') 
parser.add_argument('--var', type=float, default=0.0, help='variance of noise')  
parser.add_argument('--save_out', '-no_save', action='store_false', help='save the results of recognition rate')
parser.add_argument('--plot_save', action='store_true', default=True, help='choose to save the plot of recognition rate')
parser.add_argument('--plot_from_file', action='store_true', default=True, help='choose to save the plot from reading the saved recognition rate')
parser.add_argument('-f')
args = parser.parse_args()


if args.enable_train:
    if args.method == 'co':
        path = './data_experiment/Cochlear_F_7mA_4482Oe_AVG1_10_10_2016.mat'
        spokenDB_2D, spokenDB_2D_in, targets = get_in_out_exp(path)
    else:
        path = './data_experiment/Spectro_F_6mA_3497Oe_AVG1_03_11_2016.mat'
        spokenDB_2D, spokenDB_2D_in, targets = get_in_out_exp(path)
    
    #--------------------get NODE output
    path = '../NeuralODEs/Experiment_oscillator_model/output/NODE_'+args.method+'_dim'+str(args.steps)+'.mat'
    spoken_node_2D, _, _ = get_in_out_node(path)
    path = '../NeuralODEs/Experiment_oscillator_model/output/NODE_'+args.method+'_dim'+str(args.steps)+'_noi.mat'
    spoken_node_2D_noi, _, _ = get_in_out_node(path)


    accuracies_exp_all = []
    accuracies_node_all = []
    accuracies_node_noi_all = []
    #--------------------add noise into the output or not
    spoken_node_2D = add_noise_to_states(spoken_node_2D, args.var, args.seed_num)

    for uttr in np.arange(1,10):
        _, _, _, accuracy_exp = accuracy_uttr(uttr, spokenDB_2D, targets)
        _, _, _, accuracy_node = accuracy_uttr(uttr, spoken_node_2D, targets)
        _, _, _, accuracy_node_noi = accuracy_uttr(uttr, spoken_node_2D_noi, targets)
    
        print(accuracy_exp, accuracy_node, accuracy_node_noi)
        accuracies_exp_all.append(accuracy_exp)
        accuracies_node_all.append(accuracy_node)
        accuracies_node_noi_all.append(accuracy_node_noi)
        
    if args.save_out:
        mdic = {"accuracy_ts_all": accuracies_exp_all}
        sio.savemat("./output/recg_exp_dim"+str(args.steps)+".mat", mdic)
        mdic = {"accuracy_ts_all": accuracies_node_all}
        sio.savemat("./output/recg_node_"+args.method+"_dim"+str(args.steps)+".mat", mdic)
        mdic = {"accuracy_ts_all": accuracies_node_noi_all}
        sio.savemat("./output/recg_node_noi_"+args.method+"_dim"+str(args.steps)+".mat", mdic)
        
if args.plot_save:
    path = './output/recg_exp_dim'+str(args.steps)+'.mat'
    DATA = sio.loadmat(path)
    accuracies_exp_all = DATA['accuracy_ts_all'] # 

    x = np.arange(1, 10, 1)
    y_exp = accuracies_exp_all[0]

    lower_error = np.empty(9)
    upper_error = np.empty(9)
        
    fig = plt.figure(figsize=(6, 4.4))
    fig1 = fig.add_subplot(111)

    paths = ['./output/recg_node_'+args.method+'_dim'+str(args.steps)+'.mat',  
         './output/recg_node_noi_'+args.method+'_dim'+str(args.steps)+'.mat']

    legends = ['experiment', 
           'NODE (without noise)', 
           'NODE (with noise)']

    fig1.errorbar(x, y_exp, fmt='-bo',label = legends[0])

    for i in range(len(paths)):
        
        DATA = sio.loadmat(paths[i])
        accuracy_ts_all = DATA['accuracy_ts_all'] # (new!)
        y = accuracy_ts_all[0]
        fig1.errorbar(x, y, linestyle='dashed',label = legends[i+1], linewidth = 3)
            

    if args.method == 'co':
        plt.ylim(0.65,1.01)
        plot(fig1, font_size = 20, fr_thk =  2, xy_tick_thk = 2, is_xlabel = True, 
                 xlabel = 'Number of utterances, $N$',is_ylabel = True,ylabel = 'recognition rate(%)', 
             is_lim = False, xlim = [0,70], ylim = [0, 3],is_title = True, title = 'Cochlear, $k = 2$', is_legend = True, 
             legend_size = 17, leg_loc = 'lower right', bitx = 0, bity = 1)
    else:
        plt.ylim(0.0,1.0)
        plot(fig1, font_size = 20, fr_thk =  2, xy_tick_thk = 2, is_xlabel = True, 
                 xlabel = 'Number of utterances, $N$',is_ylabel = True,ylabel = 'recognition rate(%)', 
                 is_lim = False, xlim = [0,70], ylim = [0, 3],is_title = True, title = 'Spectrogram, $k = 2$', is_legend = True,
                 legend_size = 17, leg_loc = 'lower right', bitx = 0, bity = 1)
        
    plt.savefig('./output/recg_'+args.method+'_dim'+str(args.steps)+'.png', dpi=300)

        
    
    

    
