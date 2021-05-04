import numpy as np
from scipy import linalg
import itertools
import scipy.io as sio
import time



#read experimental input & output data
def get_in_out_exp(path):
    
    DATA = sio.loadmat(path)
    spokenDB_2D = DATA['spokenDB_2D']
    spokenDB_2D_in = DATA['spokenDB_2D_in'] # (new!)
    targets = DATA['targets']
    
    return spokenDB_2D, spokenDB_2D_in, targets

#read NODE output data
def get_in_out_node(path):
    
    DATA = sio.loadmat(path)
    spoken_node_2D = DATA['spoken_node_2D']
    spoken_node_time = DATA['spoken_node_time'] # (new!)
    spoken_node_loss = DATA['spoken_node_loss']
    
    return spoken_node_2D, spoken_node_time, spoken_node_loss

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
                #print(sp, uttr, di)
                mat = spoken_node_2D[sp,uttr,di]
                ntr = np.random.normal(0.0, var, mat.shape)
                spoken_node_2D_noise[sp,uttr,di] =  mat + ntr
                #print((mat + ntr).shape,spoken_node_2D_noise[sp,uttr,di].shape)
                
    return spoken_node_2D_noise



#*****-----------------------prepare the train and test data
def get_train_test_Mat(spoken_node_2D_noise, targets, train_idx, test_idx):  
    scale = 1
    train_mat = spoken_node_2D_noise[:,train_idx,:]*scale
    test_mat = spoken_node_2D_noise[:,test_idx,:]*scale
    train_tar  = targets[:,train_idx,:]
    test_tar = targets[:,test_idx,:]

    sps, utters_tr, digs = train_mat.shape
    sps, utters_ts, digs = test_mat.shape
    train_mat_flatten = []
    test_mat_flatten = []
    train_target_flatten = []
    test_target_flatten = []
    #np.random.seed(seed_num)
    
    for utter in range(utters_tr):
        for sp in range(sps):
            train_mat_flatten_sp = []
            train_target_flatten_sp = []
            for dig in range(digs):
                mat = train_mat[sp, utter, dig]
                tar = train_tar[sp, utter, dig]
                #ntr = np.random.normal(0.0, var, mat.shape)
                train_mat_flatten_sp.append(mat)
                train_target_flatten_sp.append(tar)
            train_mat_flatten_sp = np.hstack(train_mat_flatten_sp)
            train_target_flatten_sp = np.hstack(train_target_flatten_sp)
    #print(len(train_mat_flatten),len(train_target_flatten))           
            train_mat_flatten.append(train_mat_flatten_sp)
            train_target_flatten.append(train_target_flatten_sp)
    
    
    
    for utter in range(utters_ts):
        for sp in range(sps):
            test_mat_flatten_sp = []
            test_target_flatten_sp = []
            for dig in range(digs):
                mat = test_mat[sp, utter, dig]
                tar = test_tar[sp, utter, dig]
                #nts = np.random.normal(0.0, var, mat.shape)
                test_mat_flatten_sp.append(mat)
                test_target_flatten_sp.append(tar)
            test_mat_flatten_sp = np.hstack(test_mat_flatten_sp)
            test_target_flatten_sp = np.hstack(test_target_flatten_sp)
        
            test_mat_flatten.append(test_mat_flatten_sp)
            test_target_flatten.append(test_target_flatten_sp)
    #test_mat_flatten = np.hstack(test_mat_flatten)
    #test_target_flatten = np.hstack(test_target_flatten)
    #print(len(test_mat_flatten),len(test_target_flatten))
    #nt = np.random.normal(0, 0.01, train_mat_flatten.shape)
    return train_mat_flatten, train_target_flatten, test_mat_flatten, test_target_flatten

#*****-----------------------train the output matrix
def get_output_matr(output_matr_itrs, y_target_itrs):
    
    output_matr_itrs = np.concatenate((np.ones((1,output_matr_itrs.shape[1])), output_matr_itrs), axis = 0)
    
    Y_wave = y_target_itrs
    S_star = linalg.pinv(output_matr_itrs, return_rank=False)
    #S_star = np.linalg.pinv(output_matr_itrs)
    #print(np.allclose(S_star, S_star2))
    
    Wout = np.matmul(Y_wave, S_star)
    Yout = np.dot(Wout ,output_matr_itrs)
    #accr = np.argmax(np.matmul(Wout, output_matr_itrs),axis =0) == np.argmax(y_target_itrs)
    
    return Wout


#*****-----------------------permutate the order randomly
def permutate(output_matr_itrs, y_target_itrs):
    index = np.arange(0,len(y_target_itrs.T))
    Lables = np.argmax(np.array(y_target_itrs.T),  axis = 1)
    
    Lables_indic = []
    for lable in range(10):
        lable_indic = index[Lables==lable]
        Lables_indic.extend(lable_indic)
        
    
    return output_matr_itrs[:,Lables_indic], y_target_itrs[:,Lables_indic]


#*****-----------------------compute the recognition rate for each selection of utterrance
def get_accuracy(Wout, output_matr,y_target):
    
    Ylabels = []
    Ylabel_trues = []
    #print(len(output_matr))
    for i in range(len(output_matr)):
        output_matr[i] = np.concatenate((np.ones((1,output_matr[i].shape[1])), output_matr[i]), axis = 0)
        Yout = np.dot(Wout, output_matr[i])
        #print(Yout.shape)
        Ylabel = np.argmax(np.dot(Yout, y_target[i].T), axis = 0)
        Ylabel_true = np.argmax(np.dot(y_target[i], y_target[i].T), axis = 0)
        Ylabels.append(Ylabel)
        Ylabel_trues.append(Ylabel_true)
    
    accuracy = np.sum(np.hstack(Ylabels) == np.hstack(Ylabel_trues))/len(np.hstack(Ylabels))
    #print(accuracy.shape)
    
    return accuracy


#*****-----------------------get the index of test set
def get_test_uttr(train_uttr, uttrs):
    boolean_arr = np.ones(uttrs.shape, dtype=bool)
    #print(train_uttr,boolean_arr)
    boolean_arr[np.array(train_uttr)] = False
    
    return uttrs[boolean_arr]


#*****-----------------------collect the recognition rate for each utterrance
def accuracy_uttr(uttr, spoken_node_2D_noise, targets):
    uttrs_idx = np.arange(0,10)
    uttrs = np.arange(1, 11)
    accuracies_tr = []
    accuracies_ts = []
    
    for train_idx in itertools.combinations(uttrs_idx, uttr):
        test_idx = get_test_uttr(train_idx, uttrs_idx)
        #spokenDB_2D = add_noise_to_states(spokenDB_2D, var, seed_num)
        train_mat_flatten, train_target_flatten, test_mat_flatten, test_target_flatten = get_train_test_Mat(spoken_node_2D_noise, targets, train_idx, test_idx)
        #print(np.hstack(train_mat_flatten).shape, np.hstack(train_target_flatten).shape)
        Wout = get_output_matr(np.hstack(train_mat_flatten), np.hstack(train_target_flatten))
        #Wout = get_output_matr(train_mat_flatten, train_target_flatten)
        accuracy_ts = get_accuracy(Wout, test_mat_flatten,test_target_flatten)
        accuracy_tr = get_accuracy(Wout, train_mat_flatten,train_target_flatten)
        #print(accuracy)
        accuracies_tr.append(accuracy_tr)
        accuracies_ts.append(accuracy_ts)

    accuracy_tr = np.sum(accuracies_tr)/len(accuracies_tr)
    accuracy_ts = np.sum(accuracies_ts)/len(accuracies_ts)
    #return accuracy_ts, accuracy_tr
    return accuracies_tr, accuracies_ts, accuracy_tr, accuracy_ts
    
