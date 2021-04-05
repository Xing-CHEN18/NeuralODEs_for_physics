from sklearn.model_selection import KFold
import numpy as np

def get_processed_mg(Neural_number, scale = 1, mg_size = 10000):
    mg = np.loadtxt('mg.txt')[:mg_size]
    mg = (mg/scale).reshape(1,mg_size)
    np.random.seed(3)
    W_mask = np.random.randn(Neural_number, 1)
    input_processed = np.dot(W_mask, mg)
    sequence = np.reshape(input_processed, (-1,1), 'F').squeeze()
    sequence = np.append(sequence,0)
    
    return sequence
    
def prepare_states(data_from_reservoir, Neural_number, Total_num, keep_step, Disgard):
    true_mg = (np.loadtxt('mg.txt'))[Disgard:Total_num]
    states = (data_from_reservoir[:np.int_(Total_num*Neural_number*keep_step)])[::keep_step].reshape((Neural_number,Total_num), order='F')
    #states = states[:,Disgard:Total_num]
    return states, true_mg

def output_matr(States_train,tr, mu):
    A = np.dot(States_train,States_train.T)+mu*np.eye(len(States_train))
    Ai = np.linalg.inv(A)
    Wout = np.dot(Ai,np.dot(States_train,tr.T)).T
    
    return Wout

def get_accuracy(Wout, States_train, States_test,tr,ts):
    Y_ts = np.dot(Wout, States_test)
    Y_tr = np.dot(Wout, States_train)
    tr_error = np.sqrt(np.mean((Y_tr-tr)**2)/np.var(tr))
    ts_error = np.sqrt(np.mean((Y_ts-ts)**2)/np.var(ts))
    #tr_error = np.sqrt(np.mean((Y_tr-tr)**2))
    #ts_error = np.sqrt(np.mean((Y_ts-ts)**2))
    
    return tr_error, ts_error, Y_tr, Y_ts

def get_error(states, true_mg, Disgard, h, previous_steps, train_num, is_kfold, kfold = 5, mu = 1e-4):
    States = states[:,Disgard:]
    for i in range(previous_steps):
        temp = states[:,Disgard-(i+1):-(i+1)]
        States = np.vstack((States,temp))
        
    States = States.T
    #*******prepare for the training matrix and teaching matrx:
    if not is_kfold:
        States_train = States[:train_num].T
        States_test = States[train_num:len(States)-h].T
        tr = true_mg[h:train_num+h].reshape(1, train_num)
        ts =  true_mg[train_num+h:].reshape(1, len(true_mg[train_num+h:]))
        Wout = output_matr(States_train, tr, mu)
        tr_error, ts_error,Y_tr, Y_ts = get_accuracy(Wout, States_train, States_test,tr,ts)
        
    else:
        #States = States[:-h]
        States = States[:-h]
        y = true_mg[h:]
        kf = KFold(n_splits=kfold, shuffle = False)
        tr_errors = 0
        ts_errors = 0
        for train_index, test_index in kf.split(States):
            States_train = States[train_index].T
            States_test = States[test_index].T
            tr = y[train_index].reshape(1, len(train_index))
            ts = y[test_index].reshape(1, len(test_index))
            Wout = output_matr(States_train, tr, mu)
            tr_error, ts_error, Y_tr, Y_ts = get_accuracy(Wout, States_train, States_test,tr,ts)
            tr_errors += tr_error
            ts_errors += ts_error
            
        tr_error = tr_errors/kfold
        ts_error = ts_errors/kfold
    
    return tr_error, ts_error, Y_tr, Y_ts,  tr, ts  
    
def RC(data_from_reservoir, Neural_number, previous_steps, H, keep_step = 2, Disgard = 300, Total_num = 10000,  is_kfold = False, mu = 1e-4):
    train_num = 5000
    states, true_mg = prepare_states( data_from_reservoir, Neural_number, Total_num, keep_step, Disgard)

    Tr_errors =[]
    Ts_errors =[]
    for previous_step in previous_steps:
        tr_errors =[]
        ts_errors =[]
        for h in H:
            tr_error, ts_error, Y_tr, Y_ts,  tr, ts   = get_error(states, true_mg, Disgard, h, previous_step, train_num, is_kfold, mu = mu)
            
            #print(tr_error, ts_error)
            tr_errors.append(tr_error)
            ts_errors.append(ts_error)
        Tr_errors.append(tr_errors)
        Ts_errors.append(ts_errors)
        
    return Tr_errors, Ts_errors, Y_tr, Y_ts,  tr, ts    
    
    
    
