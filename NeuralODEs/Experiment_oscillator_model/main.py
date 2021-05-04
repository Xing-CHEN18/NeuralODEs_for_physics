import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint as odeint
import torch.nn.functional as F
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

from data_utils import *
from plot import *

parser = argparse.ArgumentParser()
###-------params for both train and test(trained model for output):
parser.add_argument('--torch_seed', type=int, default=6, help='seed number of torch: to initialize different configurations of neural network')
parser.add_argument('--np_seed', type=int, default=6, help='numpy seed number: to randomly choose the training set')
parser.add_argument('--disable_gpu', '-gpu', action='store_false', help='disable usage of gup or not')
parser.add_argument('--enable_train','-ts',  action='store_false', help='enable the training process')
parser.add_argument('--steps', type=int, default=2, metavar = 'k', help='the number of shifted steps(new dimensions k)')
parser.add_argument('--neural_num', type=int, default=100, metavar = 'nn', help='number of the hidden unit')
parser.add_argument('--ext_in', type=str, default='rand_txt', metavar = 'ei', help='the form of the extra input') #ext: 'rand_txt', 'rand_sin','rand_mg'
parser.add_argument('--ds', type=int, default=1, help='downsampling rate of the original trajectory') #down sampling rate
parser.add_argument('--dt', type=float, default=(25/2)/1000, help='nomalized time interval of the Neural ODE') #time interval
###-------params for train only:
parser.add_argument('--method', type=str, default='co', help='folder of the training set') # 'co' or 'spc'
parser.add_argument('--sp', type=int, default=0, help='choose the speaker for train: 0-4')
parser.add_argument('--itr', type=int, default=0, help='choose the iteration for train: 0-4')
parser.add_argument('--discard', type=int, default=0, help='number of data points to remove')
parser.add_argument('--data_size', type=int, default=50000, help='total number of training data points')
parser.add_argument('--test_size', type=int, default=10000, help='total number of testing data points')
parser.add_argument('--test_freq', type=int, default=100, help='test frequency of the trained function')
parser.add_argument('--batch_time', type=int, default=20, help='number of training points for each batch')
parser.add_argument('--batch_size', type=int, default=50, help='number of batches for each iteration of training')
parser.add_argument('--Loss_min', type=float, default=1, help='initial MSE loss')
parser.add_argument('--Loss_min_th', type=float, default=0.00051, help='minimum threshold loss to stop the training')
parser.add_argument('--Loss_dim', type=int, default=1, help='number of dimensions to calculate the MSE')

######---------plot the training output:
parser.add_argument('--plot_save', action='store_true', default=True, help='choose to save the plot of training output or not')
parser.add_argument('--start', type=int, default=0, help='start point to show in the figure')
parser.add_argument('--stop', type=int, default=1000, help='stop point to show in the figure')

######---------testing mode:
parser.add_argument('--noi', type=str, default='', help='the form of the extra input') #'noi' or ''
parser.add_argument('--var', type=float, default=0.0, help='variance of noise')  
parser.add_argument('--seed_num', type=int, default=8, help='seed number of random noise') 

parser.add_argument('-f')
args = parser.parse_args()

if torch.cuda.is_available() and not args.disable_gpu:
    device = torch.device('cuda:' + str(0))
    Default_tensor_type = torch.cuda.FloatTensor
    Default_dtype = torch.float32
else:
    device = torch.device('cpu')
    Default_tensor_type = torch.DoubleTensor
    Default_dtype = torch.float64

print(device)
torch.set_default_tensor_type(Default_tensor_type)
torch.set_default_dtype(Default_dtype)



def get_batch(data_size, batch_time, batch_size, t, true_y, ext):
    s = torch.from_numpy(np.random.choice(np.arange(1, data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    #print(s)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    batch_ext = torch.stack([ext[s + i] for i in range(batch_time)], dim=0)
    return batch_y0, batch_t, batch_y, batch_ext
    
    
    
class ODEFunc(nn.Module):
    """
    ODEFunc is the module for NeuralODE y' = f(y, t, ext). The input into the NeuralODE can be random sine wave:'rand_sin', any B-spline 
    representation of a 1-D curve: 'rand_txt', time-dependent input of a sequence, such as mackey-glass series :'mg' with adjustable time step (keep_step).
    """

    def __init__(self, tck, neural_num = 400,dropout_rate = 0, dim = 4, ext_dim =2, ext2_dim = 0, discard = 1000, ext_in = 'rand_sin', 
                 keep_step = 1, sample_p = 100, dt = 12.5/1000, ext_in2 = False, ext_in2_value = 0.1):
        super(ODEFunc, self).__init__()
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.dt = dt
        
        
        self.neural_num = neural_num
        self.dim = dim
        self.ext_dim = ext_dim 
        self.ext2_dim = ext2_dim
        self.Last_dim = self.dim+0
        self.dropout_rate = dropout_rate
       
        
        if ext_in == 'rand_sin':
            self.get_ext = self.get_ext_sin
            self.period = torch.tensor([self.dt*sample_p])
            np.random.seed(6)
            self.sequence = torch.tensor((np.random.rand(6001)*2-1)[np.int_(discard/sample_p):], dtype=Default_dtype)
        
            
        elif ext_in == 'rand_txt':
            self.sequence = tck
            if device.type == 'cpu':
                self.get_ext = self.get_ext_txt_cpu
            else:
                self.get_ext = self.get_ext_txt_gpu
                
        elif ext_in == 'mg':
            self.sequence = torch.tensor(tck, dtype=Default_dtype)
            self.get_ext = self.get_ext_txt
            self.period = self.dt*keep_step
            
        if ext_in2:
            if ext_in2_value == 'rand_sin':
                self.period = torch.tensor([self.dt*sample_p])
                np.random.seed(6)
                self.sequence2 = torch.tensor((np.random.rand(6001)*2-1)[3000+np.int_(discard/(sample_p*10)):], dtype=Default_dtype)
                self.get_ext2 = self.get_ext2_sin
            else:
                self.sequence2 = ext_in2_value
                self.get_ext2 = self.get_ext2_fix
        
        self.ext0 = nn.Sequential(
            nn.Linear(self.dim+self.ext_dim+self.ext2_dim, self.neural_num),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(self.neural_num, self.neural_num),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(self.neural_num, self.dim),
        )
        
        self.ext1 = nn.Sequential(
            nn.Linear(5, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2),
        )
        
        for m in self.ext0.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)
                
        for m in self.ext1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)
                
    def get_ext2_sin(self, tt):
        
        num = tt//(self.period*10)
        resp = self.sequence2[num.long()]
        ext = (resp)*torch.sin(2*self.pi/(self.period*10)*tt)
        
        return ext
    
    def get_ext2_fix(self, tt):
        
        return (self.sequence2)*torch.ones(tt.size()) 
    
    def get_ext_sin(self, tt):
        
        num = tt//self.period
        resp = self.sequence[num.long()]
        ext = (resp)*torch.sin(2*self.pi/self.period*tt)
        
        return ext
    
    
    def get_ext_txt_cpu(self, tt):
                                 
        #ext = torch.tensor(interpolate.splev((tt.data.numpy()), self.sequence), dtype=Default_dtype)
        ext = torch.tensor(self.sequence(tt.data.numpy()), dtype=Default_dtype)
        
        return ext
    
    def get_ext_txt_gpu(self, tt):
        
        ext = self.sequence.evaluate(tt)
        
        return ext[..., 0]
    
    def get_ext_txt(self, tt):
                                 
        num = tt//self.period
        ext = self.sequence[num.long()]
        
        return ext


    def forward(self, t, y):
        
        ext = self.get_ext(y[..., self.Last_dim:])
        
        for i in range(1, self.ext_dim):
            ext_ = self.get_ext(torch.add(y[..., self.Last_dim:],i*self.dt))
            ext = torch.cat((ext, ext_), dim = -1)
        
        for i in range(0, self.ext2_dim):
            ext_ = self.get_ext2(torch.add(y[..., self.Last_dim:],i*self.dt))
            ext = torch.cat((ext, ext_), dim = -1)
        
        state0 = self.ext0(torch.cat((y[..., :self.dim], ext), dim = -1))
        const = torch.ones(y[..., :1].size())   
        state = torch.cat((state0, const), dim = -1)
        
        return state
    
class Mz_out(nn.Module):

    def __init__(self,dim = 4, neural_num = 100 ):
        super(Mz_out, self).__init__()
        
        self.mz_out = nn.Sequential(
            nn.Linear(dim, neural_num),
            nn.Tanh(),
            nn.Linear(neural_num, neural_num),
            nn.Tanh(),
            nn.Linear(neural_num, 1),
        )

        for m in self.mz_out.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)


if __name__ == "__main__":

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    
    #************* running mode: Training mode, Testing mode
    #************* training experimental data:cochlear and spectrogram
    
    #************* running mode: Training mode selected
    if args.enable_train:
        if args.method == 'co':
            path = '../../Spoken_digit_recog_oscillator/data_experiment/Cochlear_F_7mA_4482Oe_AVG1_10_10_2016.mat'
            t, ext, tck = get_add_mat(path, args.data_size+10, args.discard, args.ds, args.sp, 
                                      args.itr,device, dt = args.dt)
            true_y0, true_y = get_data_mat(path, args.data_size+ args.steps -1, 
                 args.discard, args.steps, args.ds, args.sp, args.itr, Default_dtype = Default_dtype)
            
        else:
            path = '../../Spoken_digit_recog_oscillator/data_experiment/Spectro_F_6mA_3497Oe_AVG1_03_11_2016.mat'
            t, ext, tck = get_add_mat(path, args.data_size+10, args.discard, args.ds, args.sp, 
                                      args.itr,device, dt = args.dt)
            true_y0, true_y = get_data_mat(path, args.data_size+ args.steps -1, 
                 args.discard, args.steps, args.ds, args.sp, args.itr, Default_dtype = Default_dtype)
        
        print(true_y.size())
        
        func = ODEFunc(tck, neural_num = args.neural_num, dropout_rate = 0, dim = true_y.size()[2], 
               ext_dim = args.steps, discard = args.discard, ext_in = args.ext_in, dt = args.dt)
        
        func.to(device)
        optimizer = optim.Adam(func.parameters(), lr=1e-3, weight_decay=0)
        
        Losses = []
        c0 = time.perf_counter()
        
        itr = 0
        while (args.Loss_min > args.Loss_min_th):
            itr = itr+1
            #print(itr)
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_ext = get_batch(args.data_size, 
                                        args.batch_time, args.batch_size, t, true_y, ext)
            pred_y = odeint(func, torch.cat((batch_y0, (batch_ext[0, :, :, 1:])), dim = -1), batch_t, method = 'rk4')
            loss = F.mse_loss(pred_y, torch.cat((batch_y, batch_ext[:, :, :, 1:]), dim = -1)) 
            loss.backward()
            optimizer.step()
            
            if itr % args.test_freq == 0:
                with torch.no_grad():
                    pred_y = odeint(func, torch.cat((true_y0, ext[0,:,1:]), dim = -1), t, method = 'rk4')
                    loss = F.mse_loss(pred_y[:,:,0:args.Loss_dim],true_y[:,:,0:args.Loss_dim])
                    Losses.append(loss.item())
                    np.savetxt('./output/Losses_'+args.method+'_dim'
                    +str(args.steps)+'_st'+str(args.torch_seed)+'_sn'+str(args.np_seed)+'.out', np.array(Losses))
                    print('Iter {:04d} | MSE Loss {:.6f}'.format(itr, loss.item()))
                    
                    if loss.item()< args.Loss_min:
                        args.Loss_min = loss.item()
                        torch.save({
                                        'model_state_dict': func.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        }, './output/params_'+args.method+'_dim'
                            +str(args.steps)+'_st'+str(args.torch_seed)+'_sn'+str(args.np_seed)+'.pkl')
                        c1 = time.perf_counter()
                        print(str(c1-c0)+ "seconds.")
        if args.plot_save:
            visualize(t/t[1],ext, torch.cat((true_y, ext), dim = -1)[:,:,0:1], pred_y[:,:,0:1], args.start, args.stop, 
                      plt_name = './output/'+args.method+'_dim'
                      +str(args.steps)+'_st'+str(args.torch_seed)+'_sn'+str(args.np_seed))

    #************* running mode: Testing mode selected
    else: ###------------test for all other speakers and iterations
        if args.method == 'co':
            path = '../../Spoken_digit_recog_oscillator/data_experiment/Cochlear_F_7mA_4482Oe_AVG1_10_10_2016.mat'
        else:
            path = '../../Spoken_digit_recog_oscillator/data_experiment/Spectro_F_6mA_3497Oe_AVG1_03_11_2016.mat'

        DATA = sio.loadmat(path)
        spokenDB_2D = DATA['spokenDB_2D']*10
        spokenDB_2D_in = DATA['spokenDB_2D_in']*10 # 

        spokenDB_2D_in_noi = add_noise_to_states(spokenDB_2D_in, args.var, args.seed_num)
        
        ode_func = ODEFunc(0, neural_num = 100, dropout_rate = 0, dim = args.steps, 
                   ext_dim = args.steps, ext_in = 'rand_txt' , dt = args.dt)
        ode_func.to(device)
        ode_func.load_state_dict(torch.load('./output/params_'+args.method+'_dim'+str(args.steps)+'.pkl')['model_state_dict'])
        
        spoken_node_2D, spoken_node_time, spoken_node_loss = test_all_speakers(ode_func, spokenDB_2D_in_noi, spokenDB_2D, 
                                  args.dt, args.steps, Default_dtype = Default_dtype)
        
        sio.savemat('./output/NODE_'+args.method+'_dim'+str(args.steps)+'_'+args.noi+'.mat', mdic)
            
            
            
            
            
            
            
            
  
