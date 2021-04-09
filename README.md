# Neural ODEs for physics
This repository contains the code producing the results of the paper "Predicting  the results of spintronic experiments with Neural Ordinary Differential Equations" (Neural ODEs). GPU is needed for Mumax3 simulations. GPU/CPU is needed for Neural ODEs simulations. Python 3.7 and Pytoch 1.4.0 is needed.

To install MuMax3, see [MuMax3](https://github.com/mumax/3).   
To install pytorch, see [pytorch](https://pytorch.org/).    
The code for Neural ODEs was adapted from this [repo](https://github.com/rtqichen/torchdiffeq).   

The folder 'Mumax3_simulaions' contains all the Mumax simulations code. The folder 'NeuralODEs' contains all the training modules of Neural ODEs. The folder 'Mackey_Glass_series_prediction_skyrmion' contains code for Mackey-Glass prediction task using Mumax output and Neural ODE output. 'Spoken_digit_recog_oscillator' contains code for spoken digit recognition task.

## Mumax3 simulations

Run the '.mx3' file in each folder of Mumax3_simulations.  
To obtain the training data of voltage-induced skyrmion system, run:
```sh
#one skyrmion system
nohup ./mumax3 -gpu 0 1skyrmion_voltage_train.mx3 > outputfile &
``` 
```sh
#multiple skyrmion system
nohup ./mumax3 -gpu 0 4skyrmion_voltage_train.mx3 > outputfile &
``` 
To obtain the results of time-varing output from reservoir of skyrmion system using Mumax3, run:
```sh
#one skyrmion system, requires simulation time of 2~3 days
nohup ./mumax3 -gpu 0 1skyrmion_voltage_mackey_glass.mx3 > outputfile &
``` 
```sh
#multiple skyrmion system, requires simulation time of more than 4 days
nohup ./mumax3 -gpu 0 4skyrmion_voltage_mackey_glass.mx3 > outputfile &
``` 

## Training voltage-induced skyrmion dynamics  

To obtain the results of Fig. 2(b) for one skyrmion system, run train mode of main.py in the folder of NeuralODEs/Mumax_skyrmion_model:  
```sh
#NODE k = 1
python main.py --steps 1   
```
```sh
#NODE k = 2
python main.py  
```
Change the seed number randomly:  
```sh
#NODE k = 2
python main.py --torch_seed 7 --np_seed 7   
```
To run different seed number consecutively:
```sh
#NODE k = 2
python execute.py    
```

To obtain results of Fig. 2(c) for multiple skyrmions system, run:  
```sh
#NODE k = 1
python main.py --name 4skyrmion_voltage --steps 1 --data_size 15000 --Loss_min_th 5e-3  
```
```sh
#NODE k = 2
python main.py --name 4skyrmion_voltage --data_size 15000 --Loss_min_th 5e-3  
```
Change the seed number randomly:  
```sh
#NODE k = 2
python main.py --name 4skyrmion_voltage --steps 1 --data_size 15000 --Loss_min_th 5e-3 --torch_seed 8 --np_seed 10   
```
## Mackey-Glass time series prediction using skyrmion system with voltage as input
To obtain the results of time-varing output from reservoir of skyrmion system using the trained model, run test mode of main.py in the folder of NeuralODEs/Mumax_skyrmion_model:
```sh
#one skyrmion system
python main.py -ts   
```
```sh
#multiple skyrmion system
python main.py -ts --name 4skyrmion_voltage --mg_scale_factor 0.25
```
To obtain the results of mackey-glass prediction in Fig. 3(b) and 3(c), run main.py in the folder of Mackey_Glass_series_prediction_skyrmion:  
```sh
#one skyrmion system
python main.py 
```
```sh
#multiple skyrmion system
python main.py --name 4skyrmion_voltage
```
To obtain the results of Fig. 3(c) and 3(e),  
```sh
#one skyrmion system
python main.py --h_max 7 -pe
```
## Training parameters based model
To obtain the results of Fig. 4(c-d), run train mode of main.py in the folder of NeuralODEs/Mumax_parameters_model_tr_ts:  
```sh
python main.py 
```
To obtain the results of Fig. 4(e-f), run test mode of main.py in the folder of NeuralODEs/Mumax_parameters_model_tr_ts:  
```sh
#Fig. 4(e)
python main.py -ts --ext_in 'rand_txt' --discard 0 --data_size 16800
```
```sh
#Fig. 4(f)
python main.py -ts -dmi --ext_in 'rand_txt' --discard 0 --data_size 12800 --num_pars 16
```
## Training experimental spintronic oscillator dynamics
To train the NODE model of oscillators, run train mode of main.py in the folder of NeuralODEs/Experiment_oscillator_model:  
```sh
#For cochlear method, k = 2
python main.py --steps 2
```
```sh
#For spectrogram method, k = 2
python main.py --method 'spc' --steps 2
```
To obtain the results of NODE output, run the test mode:
```sh
#For cochlear method, k = 2
python main.py -ts --steps 2
```
```sh
#For spectrogram method, k = 2
python main.py  --method 'spc' -ts --steps 2
```
## spoken digit recognition task using oscillators
To obtain the results of recognition rate, run main.py in the folder of Spoken_digit_recog_oscillator :
```sh
#For cochlear method, k = 2
python main.py 
python main.py -ts 
```
```sh
#For spectrogram method, k = 2
python main.py  --method 'spc' 
python main.py --method 'spc' -ts 
```
 
 
 
 
 
