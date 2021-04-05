# Neural ODEs for physics
This repository contains the code producing the results of the paper "Predicting  the results of spintronic experiments with Neural Ordinary Differential Equations" (Neural ODEs). GPU is needed for Mumax3 simulations. GPU/CPU is needed for Neural ODEs simulations. 

To install Mumax3:  

To set the environment run in your conda main environment:   


The folder 'Mumax3_simulaions' contains all the Mumax simulations code. The folder 'NeuralODEs' contains all the training modules of Neural ODEs.  

The code for Neural ODEs was adapted from this [repo](https://github.com/rtqichen/torchdiffeq).   

## Training voltage-induced skyrmion dynamics  

To obtain the results of Fig. 2(b) for one skyrmion system, run:  
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
## Mackey-Glass time series prediction using skyrmion system
To obtain the results of time-varing output from reservoir of skyrmion system, run:
```sh
#one skyrmion system
python main.py -ts   
```

 
 
 
 
 
 
