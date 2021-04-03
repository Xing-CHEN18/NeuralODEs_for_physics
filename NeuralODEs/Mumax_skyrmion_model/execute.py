import os

# define some args
seeds = [7, 8, 16, 18]


for seed in seeds:
    os.system("python main.py --torch_seed " + str(seed) + " --np_seed " + str(seed))
