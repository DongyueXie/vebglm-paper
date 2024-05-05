#!/bin/bash
# default s = 20, n=500, p = 1000, beta_dist='normal'
# p_values="20,50,100,200,500,1000,2000,3000"
# r_values="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99"
# s_values="1,5,10,30,50,100,200,300"
beta_dist='normal,uniform,const,laplace'
# n_values="200,300,500,750,1000,1500,2000,3000"
p_values="1000"
r_values="0.0"
s_values="20"
n_values="500"
file_name="beta_dist"
# beta_dist='normal'
reps=20
python -u simulation/main_binary.py --n_values $n_values --p_values $p_values --r_values $r_values --s_values $s_values --file_name $file_name --beta_dist $beta_dist --repetitions $reps


