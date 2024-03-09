#!/bin/bash
# default s = 20, n=500, p = 1000, beta_dist='normal'
# p_values="1000,3000,5000,7500,10000,15000,20000"
# r_values="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99"
# s_values="1,5,10,30,60,100,300,20"
p_values="1000,3000"
r_values="0.0,0.9"
s_values="20"
n_values="500"
file_name="tol"
beta_dist='normal'
reps=10
python -u simulation/main_binary.py --n_values $n_values --p_values $p_values --r_values $r_values --s_values $s_values --file_name $file_name --beta_dist $beta_dist --repetitions $reps

#--repetitions 20 --target_var 6.0
