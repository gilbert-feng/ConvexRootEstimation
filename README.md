To reproduce the simulation related to this paper, you can create your own "simulation" file and run the "main" functions listed below.

# 1.Basic function
power_sum.m	Inverse approximation function  <br>
tr_AB.m		Calculate trace of two compatible square matrices

# 2.Weight matrix
matrix_hh.m	Weight Matrix with k1 and k2 neighbors<br>
make_neighborsw.m  Coordinate based weight matrix with given neighbors

# 3.Basic estimator
est_initial.m 	Initial IV estimator<br>
root_est.m 	    Root estimator<br>
qmle.m		    QMLE estimator<br>
gmm.m		    GMM estimator<br>
impact.m		Impact measure of ADI, ATI, AII

# 4.Main function
sim_total_time.m	Computational time for different methods (not include impact)<br>
sim_total_est.m	All estimation results for different methods (include impact)

# 5.Folders
simulation	Simulation results
