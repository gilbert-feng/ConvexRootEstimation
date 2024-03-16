To reproduce the simulation related to this paper, you can create your own "simulation" file and run the "main functions listed below".

# 1.basic function
power_sum.m	Inverse approximation function  <br>
tr_AB.m		Calculate trace of two compatible square matrices

# 2.weight matrix
matrix_hh.m	Weight matrix with k1 and k2 neighbors<br>
make_neighborsw.m  Coordinate based weight matrix with given neighbors

# 3.basic estimator
est_initial.m 	Initial IV estimator<br>
root_est.m 	    Root estimator<br>
qmle.m		    QMLE estimator<br>
gmm.m		    GMM estimator<br>
impact.m		Impact measure of ADI, ATI, AII

# 4.main function
sim_total_time.m	Computational time for different methods (not include impact)<br>
sim_total_est.m	all Estimation results for different methods (include impact)

# 5.folders
simulation	simulation results
