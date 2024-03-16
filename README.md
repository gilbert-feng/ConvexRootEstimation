To reproduce the simulation related to this paper, you can create your own "simulation" file and run the "main functions listed below".

# 1.basic function
power_sum.m	inverse approximation function
tr_AB.m		calculate trace of two compatible square matrices

# 2.weight matrix
matrix_hh.m	weight matrix with k1 and k2 neighbors
make_neighborsw.m  coordinate based weight matrix with given neighbors

# 3.basic estimator
est_initial.m 	initial IV estimator
root_est.m 	root estimator
qmle.m		QMLE estimator
gmm.m		GMM estimator
impact.m		impact measure of ADI, ATI, AII

# 4.main function
sim_total_time.m	computational time for different methods (not include impact)
sim_total_est.m	all estimation results for different methods (include impact)

# 5.folders
simulation	simulation results
