import numpy as np
import pylab as plt
import copy
import math

from scipy.optimize import minimize
plt.close('all')

C1 = 0.424

C2_1 = 0.666
C2_2 = 0.497

C3_1 = 0.333
C3_2 = 0.754
C3_3 = 0.671

C4_1 = 0.281
C4_2 = 0.666

C5_1 = 0.00002
C5_2 = 0.57105

C6_1 = 0.43
C6_2 = 0.600
C6_3 = 0.673

C0 =np.array([C1, C2_1, C2_2, C3_1, C3_2, C3_3, C4_1, C4_2, C5_1, C5_2, C6_1, C6_2, C6_3])

def RSS_func(C,flag=0.0):

    Cfilm_M1 = C[0]

    Cfilm_M2 = C[1]
    Clam_M2 = C[2]

    Clam1_M3 = C[-10]
    Clam2_M3 = C[-9]
    Cfilm_M3 = C[-8]

    Clam_M4 = C[-7]
    Cbulk_M4 = C[-6]

    Cint_M5 = C[-5]
    Cfilm_M5 = C[-4]

    Cint_M6 = C[-3]
    Clam_M6 = C[-2]
    Cfilm_M6 = C[-1]

    T = 302.75

    t_Karan = np.array([4.0, 10.0, 30.0, 55.0, 160.0, 300.0])
    sigma_30_Karan = np.array([21.481, 22.062, 20.185, 34.810, 43.913])/10.
    sigma_25_Karan = np.array([21.888, 23.9213, 23.9051, 22.753,54.0428])/10.
    sigma_50_Karan = np.array([37.686, 57.339, 48.934, 72.006, 83.923])/10.
    sigma_40_Karan = np.array([30.321, 48.799, 40.439, 68.326, 68.334])/10.
    sigma_60_Karan = np.array([55.18, 62.39, 63.20, 93.33, 100.60])/10.

    V_wa = np.array([0.214, 0.236, 0.153, 0.165, 0.160, 0.181, 0.254, 0.254, 0.261, 0.259])

    V_wb = np.array([0.124, 0.152, 0.147, 0.176, 0.246, 0.246, 0.257, 0.255])

    SLDs = np.array([[-0.2237, 4.5043, 1.6979, 4.15743, 4.15743, 4.15743, 4.15743],
                    [-0.2237, 4.5043, 1.6979, 3.9113, 4.15743, 4.15743, 4.15743], #
                    [-0.0958, 4.373, 2.583, 3.791, 2.6508, 4.15743, 4.15743],
                    [0.6430, 4.8930, 2.5818, 4.077, 3.3162, 3.870, 3.5726],
                    [0.7325, 4.3528, 2.9012, 3.7917, 3.2255, 3.5664, 3.4425],
                    [0.1142, 4.2299, 2.9748, 3.8117, 3.4024, 3.6089, 3.4619],
                    [0.8616, 4.3067, 2.8787, 3.7708, 3.2241, 3.5187, 3.3272],
                    [1.3948, 4.3831, 2.3515, 3.2838, 2.9463, 2.9985, 4.15743],
                    [-0.5128, 4.4076, 1.1305, 3.4066, 2.1211, 2.9965, 3.3793],
                    [-0.3722, 3.5109, 1.4651, 3.2542, 2.5756, 2.9446, 3.1720],
                    [-0.4081, 4.0207, 2.1275, 3.2554, 2.3396, 2.9550, 4.15743],
                    [2.97, 2.97, 2.97, 2.97, 2.97, 2.97, 2.97]])

    t_i = np.array([[8.612, 24.118, 10.505, 0, 0, 0, 0], #fake 4nm
                    [8.612, 24.118, 10.505, 15.281, 0, 0, 0], #t5
                    [7.5947, 20.907, 15.228, 18.3322, 26.467, 0, 0], #t7
                    [10.219, 18.002, 17.477, 14.441, 26.715, 5.352, 44.316], #t12 -7th
                    [9.951, 16.605, 14.805, 19.240, 12.915, 16.200, 123.3766], #t18  -7th
                    [7.909, 18.605, 14.979, 15.534, 22.633, 9.266, 143.297], #t20 -t
                    [10.866, 16.236, 16.600, 16.202, 21.114, 11.830, 417.886], #t42
                    [21.295, 10.124, 23.221, 14.818, 16.307, 720.537, 0], #t60
                    [11.505, 14.579, 21.258, 18.612, 17.675, 1257.711, 35.4610], #t103
                    [9.481, 22.070, 16.776, 19.297, 18.831, 1477.106, 56.649], #t120
                    [11.783, 16.619, 23.806, 18.424, 12.076, 1992.758, 0.0], #t154
                    [15.423, 15.586, 25.244, 11.066, 18.562, 29.594, 1971.04]])

    t_i_norm = copy.deepcopy(t_i)
    t_i_norm[t_i_norm==0] = 1.0

    V_w = (4.15743 - SLDs)/(4.15743 + 0.559883)

    V_w[3,3] = 0.077

    Lambda_i = 1.006*1100*V_w/(1-V_w)/18.016/1.98

    Lambda_calc_low = np.array([34.6, 7.1])

    SLD_high = np.array([0.32, 3.78])
    V_w_high = (4.15743 - SLD_high)/(4.15743 + 0.559883)
    Lambda_calc_high = 1.006*1100*V_w_high/(1-V_w_high)/18.016/1.98

    Lambda_i[0:7,0:2] = Lambda_calc_low[0:2]
    Lambda_i[7:-1,0:2] = Lambda_calc_high[0:2]

    sigma_io_i = math.exp(1286.*(1./303. - 1./T))*(0.514*Lambda_i - 0.326)

    #print( sigma_io_i )

    a_w_ca = 0.92;
    lambda_bulk = (0.043 + 17.81*a_w_ca - 39.85*a_w_ca**2 + 36.0*a_w_ca**3);


    V_water_Bulk = 1.98*18.016*lambda_bulk/1.006/1100.0/(1 + 1.98*18.016*lambda_bulk/1.006/1100.0)

    sigma_io_i_M1 = copy.deepcopy(sigma_io_i)
    sigma_io_i_M2 = copy.deepcopy(sigma_io_i)
    sigma_io_i_M3 = copy.deepcopy(sigma_io_i)
    sigma_io_i_M4 = copy.deepcopy(sigma_io_i)
    sigma_io_i_M5 = copy.deepcopy(sigma_io_i)
    sigma_io_i_M6 = copy.deepcopy(sigma_io_i)

    sigma_io_i_M1 *= Cfilm_M1

    sigma_io_i_M2 *= Cfilm_M2
    sigma_io_i_M2[:,0:5] *= Clam_M2
    """sigma_io_i_M2[3,6] *= (1.0-Cbulk_M2*(V_water_Bulk - V_w[3,6]))
    #sigma_io_i[4,6] *= (1.0-bulkFac*(0.25 - V_w[4,6]))
    #sigma_io_i[5,6] *= (1.0-bulkFac*(0.25 - V_w[5,6]))
    sigma_io_i_M2[6,6] *= (1.0-Cbulk_M2*(V_water_Bulk - V_w[6,6]))
    sigma_io_i_M2[7,5] *= (1.0-Cbulk_M2*(V_water_Bulk - V_w[7,5]))
    sigma_io_i_M2[8,5] *= (1.0-Cbulk_M2*(V_water_Bulk - V_w[8,5]))
    sigma_io_i_M2[9,5] *= (1.0-Cbulk_M2*(V_water_Bulk - V_w[9,5]))"""

    sigma_io_i_M3 *= Cfilm_M3
    #sigma_io_i_M3[:,0:5] *= Clam_M3
    #sigma_io_i_M3[:,0] *= Clam1_M3
    sigma_io_i_M3[:,0] *= ((1.0 - 0./4.)*Clam1_M3 + (0./4.)*Clam2_M3)
    sigma_io_i_M3[:,1] *= ((1.0 - 1./4.)*Clam1_M3 + (1./4.)*Clam2_M3)
    sigma_io_i_M3[:,2] *= ((1.0 - 2./4.)*Clam1_M3 + (2./4.)*Clam2_M3)
    sigma_io_i_M3[:,3] *= ((1.0 - 3./4.)*Clam1_M3 + (3./4.)*Clam2_M3)
    sigma_io_i_M3[:,4] *= ((1.0 - 4./4.)*Clam1_M3 + (4./4.)*Clam2_M3)

    sigma_io_i_M4 *= Cbulk_M4
    Clam2_M4 = 1.0
    sigma_io_i_M4[:,0] *= ((1.0 - 0./5.)*Clam_M4 + (0./5.)*Clam2_M4)
    sigma_io_i_M4[:,1] *= ((1.0 - 1./5.)*Clam_M4 + (1./5.)*Clam2_M4)
    sigma_io_i_M4[:,2] *= ((1.0 - 2./5.)*Clam_M4 + (2./5.)*Clam2_M4)
    sigma_io_i_M4[:,3] *= ((1.0 - 3./5.)*Clam_M4 + (3./5.)*Clam2_M4)
    sigma_io_i_M4[:,4] *= ((1.0 - 4./5.)*Clam_M4 + (4./5.)*Clam2_M4)

    sigma_io_i_M5 *= Cfilm_M5
    sigma_io_i_M5[:,0] *= Cint_M5

    sigma_io_i_M6 *= Cfilm_M6
    sigma_io_i_M6[:,0:5] *= Clam_M6
    sigma_io_i_M6[:,0] *= Cint_M6

    """sigma_io_i_M3[3,6] *= (1.0-Cbulk_M3*(V_water_Bulk - V_w[3,6]))
    #sigma_io_i[4,6] *= (1.0-bulkFac*(0.25 - V_w[4,6]))
    #sigma_io_i[5,6] *= (1.0-bulkFac*(0.25 - V_w[5,6]))
    sigma_io_i_M3[6,6] *= (1.0-Cbulk_M3*(V_water_Bulk - V_w[6,6]))
    sigma_io_i_M3[7,5] *= (1.0-Cbulk_M3*(V_water_Bulk - V_w[7,5]))
    sigma_io_i_M3[8,5] *= (1.0-Cbulk_M3*(V_water_Bulk - V_w[8,5]))
    sigma_io_i_M3[9,5] *= (1.0-Cbulk_M3*(V_water_Bulk - V_w[9,5]))"""

    #print( sigma_io_i )

    R_io_trans_eff_M1 = np.zeros(t_i.shape[0]-1)
    R_io_trans_eff_M2 = np.zeros(t_i.shape[0]-1)
    R_io_trans_eff_M3 = np.zeros(t_i.shape[0]-1)
    R_io_trans_eff_M4 = np.zeros(t_i.shape[0]-1)
    R_io_trans_eff_M5 = np.zeros(t_i.shape[0]-1)
    R_io_trans_eff_M6 = np.zeros(t_i.shape[0]-1)


    sigma_io_trans_eff_M1 = np.zeros(t_i.shape[0]-1)
    sigma_io_trans_eff_M2 = np.zeros(t_i.shape[0]-1)
    sigma_io_trans_eff_M3 = np.zeros(t_i.shape[0]-1)
    sigma_io_trans_eff_M4 = np.zeros(t_i.shape[0]-1)
    sigma_io_trans_eff_M5 = np.zeros(t_i.shape[0]-1)
    sigma_io_trans_eff_M6 = np.zeros(t_i.shape[0]-1)

    sigma_io_par_eff_M1 = np.zeros(t_i.shape[0]-1)
    sigma_io_par_eff_M2 = np.zeros(t_i.shape[0]-1)
    sigma_io_par_eff_M3 = np.zeros(t_i.shape[0]-1)
    sigma_io_par_eff_M4 = np.zeros(t_i.shape[0]-1)
    sigma_io_par_eff_M5 = np.zeros(t_i.shape[0]-1)
    sigma_io_par_eff_M6 = np.zeros(t_i.shape[0]-1)

    t_film = np.zeros(t_i.shape[0]-1)

    Lambda_avg = 1.006*1100.*V_wa/(1.-V_wa)/18.016/1.98
    sigma_io_avg = math.exp(1286.*(1./303. - 1./T))*(0.514*Lambda_avg - 0.326)

    Lambda_outer = 1.006*1100.*V_wb/(1.-V_wb)/18.016/1.98
    sigma_io_outer = math.exp(1286.*(1./303. - 1./T))*(0.514*Lambda_outer - 0.326)

    for j in np.arange(t_i.shape[0]-1):
        t_film[j] = sum(t_i[j,:])
        for i in np.arange(t_i.shape[1]):
            sigma_io_par_eff_M6[j] += sigma_io_i_M6[j,i]*t_i[j,i]/t_film[j]
            sigma_io_par_eff_M5[j] += sigma_io_i_M5[j,i]*t_i[j,i]/t_film[j]
            sigma_io_par_eff_M4[j] += sigma_io_i_M4[j,i]*t_i[j,i]/t_film[j]
            sigma_io_par_eff_M3[j] += sigma_io_i_M3[j,i]*t_i[j,i]/t_film[j]
            sigma_io_par_eff_M2[j] += sigma_io_i_M2[j,i]*t_i[j,i]/t_film[j]
            sigma_io_par_eff_M1[j] += sigma_io_i_M1[j,i]*t_i[j,i]/t_film[j]
            #print( t_i[j,i], sigma_io_i[j,i] )
            R_io_trans_eff_M1[j] += t_i[j,i]/sigma_io_i_M1[j,i]/t_film[j]
            R_io_trans_eff_M2[j] += t_i[j,i]/sigma_io_i_M2[j,i]/t_film[j]
            R_io_trans_eff_M3[j] += t_i[j,i]/sigma_io_i_M3[j,i]/t_film[j]
            R_io_trans_eff_M4[j] += t_i[j,i]/sigma_io_i_M4[j,i]/t_film[j]
            R_io_trans_eff_M5[j] += t_i[j,i]/sigma_io_i_M5[j,i]/t_film[j]
            R_io_trans_eff_M6[j] += t_i[j,i]/sigma_io_i_M6[j,i]/t_film[j]



    #print( t_film[[1,2,6,9]] )
    #print( 1/(1.214/(10**(-8)*t_film[[1]])) )
    #print( 1/(2.857/(10**(-8)*t_film[[2]])) )
    #print( 1/(5.0315/(10**(-8)*t_film[[6]])) )

    sigma_io_trans_eff_M1 = 1/R_io_trans_eff_M1
    sigma_io_trans_eff_M2 = 1/R_io_trans_eff_M2
    sigma_io_trans_eff_M3 = 1/R_io_trans_eff_M3
    sigma_io_trans_eff_M4 = 1/R_io_trans_eff_M4
    sigma_io_trans_eff_M5 = 1/R_io_trans_eff_M5
    sigma_io_trans_eff_M6 = 1/R_io_trans_eff_M6

    #print( sigma_io_par_eff_M3 )

    sigma_io_bulk = math.exp(1286.*(1./303. - 1./T))*(0.514*lambda_bulk - 0.326)

    #print( (sigma_io_par_eff_M1[1] - sigma_30_Karan[0])**2 )
    #print( (sigma_io_par_eff_M1[2] - sigma_30_Karan[1])**2 )
    #print( (sigma_io_par_eff_M1[6] - sigma_30_Karan[2])**2 )
    #print( (sigma_io_par_eff_M1[9] - sigma_30_Karan[3])**2 )

    RSS_1 = (sigma_io_par_eff_M1[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M1[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M1[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M1[9] - sigma_30_Karan[3])**2
    RSS_2 = (sigma_io_par_eff_M2[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M2[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M2[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M2[9] - sigma_30_Karan[3])**2
    RSS_3 = (sigma_io_par_eff_M3[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M3[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M3[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M3[9] - sigma_30_Karan[3])**2
    RSS_4 = (sigma_io_par_eff_M4[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M4[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M4[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M4[9] - sigma_30_Karan[3])**2
    RSS_5 = (sigma_io_par_eff_M5[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M5[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M5[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M5[9] - sigma_30_Karan[3])**2
    RSS_6 = (sigma_io_par_eff_M6[1] - sigma_30_Karan[0])**2 + (sigma_io_par_eff_M6[2] - sigma_30_Karan[1])**2 + (sigma_io_par_eff_M6[6] - sigma_30_Karan[2])**2 + (sigma_io_par_eff_M6[9] - sigma_30_Karan[3])**2

    if flag==0:
        return RSS_1 + RSS_2 + RSS_3 + RSS_4 + RSS_5 + RSS_6
    elif flag ==1:
        return sigma_io_par_eff_M1
    elif flag ==2:
        return sigma_io_par_eff_M2
    elif flag ==3:
        return sigma_io_par_eff_M3
    elif flag ==4:
        return sigma_io_par_eff_M4
    elif flag ==5:
        return sigma_io_par_eff_M5
    elif flag ==6:
        return sigma_io_par_eff_M6
    elif flag == -1:
        return RSS_1
    elif flag == -2:
        return RSS_2
    elif flag == -3:
        return RSS_3
    elif flag == -4:
        return RSS_4
    elif flag == -5:
        return RSS_5
    elif flag == -6:
        return RSS_6
    elif flag == -7:
        return t_film
    elif flag == -8:
        return sigma_io_trans_eff_M3


#print( RSS_1, RSS_2, RSS_3, RSS_4, RSS_5, RSS_6 )

bounds = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],
                   [0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]])
res = minimize(RSS_func, C0,method='L-BFGS-B',bounds=bounds,options={'disp':True,'xtol':1e-3},args=(0.0,))
print( res.x[3:6] )

#print( V_wa, V_wa_U )
ms = 3.5
UL = 4.0
ff = 10
yt = [0,15,30, 45]
cs = 4.5
lw = 1.5

fig = plt.figure()
ax = fig.add_axes([0.175,0.175,0.775,0.75])
fig.set_size_inches((3,2.25))

#print tNaf_outer, sigma_io_outer
t_film = RSS_func(res.x,-7.0)
#print( t_film )
sigma_io_par_eff_M1 = RSS_func(res.x,1.0)
sigma_io_par_eff_M2 = RSS_func(res.x,2.0)
sigma_io_par_eff_M3 = RSS_func(res.x,3.0)
sigma_io_par_eff_M4 = RSS_func(res.x,4.0)
sigma_io_par_eff_M5 = RSS_func(res.x,5.0)
sigma_io_par_eff_M6 = RSS_func(res.x,6.0)


#print( RSS_func(res.x,-1.0) )
#print( RSS_func(res.x,-2.0) )
#print( RSS_func(res.x,-3.0) )
#print( RSS_func(res.x,-4.0) )
#print( RSS_func(res.x,-5.0) )
#print( RSS_func(res.x,-6.0) )
#print( 'Done' )


T = 302.75

t_Karan = np.array([4.0, 10.0, 30.0, 55.0, 160.0, 300.0])
sigma_30_Karan = np.array([21.481, 22.062, 20.185, 34.810, 43.913])/10.
sigma_25_Karan = np.array([21.888, 23.9213, 23.9051, 22.753,54.0428])/10.
sigma_50_Karan = np.array([37.686, 57.339, 48.934, 72.006, 83.923])/10.
sigma_40_Karan = np.array([30.321, 48.799, 40.439, 68.326, 68.334])/10.
sigma_60_Karan = np.array([55.18, 62.39, 63.20, 93.33, 100.60])/10.


cmap = plt.get_cmap('hsv')
ndata = 4
colors = np.linspace(0,0.85,ndata)

plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'x',markeredgecolor = 'k',markersize=4.0,color='none',zorder=10,markeredgewidth=1.0)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M1[[1,2,6,9]],'s',color = cmap(colors[1]),markersize=ms,markeredgecolor='none',zorder=1)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M2[[1,2,6,9]],'^',color = cmap(colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M4[[1,2,6,9]],'D',color = cmap(colors[0]),markersize=ms,markeredgecolor='none',zorder=3)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M3[[1,2,6,9]],'o',color = 'k',markersize=ms,markeredgecolor='none',zorder=4)
plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'-',zorder=10,color = 'k',linewidth=1.0)
#,markeredgecolor='none')
#plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'D',color = cmap(colors[2]),markersize=ms,markeredgecolor='none')


"""plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_par_eff[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[0]),markersize=ms,markeredgecolor='none')
plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_trans_eff[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[3]),markersize=ms,markeredgecolor='none')
plt.plot(tNaf_outer[[0,2,3,4,5,6,7]],sigma_io_outer[[0,2,3,4,5,6,7]], 'D',color = cmap(colors[2]),markersize=ms,markeredgecolor='none')
plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_avg[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[1]),markersize=ms,markeredgecolor='none')
plt.plot([0,160],[sigma_io_bulk,sigma_io_bulk],'--',color='0.5')

plt.plot(tNaf, V_wl, 'bD',markersize=ms,markeredgecolor='none')
plt.errorbar(tNaf,V_wl,xerr=[tNaf_L,tNaf_U],yerr=[V_wl_L,V_wl_U],capsize=cs,elinewidth=lw,ecolor='b',fmt=None)

plt.plot(tNb, V_wb, 'rD',markersize=ms,markeredgecolor='none')
plt.errorbar(tNb,V_wb,xerr=[tNb_L,tNb_U],yerr=[V_wb_L,V_wb_U],capsize=cs,elinewidth=lw,ecolor='r',fmt=None)


plt.plot(tNaf,V_wa,'kD',markersize=ms,markeredgecolor='none')
plt.errorbar(tNaf,V_wa,xerr=[tNaf_L, tNaf_U],yerr=[V_wa_L, V_wa_U],capsize=cs, elinewidth=lw,ecolor='k',fmt=None)"""



plt.ylim(0,UL)
plt.xlim(0,170)
plt.xticks([0,40,80,120,160])
plt.yticks([0.0,1.0,2.0,3.0,4.0])

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=8)
plt.legend(['Measured','Model 1, SSR = 2.393','Model 2, SSR = 0.104','Model 3, SSR = 0.017','Model 4, SSR = 0.001'],prop=font,handletextpad=-0.1,numpoints=1,loc=4, frameon=False)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()

majorFormatter = plt.FormatStrFormatter('%0.1f')
ax.yaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)

#ax.annotate('a',xy=[10,3.5],family='Arial',size=24)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=3.0)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=3.0)

#plt.savefig('Fig8a_sigma_v_data_30C_05232017.pdf',dpi=500,format='pdf')




fig = plt.figure()
ax = fig.add_axes([0.175,0.175,0.775,0.75])
fig.set_size_inches((3,2.25))

cmap = plt.get_cmap('hsv')
ndata = 6
colors = np.linspace(0,0.85,ndata)

plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'x',markeredgecolor = 'k',markersize=4.0,color='none',zorder=10,markeredgewidth=1.0)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M5[[1,2,6,9]],'s',color = cmap(colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M6[[1,2,6,9]],'^',color = cmap(colors[4]),markersize=ms,markeredgecolor='none',zorder=2)
plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'-',zorder=10,color = 'k',linewidth=1.0)
#,markeredgecolor='none')
#plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'D',color = cmap(colors[2]),markersize=ms,markeredgecolor='none')


"""plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_par_eff[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[0]),markersize=ms,markeredgecolor='none')
    plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_trans_eff[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[3]),markersize=ms,markeredgecolor='none')
    plt.plot(tNaf_outer[[0,2,3,4,5,6,7]],sigma_io_outer[[0,2,3,4,5,6,7]], 'D',color = cmap(colors[2]),markersize=ms,markeredgecolor='none')
    plt.plot(tNaf[[1,2,4,5,6,7,8,9]],sigma_io_avg[[1,2,4,5,6,7,8,9]], 'D',color = cmap(colors[1]),markersize=ms,markeredgecolor='none')
    plt.plot([0,160],[sigma_io_bulk,sigma_io_bulk],'--',color='0.5')

    plt.plot(tNaf, V_wl, 'bD',markersize=ms,markeredgecolor='none')
    plt.errorbar(tNaf,V_wl,xerr=[tNaf_L,tNaf_U],yerr=[V_wl_L,V_wl_U],capsize=cs,elinewidth=lw,ecolor='b',fmt=None)

    plt.plot(tNb, V_wb, 'rD',markersize=ms,markeredgecolor='none')
    plt.errorbar(tNb,V_wb,xerr=[tNb_L,tNb_U],yerr=[V_wb_L,V_wb_U],capsize=cs,elinewidth=lw,ecolor='r',fmt=None)


    plt.plot(tNaf,V_wa,'kD',markersize=ms,markeredgecolor='none')
    plt.errorbar(tNaf,V_wa,xerr=[tNaf_L, tNaf_U],yerr=[V_wa_L, V_wa_U],capsize=cs, elinewidth=lw,ecolor='k',fmt=None)"""

LL = 1.25

plt.ylim(LL,3.75)
plt.xlim(0,170)
plt.xticks([0,40,80,120,160])
plt.yticks([2.0,3.0])

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=8)
plt.legend(['Measured','Model 5, SSR = 0.651','Model 6, SSR = 0.008'],prop=font,handletextpad=-0.1,numpoints=1,loc=4, frameon=False)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()

majorFormatter = plt.FormatStrFormatter('%0.1f')
ax.yaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)

#ax.annotate('a',xy=[10,3.5],family='Arial',size=24)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=3.0)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=3.0)

#plt.savefig('FigS2_sigma_v_data_30C_Models5_6.pdf',dpi=500,format='pdf')





" 50 C "
"Activation energies from Paul, et al supporting info [J/mol]:"
Ea_4a = 3269.3#25.45*1000
Ea_10a = 7528.8
Ea_55a = 6589.9
Ea_160a = 6395.8

Ea_4b = 3110.6
Ea_10b = 1284.2
Ea_55b = 2322.7
Ea_160b = 1613.6


#print sigma_io_par_eff_M3[9]

"""A_4 = sigma_25_Karan[0]/math.exp(-Ea_4/298.15)
A_10 = sigma_25_Karan[1]/math.exp(-Ea_10/298.15)
A_55 = sigma_25_Karan[3]/math.exp(-Ea_55/298.15)
A_160 = sigma_25_Karan[4]/math.exp(-Ea_160/298.15)"""

T1 = 30
"""A_4_M3 = sigma_io_par_eff_M3[1]/math.exp(-Ea_4/(273.15+T))
A_10_M3 = sigma_io_par_eff_M3[2]/math.exp(-Ea_10/(273.15+T))
A_55_M3 = sigma_io_par_eff_M3[6]/math.exp(-Ea_55/(273.15+T))
A_160_M3 = sigma_io_par_eff_M3[9]/math.exp(-Ea_160/(273.15+T))


#print A_4, A_10, A_55, A_160
fac50 = 1.4

sigma_50_pred_4 = A_4*math.exp(-Ea_4/(273.15+50))
sigma_50_pred_10 = A_10*math.exp(-Ea_10/(273.15+50))
sigma_50_pred_55 = A_55*math.exp(-Ea_55/(273.15+50))
sigma_50_pred_160 = A_160*math.exp(-Ea_160/(273.15+50))"""

T2 = 40
"""sigma_M3_4 = A_4_M3*math.exp(-Ea_4/(273.15+T2))
sigma_M3_10 = A_10_M3*math.exp(-Ea_10/(273.15+T2))
sigma_M3_55 = A_55_M3*math.exp(-Ea_55/(273.15+T2))
    sigma_M3_160 = A_160_M3*math.exp(-Ea_160/(273.15+T2))"""

sigma_M1_4_40 = sigma_io_par_eff_M1[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M1_10_40 = sigma_io_par_eff_M1[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M1_55_40 = sigma_io_par_eff_M1[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M1_160_40 = sigma_io_par_eff_M1[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_M2_4_40 = sigma_io_par_eff_M2[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M2_10_40 = sigma_io_par_eff_M2[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M2_55_40 = sigma_io_par_eff_M2[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M2_160_40 = sigma_io_par_eff_M2[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_M4_4_40 = sigma_io_par_eff_M4[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M4_10_40 = sigma_io_par_eff_M4[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M4_55_40 = sigma_io_par_eff_M4[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M4_160_40 = sigma_io_par_eff_M4[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_M3_4_40 = sigma_io_par_eff_M3[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_10_40 = sigma_io_par_eff_M3[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_55_40 = sigma_io_par_eff_M3[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_160_40 = sigma_io_par_eff_M3[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))



sigma_M5_4_40 = sigma_io_par_eff_M5[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M5_10_40 = sigma_io_par_eff_M5[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M5_55_40 = sigma_io_par_eff_M5[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M5_160_40 = sigma_io_par_eff_M5[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_M6_4_40 = sigma_io_par_eff_M6[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M6_10_40 = sigma_io_par_eff_M6[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M6_55_40 = sigma_io_par_eff_M6[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M6_160_40 = sigma_io_par_eff_M6[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_io_trans_eff_M3 = RSS_func(res.x,-8.0)

sigma_trans_4_40 = sigma_io_trans_eff_M3[1]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_10_40 = sigma_io_trans_eff_M3[2]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_55_40 = sigma_io_trans_eff_M3[6]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_160_40 = sigma_io_trans_eff_M3[9]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

T3 = 50
"""sigma_M3_4 = A_4_M3*math.exp(-Ea_4/(273.15+T2))
    sigma_M3_10 = A_10_M3*math.exp(-Ea_10/(273.15+T2))
    sigma_M3_55 = A_55_M3*math.exp(-Ea_55/(273.15+T2))
    sigma_M3_160 = A_160_M3*math.exp(-Ea_160/(273.15+T2))"""

sigma_M1_4_50 = sigma_M1_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M1_10_50 = sigma_M1_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M1_55_50 = sigma_M1_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M1_160_50 = sigma_M1_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_M2_4_50 = sigma_M2_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M2_10_50 = sigma_M2_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M2_55_50 = sigma_M2_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M2_160_50 = sigma_M2_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_M4_4_50 = sigma_M4_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M4_10_50 = sigma_M4_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M4_55_50 = sigma_M4_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M4_160_50 = sigma_M4_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_M3_4_50 = sigma_M3_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_10_50 = sigma_M3_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_55_50 = sigma_M3_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_160_50 = sigma_M3_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_M5_4_50 = sigma_M5_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M5_10_50 = sigma_M5_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M5_55_50 = sigma_M5_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M5_160_50 = sigma_M5_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_M6_4_50 = sigma_M6_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M6_10_50 = sigma_M6_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M6_55_50 = sigma_M6_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M6_160_50 = sigma_M6_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_trans_4_50 = sigma_trans_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_10_50 = sigma_trans_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_55_50 = sigma_trans_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_160_50 = sigma_trans_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

#print( 'sigma_par' )
#print( sigma_M3_4_50, sigma_M3_10_50, sigma_M3_55_50, sigma_M3_160_50 )
#
#print( 'sigma_trans' )
#print( sigma_trans_4_50, sigma_trans_10_50, sigma_trans_55_50, sigma_trans_160_50 )


"""print sigma_50_pred_4, sigma_M3_4, sigma_50_Karan[0]

    print sigma_50_pred_10, sigma_M3_10, sigma_50_Karan[1]

    print sigma_50_pred_55, sigma_M3_55, sigma_50_Karan[2]

    print sigma_50_pred_160, sigma_M3_160, sigma_50_Karan[3]"""

T4 = 60


sigma_M1_4_60 = sigma_M1_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M1_10_60 = sigma_M1_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M1_55_60 = sigma_M1_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M1_160_60 = sigma_M1_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_M2_4_60 = sigma_M2_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M2_10_60 = sigma_M2_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M2_55_60 = sigma_M2_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M2_160_60 = sigma_M2_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_M4_4_60 = sigma_M4_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M4_10_60 = sigma_M4_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M4_55_60 = sigma_M4_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M4_160_60 = sigma_M4_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_M3_4_60 = sigma_M3_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_10_60 = sigma_M3_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_55_60 = sigma_M3_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_160_60 = sigma_M3_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_M5_4_60 = sigma_M5_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M5_10_60 = sigma_M5_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M5_55_60 = sigma_M5_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M5_160_60 = sigma_M5_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_M6_4_60 = sigma_M6_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M6_10_60 = sigma_M6_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M6_55_60 = sigma_M6_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M6_160_60 = sigma_M6_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_trans_4_60 = sigma_trans_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_10_60 = sigma_trans_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_55_60 = sigma_trans_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_160_60 = sigma_trans_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

"""fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((6,4.5))

#print tNaf_outer, sigma_io_outer


cmap = plt.get_cmap('hsv')
ndata = 4
colors = np.linspace(0,0.85,ndata)

plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M3_4_60, sigma_M3_10_60, sigma_M3_55_60, sigma_M3_160_60],'D',color = cmap(colors[0]),markersize=ms,markeredgecolor='none')
plt.plot(t_Karan[[0,1,3,4]],sigma_60_Karan[:-1],'D',color = cmap(colors[1]),markersize=ms,markeredgecolor='none')

plt.ylim(0,10)


font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=16)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(15)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(16)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)



plt.xlabel(r'$\mathdefault{t_{Naf} \,[nm]}$',fontname='Arial',fontsize=ff,labelpad=10)
plt.ylabel(r'$\mathdefault{\sigma_{\rm io}\,\left[\frac{\rm S}{\rmcm}\right]}$',fontname='Arial',fontsize=ff,labelpad=0.05)

plt.savefig('Fig7_sigma_v_data_60C.pdf',dpi=500,format='pdf')"""


fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M3_4_60, sigma_M3_10_60, sigma_M3_55_60, sigma_M3_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M3_4_50, sigma_M3_10_50, sigma_M3_55_50, sigma_M3_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M3_4_40, sigma_M3_10_40, sigma_M3_55_40, sigma_M3_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M3[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)



#xshift = 5
#yshift = 0.2
#plt.plot(11.2+xshift,7+yshift+0.15,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
#plt.plot(29.2+xshift,7+yshift+0.15,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=16)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('d) Model 4',xy=[38,7],family='Arial',size=12)
#ax.annotate('Model 3',xy=[20,7],family='Arial',size=18)
#ax.annotate(' = Measured',xy=[12+xshift,7+yshift],family='Arial',size=ff-5)
#ax.annotate(' = Predicted from NR',xy=[30+xshift,7+yshift],family='Arial',size=ff-5)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)


#plt.savefig('Fig7d_sigma_v_data_Summary_M4.pdf',dpi=500,format='pdf')



"Model 3"

fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M4_4_60, sigma_M4_10_60, sigma_M4_55_60, sigma_M4_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M4_4_50, sigma_M4_10_50, sigma_M4_55_50, sigma_M4_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M4_4_40, sigma_M4_10_40, sigma_M4_55_40, sigma_M4_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M4[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)


#xshift = 5
#yshift = 0.2
#plt.plot(11.2+xshift,7+yshift+0.15,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
#plt.plot(29.2+xshift,7+yshift+0.15,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=16)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('c) Model 3',xy=[38,7],family='Arial',size=12)
#ax.annotate(' = Measured',xy=[12+xshift,7+yshift],family='Arial',size=ff-5)
#ax.annotate(' = Predicted from NR',xy=[30+xshift,7+yshift],family='Arial',size=ff-5)


plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

#plt.legend([T30,T40,T50,T60],[r'$\mathdefault{30^\circ C}$',r'$\mathdefault{40^\circ C}$',r'$\mathdefault{50^\circ C}$',r'$\mathdefault{60^\circ C}$'],prop=font,handletextpad=0.05,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.savefig('Fig7c_sigma_v_data_Summary_M3.pdf',dpi=500,format='pdf')





"Model 2"

fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M2_4_60, sigma_M2_10_60, sigma_M2_55_60, sigma_M2_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M2_4_50, sigma_M2_10_50, sigma_M2_55_50, sigma_M2_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M2_4_40, sigma_M2_10_40, sigma_M2_55_40, sigma_M2_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M2[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)


#xshift = 5
#yshift = 0.2
#plt.plot(11.2+xshift,7+yshift+0.15,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
#plt.plot(29.2+xshift,7+yshift+0.15,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=10)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('b) Model 2',xy=[38,7],family='Arial',size=12)
#ax.annotate(' = Measured',xy=[12+xshift,7+yshift],family='Arial',size=ff-5)
#ax.annotate(' = Predicted from NR',xy=[30+xshift,7+yshift],family='Arial',size=ff-5)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

plt.legend([T30,T40,T50,T60],['30$^\circ$C','40$^\circ$C','50$^\circ$C','60$^\circ$C'],prop=font,handletextpad=-0.5,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.legend([T30,T40,T50,T60],[r'$\mathdefault{30^\circ C}$',r'$\mathdefault{40^\circ C}$',r'$\mathdefault{50^\circ C}$',r'$\mathdefault{60^\circ C}$'],prop=font,handletextpad=0.05,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.savefig('Fig7b_sigma_v_data_Summary_M2.pdf',dpi=500,format='pdf')



"Model 1"

fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3.0,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M1_4_60, sigma_M1_10_60, sigma_M1_55_60, sigma_M1_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M1_4_50, sigma_M1_10_50, sigma_M1_55_50, sigma_M1_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M1_4_40, sigma_M1_10_40, sigma_M1_55_40, sigma_M1_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M1[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)



xshift = -9.4
yshift = .5
plt.plot(11.2+xshift,0+yshift+0.13,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
plt.plot(34.2+xshift,0+yshift+0.13,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=10)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('a) Model 1',xy=[38,7],family='Arial',size=12)
ax.annotate(' = Measured',xy=[12+xshift,-0.1+yshift],family='Arial',size=10)
ax.annotate(' = Predicted from NR',xy=[35+xshift,-0.1+yshift],family='Arial',size=10)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

#plt.legend([T30,T40,T50,T60],[r'$\mathdefault{30^\circ C}$',r'$\mathdefault{40^\circ C}$',r'$\mathdefault{50^\circ C}$',r'$\mathdefault{60^\circ C}$'],prop=font,handletextpad=0.05,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.savefig('Fig7a_sigma_v_data_Summary_M1.pdf',dpi=500,format='pdf')



fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M5_4_60, sigma_M5_10_60, sigma_M5_55_60, sigma_M5_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M5_4_50, sigma_M5_10_50, sigma_M5_55_50, sigma_M5_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M5_4_40, sigma_M5_10_40, sigma_M5_55_40, sigma_M5_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M5[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)



xshift = -1.4
yshift = .5
plt.plot(4.5+xshift,0.05+yshift+0.15,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
plt.plot(27.2+xshift,0.05+yshift+0.15,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=10)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('a) Model 5',xy=[38,7],family='Arial',size=12)
ax.annotate(' = Measured',xy=[5+xshift,0+yshift],family='Arial',size=ff)
ax.annotate(' = Predicted from NR',xy=[28+xshift,0+yshift],family='Arial',size=ff)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

#plt.legend([T30,T40,T50,T60],[r'$\mathdefault{30^\circ C}$',r'$\mathdefault{40^\circ C}$',r'$\mathdefault{50^\circ C}$',r'$\mathdefault{60^\circ C}$'],prop=font,handletextpad=0.05,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.savefig('FigS3a_sigma_v_data_Summary_M5.pdf',dpi=500,format='pdf')



fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.8,0.8])
fig.set_size_inches((3,2.25))


cmap = plt.get_cmap('gnuplot2')
ndata = 6
colors = np.linspace(0,0.99,ndata)

mw = 1.0
lw = 1.0

colors = ['k','b','r','#FFA500']

plt.plot(t_Karan[[0,1,3]],sigma_30_Karan[:-2],'-',marker='x',markeredgecolor = colors[0],markersize=ms,zorder=10,color=colors[0],markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_40_Karan[:-2],'-',marker='x',markeredgecolor = colors[1],color = colors[1],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_50_Karan[:-2],'-',marker='x',markeredgecolor = colors[2],color=colors[2],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)
plt.plot(t_Karan[[0,1,3]],sigma_60_Karan[:-2],'-',marker='x',markeredgecolor = colors[3],color=colors[3],markersize=ms,zorder=10,markeredgewidth=mw,markerfacecolor='none',linewidth=lw)

T60, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M6_4_60, sigma_M6_10_60, sigma_M6_55_60, sigma_M6_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M6_4_50, sigma_M6_10_50, sigma_M6_55_50, sigma_M6_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film[[1,2,6,9]]/10.0,[sigma_M6_4_40, sigma_M6_10_40, sigma_M6_55_40, sigma_M6_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film[[1,2,6,9]]/10.0,sigma_io_par_eff_M6[[1,2,6,9]],'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)


#xshift = 5
#yshift = 0.2
#plt.plot(11.2+xshift,7+yshift+0.15,'x',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='none')
#plt.plot(29.2+xshift,7+yshift+0.15,'o',markeredgecolor='k',markersize=ms,markeredgewidth=mw,markerfacecolor='k')


plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=10)

###plt.legend([r'$\mathdefault{\sigma_{\rm parallel}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm normal}^{\it eff}}$', r'$\mathdefault{\sigma_{\rm outer}}$',r'$\mathdefault{\sigma_{\rm average}}$',r'$\mathdefault{\sigma_{\rm bulk}}$'],prop=font, handletextpad=0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()
#majorFormatter = plt.FormatStrFormatter('%0.0f')
#ax.xaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

#plt.xticks([0,40,80,120])
#plt.yticks(yt)
ax.annotate('b) Model 6',xy=[38,7],family='Arial',size=12)
#ax.annotate(' = Measured',xy=[12+xshift,7+yshift],family='Arial',size=ff-5)
#ax.annotate(' = Predicted from NR',xy=[30+xshift,7+yshift],family='Arial',size=ff-5)

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

plt.legend([T30,T40,T50,T60],['30$^\circ$C','40$^\circ$C','50$^\circ$C','60$^\circ$C'],prop=font,handletextpad=-0.5,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)


#plt.legend([T30,T40,T50,T60],[r'$\mathdefault{30^\circ C}$',r'$\mathdefault{40^\circ C}$',r'$\mathdefault{50^\circ C}$',r'$\mathdefault{60^\circ C}$'],prop=font,handletextpad=0.05,numpoints=1,loc=8, ncol=4,mode="expand",frameon=False)

#plt.savefig('FigS3b_sigma_v_data_Summary_M6.pdf',dpi=500,format='pdf')

#plt.show()

