import numpy as np
import pylab as plt
import copy
import math

from scipy.optimize import minimize
plt.close('all')

C3_1 = 0.333
C3_2 = 0.754
C3_3 = 0.671

C0 = np.array([C3_1, C3_2, C3_3])

def RSS_func(C,flag=0.0):

    Clam1_M3 = C[0]
    Clam2_M3 = C[1]
    Cfilm_M3 = C[2]

    T = 302.75
    
    sigma_30_Karan = np.array([21.481, 22.062, 20.185, 34.810, 43.913])/10.

    SLDs = np.array([[-0.2237, 4.5043, 1.6979, 3.9113, 4.15743, 4.15743, 4.15743], #t5
                    [-0.0958, 4.373, 2.583, 3.791, 2.6508, 4.15743, 4.15743], #t7
                    [0.8616, 4.3067, 2.8787, 3.7708, 3.2241, 3.5187, 3.3272], #t42
                    [-0.3722, 3.5109, 1.4651, 3.2542, 2.5756, 2.9446, 3.1720]]) #t120

    t_i = np.array([[8.612, 24.118, 10.505, 15.281, 0, 0, 0], #t5
                    [7.5947, 20.907, 15.228, 18.3322, 26.467, 0, 0], #t7
                    [10.866, 16.236, 16.600, 16.202, 21.114, 11.830, 417.886], #t42
                    [9.481, 22.070, 16.776, 19.297, 18.831, 1477.106, 56.649]]) #t120

    t_i_norm = copy.deepcopy(t_i)
    t_i_norm[t_i_norm==0] = 1.0

    V_w = (4.15743 - SLDs)/(4.15743 + 0.559883)
    Lambda_i = 1.006*1100*V_w/(1-V_w)/18.016/1.98
    Lambda_calc_low = np.array([34.6, 7.1])

    SLD_high = np.array([0.32, 3.78])
    V_w_high = (4.15743 - SLD_high)/(4.15743 + 0.559883)
    Lambda_calc_high = 1.006*1100*V_w_high/(1-V_w_high)/18.016/1.98

    Lambda_i[0:3,0:2] = Lambda_calc_low[0:2]
    Lambda_i[3:,0:2] = Lambda_calc_high[0:2]

    sigma_io_i = math.exp(1286.*(1./303. - 1./T))*(0.514*Lambda_i - 0.326)
    sigma_io_i_M3 = copy.deepcopy(sigma_io_i)

    sigma_io_i_M3 *= Cfilm_M3
    sigma_io_i_M3[:,0] *= ((1.0 - 0./4.)*Clam1_M3 + (0./4.)*Clam2_M3)
    sigma_io_i_M3[:,1] *= ((1.0 - 1./4.)*Clam1_M3 + (1./4.)*Clam2_M3)
    sigma_io_i_M3[:,2] *= ((1.0 - 2./4.)*Clam1_M3 + (2./4.)*Clam2_M3)
    sigma_io_i_M3[:,3] *= ((1.0 - 3./4.)*Clam1_M3 + (3./4.)*Clam2_M3)
    sigma_io_i_M3[:,4] *= ((1.0 - 4./4.)*Clam1_M3 + (4./4.)*Clam2_M3)

    R_io_trans_eff_M3 = np.zeros(t_i.shape[0])
    sigma_io_trans_eff_M3 = np.zeros(t_i.shape[0])
    sigma_io_par_eff_M3 = np.zeros(t_i.shape[0])
    t_film = np.zeros(t_i.shape[0])

    for j in np.arange(t_i.shape[0]):
        t_film[j] = sum(t_i[j,:])
        for i in np.arange(t_i.shape[1]):
            sigma_io_par_eff_M3[j] += sigma_io_i_M3[j,i]*t_i[j,i]/t_film[j]
            R_io_trans_eff_M3[j] += t_i[j,i]/sigma_io_i_M3[j,i]/t_film[j]

    sigma_io_trans_eff_M3 = 1/R_io_trans_eff_M3
    RSS_3 = (sigma_io_par_eff_M3[0] - sigma_30_Karan[0])**2 \
          + (sigma_io_par_eff_M3[1] - sigma_30_Karan[1])**2 \
          + (sigma_io_par_eff_M3[2] - sigma_30_Karan[2])**2 \
          + (sigma_io_par_eff_M3[3] - sigma_30_Karan[3])**2

    if flag==0:
        return RSS_3
    elif flag == 1:
        return t_film
    elif flag ==2:
        return sigma_io_par_eff_M3
    elif flag == 3:
        return sigma_io_trans_eff_M3

bounds = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]])
res = minimize(RSS_func, C0,method='L-BFGS-B',bounds=bounds,options={'disp':True,'xtol':1e-3},args=(0.0,))
print( res.x )

ms = 3.5
UL = 4.0
ff = 10
yt = [0,15,30, 45]
cs = 4.5
lw = 1.5

fig = plt.figure()
ax = fig.add_axes([0.175,0.175,0.775,0.75])
fig.set_size_inches((3,2.25))

t_film = RSS_func(res.x,1)
sigma_io_par_eff_M3 = RSS_func(res.x,2)

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
plt.plot(t_film/10.0,sigma_io_par_eff_M3,'o',color = 'k',markersize=ms,markeredgecolor='none',zorder=4)
plt.plot(t_Karan[[0,1,3,4]],sigma_30_Karan[:-1],'-',zorder=10,color = 'k',linewidth=1.0)

plt.ylim(0,UL)
plt.xlim(0,170)
plt.xticks([0,40,80,120,160])
plt.yticks([0.0,1.0,2.0,3.0,4.0])

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=8)
plt.legend(['Measured','Model 4, SSR = 0.001'],prop=font,handletextpad=-0.1,numpoints=1,loc=4, frameon=False)

ax = plt.gca()

majorFormatter = plt.FormatStrFormatter('%0.1f')
ax.yaxis.set_major_formatter( majorFormatter)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=3.0)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=3.0)

" 50 C "
"Activation energies from Paul, et al supporting info [J/mol]:"
Ea_4a = 3269.3 #25.45*1000
Ea_10a = 7528.8
Ea_55a = 6589.9
Ea_160a = 6395.8

Ea_4b = 3110.6
Ea_10b = 1284.2
Ea_55b = 2322.7
Ea_160b = 1613.6

T1 = 30
T2 = 40

sigma_M3_4_40 = sigma_io_par_eff_M3[0]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_10_40 = sigma_io_par_eff_M3[1]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_55_40 = sigma_io_par_eff_M3[2]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_M3_160_40 = sigma_io_par_eff_M3[3]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

sigma_io_trans_eff_M3 = RSS_func(res.x,3)

sigma_trans_4_40 = sigma_io_trans_eff_M3[0]*math.exp(Ea_4a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_10_40 = sigma_io_trans_eff_M3[1]*math.exp(Ea_10a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_55_40 = sigma_io_trans_eff_M3[2]*math.exp(Ea_55a*(1/(273.15+T1) - 1/(273.15+T2)))
sigma_trans_160_40 = sigma_io_trans_eff_M3[3]*math.exp(Ea_160a*(1/(273.15+T1) - 1/(273.15+T2)))

T3 = 50

sigma_M3_4_50 = sigma_M3_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_10_50 = sigma_M3_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_55_50 = sigma_M3_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_M3_160_50 = sigma_M3_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

sigma_trans_4_50 = sigma_trans_4_40*math.exp(Ea_4b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_10_50 = sigma_trans_10_40*math.exp(Ea_10b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_55_50 = sigma_trans_55_40*math.exp(Ea_55b*(1/(273.15+T2) - 1/(273.15+T3)))
sigma_trans_160_50 = sigma_trans_160_40*math.exp(Ea_160b*(1/(273.15+T2) - 1/(273.15+T3)))

T4 = 60

sigma_M3_4_60 = sigma_M3_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_10_60 = sigma_M3_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_55_60 = sigma_M3_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_M3_160_60 = sigma_M3_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

sigma_trans_4_60 = sigma_trans_4_50*math.exp(Ea_4b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_10_60 = sigma_trans_10_50*math.exp(Ea_10b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_55_60 = sigma_trans_55_50*math.exp(Ea_55b*(1/(273.15+T3) - 1/(273.15+T4)))
sigma_trans_160_60 = sigma_trans_160_50*math.exp(Ea_160b*(1/(273.15+T3) - 1/(273.15+T4)))

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

T60, = plt.plot(t_film/10.0,[sigma_M3_4_60, sigma_M3_10_60, sigma_M3_55_60, sigma_M3_160_60],'o',color = (colors[3]),markersize=ms,markeredgecolor='none',zorder=1)
T50, = plt.plot(t_film/10.0,[sigma_M3_4_50, sigma_M3_10_50, sigma_M3_55_50, sigma_M3_160_50],'o',color = (colors[2]),markersize=ms,markeredgecolor='none',zorder=2)
T40, = plt.plot(t_film/10.0,[sigma_M3_4_40, sigma_M3_10_40, sigma_M3_55_40, sigma_M3_160_40],'o',color = (colors[1]),markersize=ms,markeredgecolor='none',zorder=3)
T30, = plt.plot(t_film/10.0,sigma_io_par_eff_M3,'o',color = (colors[0]),markersize=ms,markeredgecolor='none',zorder=3)

plt.ylim(0,8)
plt.xlim(0,60)

font = plt.matplotlib.font_manager.FontProperties(family='Arial',size=16)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
    tick.label1.set_fontname('Arial')

ax.annotate('d) Model 4',xy=[38,7],family='Arial',size=12)
plt.xlabel('Thickness (nm)',fontname='Arial',fontsize=ff,labelpad=0.75)
plt.ylabel(r'$\mathdefault{{\it \sigma}_{io}\,\left(\frac{S}{m}\right)}$',fontname='Arial',fontsize=ff,labelpad=0.75)

