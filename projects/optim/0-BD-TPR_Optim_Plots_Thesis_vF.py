# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:51:21 2021

@author: z5158936
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.optimize as spo
import scipy.interpolate as spi

def f_minPoly(X,*args):
    Xs,Ys = args
    # f_inter = spi.interp1d(Xs, Ys, kind='cubic',fill_value='extrapolate')
    f_inter = spi.interp1d(Xs, Ys, kind='quadratic',fill_value='extrapolate')
    return f_inter(X)


file_rslts = 'Optim_TPR_all.csv'
fldr_rslts = 'Optim_Results/'

###########################################
#%%%  DIFFERENT HEIGHT, FIX RADIATION FLUX
###########################################
#FIGURE 1

Q_av = 1.00
Q_avs = np.arange(0.50,1.51,0.5)

ms = ['o','s','d']

i=0
for Q_av in Q_avs:
    
    file_rslts = 'Optim_TPR_0D_quick.csv'
    df = pd.read_csv(file_rslts,index_col=0)
    df = df.round({'Q_av':2})
    pd.set_option('display.max_columns', None)
    df = df[(df.Q_avg_i==Q_av)].copy()
    df.sort_values(['zf','Prcv'],inplace=True)
    
    zfs = df.zf.unique()
    
    ###########################################
    #FIGURE 3
    fig, ax1 = plt.subplots(figsize=(9,6))
    mins = []
    fs = 18
    for zf in zfs:
        df2 = df[(df.zf==zf)].copy()
        
        df2.drop_duplicates(subset=['Prcv','zf','Q_avg_i'],inplace=True)
        ax1.plot(df2.Prcv,df2.LCOH,lw=1.5,label=str(zf)+' m')
    
        bounds = (df2.Prcv.min(),df2.Prcv.max())
        args = (df2.Prcv,df2.LCOH)
        res = spo.minimize_scalar(f_minPoly, bounds=bounds, args=args, method='bounded')
        Prcv_min = res.x
        LCOH_min = f_minPoly(Prcv_min,*args)
        Nhel_min = spi.interp1d(df2.Prcv, df2.N_hel, kind='cubic',fill_value='extrapolate')(Prcv_min)
        etaSF_min = spi.interp1d(df2.Prcv, df2.eta_SF, kind='cubic',fill_value='extrapolate')(Prcv_min)
        fzv_min  = spi.interp1d(df2.Prcv, df2.fzv, kind='cubic',fill_value='extrapolate')(Prcv_min)
        mins.append([zf,Prcv_min,LCOH_min,Nhel_min,etaSF_min,fzv_min])
    
    mins = pd.DataFrame(mins,columns=('zf','Prcv','LCOH','Nhel','eta_SF','fzv'))
    mins.sort_values(by='zf',inplace=True)
    
    args = (mins.Prcv,mins.LCOH)
    Prcvs = np.arange(mins.Prcv.min(),mins.Prcv.max(),0.1)
    ax1.plot(mins.Prcv,mins.LCOH,c='mediumblue',lw=3,marker='s',markersize=10,label='min')
    ax1.set_ylim(20,35)
    ax1.set_xlim(0,40)
    # ax1.set_title('LCOH for different receiver powers and tower heights with $Q_{{avg}}={:.2f}$'.format(Q_av),fontsize=fs)
    ax1.set_ylabel(r'LCOH $(USD/MW_t)$',fontsize=fs)
    ax1.set_xlabel('Receiver Power $(MW_t)$',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=1,bbox_to_anchor=(1.25, 0.98),fontsize=fs-2)
    ax1.grid()
    fig.savefig(fldr_rslts+'Prcv_zf_LCOH_Qavg_{:.2f}_fig3.png'.format(Q_av), bbox_inches='tight')
    plt.show()
    
    ###########################################
    # TOWER HEIGHT VS RECEIVER POWER
    
    def func1(x, *params):
        A,b = params
        return A*np.exp(b*x)
    
    X = mins.zf
    Y = mins.Prcv
    p0 = (1., 0.05)
    coefs, covariance = spo.curve_fit( func1, X, Y, maxfev=10000, p0=p0)
    Yc = func1(X,*coefs)
    r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    Xc = np.linspace(20,75,100)
    Yc = func1(Xc,*coefs)
    A,b = coefs
    
    
    def func3(x, *params):
        A,b,c = params
        # return c+A/x**b
        return c+A*np.exp(-x*b)
    X3 = mins.zf
    Y3 = mins.LCOH
    p0 = (10., 0.05, 20.)
    coefs_LCOH, covariance = spo.curve_fit( func3, X3, Y3, maxfev=10000, p0=p0)
    Yc3 = func3(X3,*coefs_LCOH)
    r2 = 1 - (np.sum((Y3 - Yc3)**2) / np.sum((Y3-np.mean(Y3))**2))
    A3,b3,c3 = coefs_LCOH
    print(A3,b3,c3,r2)
    
    figb, ax1b = plt.subplots(figsize=(9,6))
    ax2b = ax1b.twinx()
    fs=18
    cLCOH = 'mediumblue'
    cPrcv = 'orangered'
    # print(mins)
    # ax1b.plot(mins.zf,mins.LCOH,lw=3,c='mediumblue',marker=ms[i],markersize=10,label='{:.2f}'.format(Q_av))
    ax1b.scatter(mins.zf, mins.LCOH, c=cLCOH, marker=ms[i], s=200, label='LCOH')
    ax2b.scatter(mins.zf,mins.Prcv,c=cPrcv,marker=ms[i],s=200,label=r'$P_{rcv}$')
    
    ax2b.plot(Xc,Yc,c=cPrcv,lw=2,ls=':')
    ax1b.plot(X3,Yc3,lw=3, c=cLCOH, ls=':')
    
    ax2b.annotate(r'$P_{{rcv}}={:.1f}e^{{{:.3f}z_f}}$'.format(A,b),(Xc[-1]-18,Yc[-1]),c=cPrcv,fontsize=fs)
    
    ax1b.annotate(r'${:.1f}+{:.1f}e^{{-{:.2f}z_f}}$'.format(c3,A3,b3),(X3.iloc[-1]-20,Yc3.iloc[-1]+1),c=cLCOH,fontsize=fs-2)
    
    ax1b.scatter([],[],lw=3,c=cPrcv,marker='s',s=200,label=r'$P_{rcv}$')
    
    # ax1b.plot([],[],c='C1',lw=2,ls=':',label=r'$P_{{rcv}}={:.1f}e^{{{:.3f}z_f}}$'.format(A,b))
    
    ax1b.legend(loc=2,fontsize=fs)
    ax1b.set_ylim(20,35)
    ax1b.grid()
    # ax1b.set_title('LCOH and optimal receiver power for different tower heights with $Q_{{avg}}={:.2f}$'.format(Q_av),fontsize=fs)
    ax1b.set_xlabel(r'Tower height $(m)$',fontsize=fs)
    ax1b.set_ylabel(r'LCOH $(USD/MW_t)$',fontsize=fs)
    ax2b.set_ylabel('Receiver Power $(MW_t)$',fontsize=fs)
    ax2b.spines['left'].set_color('mediumblue')
    ax2b.spines['right'].set_color('C1')
    ax1b.tick_params(axis='y', colors=cLCOH,size=10)
    ax2b.tick_params(axis='y', colors=cPrcv,size=10)
    ax1b.yaxis.label.set_color(cLCOH)
    ax2b.yaxis.label.set_color(cPrcv)
    ax1b.tick_params(axis='both', which='major', labelsize=fs-2)
    ax2b.tick_params(axis='both', which='major', labelsize=fs-2)
    ax2b.set_yticks(np.linspace(ax2b.get_yticks()[0], ax2b.get_yticks()[-1], len(ax1b.get_yticks())))
    ax2b.set_yticks(np.linspace(0, 35, len(ax1b.get_yticks())))
    plt.show()
    
    i+=1
    figb.savefig(fldr_rslts+'zf_Prcv_min_Qavg_{:.2f}.png'.format(Q_av), bbox_inches='tight')
    
    
# sys.exit()
    
#%%% INFLUENCE OF RADIATION FLUX

file_rslts = 'Optim_TPR_0D_quick.csv'
df = pd.read_csv(file_rslts,index_col=0)
df = df.round({'Q_av':2})
zfs = np.arange(20,76,5)
# zfs = [30,40,50,60]
mins = []

for zf in zfs:
    
    df = pd.read_csv(file_rslts,index_col=0)
    df = df.round({'Q_av':2})
    pd.set_option('display.max_columns', None)
    df = df[(df.zf==zf)].copy()
    df.sort_values(['Q_av','Prcv'],inplace=True)
    
    idx_min   = df.LCOH.idxmin()
    Prcv_min  = df.loc[idx_min]['Prcv']
    LCOH_min  = df.loc[idx_min]['LCOH']
    Nhel_min  = df.loc[idx_min]['N_hel']
    etaSF_min = df.loc[idx_min]['eta_SF']
    fzv_min   = df.loc[idx_min]['fzv']
    Q_av      = df.loc[idx_min]['Q_avg_i']
    Arcv      = df.loc[idx_min]['Arcv']
    eta_rcv   = df.loc[idx_min]['eta_rcv']
    S_HB      = df.loc[idx_min]['S_HB']
    S_TOD     = df.loc[idx_min]['S_TOD']
    S_land    = Prcv_min * 1e4 / df.loc[idx_min]['land_prod']
    mins.append([idx_min,zf,Prcv_min,LCOH_min,Nhel_min,etaSF_min,fzv_min,Q_av,Arcv, eta_rcv,S_TOD,S_HB,S_land])
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    Qavgs = np.arange(0.5,2.1,0.25)
    for Q_av in Qavgs:
        df2 = df[(df.Q_avg_i==Q_av)]
        df2.drop_duplicates(subset=['Prcv','zf','fzv','Q_avg_i'],inplace=True)
        # df2.drop_duplicates(inplace=True)
        ax1.plot(df2.Prcv,df2.LCOH,lw=3, label=r'${:.2f} MW/m^2$'.format(Q_av))
    
    ax1.scatter([Prcv_min],[LCOH_min],lw=3,c='red',marker='*',s=200,label='Design')
    y1,y2 = 20,30
    ax1.plot([Prcv_min,Prcv_min],[y1,y2],lw=2,c='red',ls=':')
    # ax1.set_title(r'Min LCOH for tower $z_{{f}}={:.1f}$'.format(zf),fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.set_xlim(0,40)
    ax1.set_ylim(y1,y2)
    # ax1.plot(Prcvs,LCOHs,lw=3,c='mediumblue',ls='--',label='min(LCOH)')
    ax1.set_ylabel(r'LCOH $(USD/MW_t)$',fontsize=fs)
    ax1.set_xlabel('Receiver Power $(MW_t)$',fontsize=fs)
    ax1.legend(loc=1,bbox_to_anchor=(1.35, 1.00),fontsize=fs-2)
    ax1.grid()
    plt.show()
    fig.savefig(fldr_rslts+'LCOH_Prcv_Qavg_zf_{:.0f}m.png'.format(zf), bbox_inches='tight')

mins = pd.DataFrame(mins,columns=('idx_min', 'zf', 'Prcv', 'LCOH', 'Nhel', 'eta_SF', 'fzv', 'Q_av', 'Arcv', 'eta_rcv', 'S_TOD', 'S_HB', 'S_land'))

# mins.loc[7,'Prcv'] = 21
# mins.loc[8,'Prcv'] = 23

def func1_2var(X, *params):
    zf, Q_av = X['zf'],X['Q_av']
    A,b = params
    return A*Q_av*np.exp(b*zf)

X = mins[['zf','Q_av']]
Y = mins.Prcv
p0 = (1., 0.05)
coefs, covariance = spo.curve_fit( func1_2var, X, Y, maxfev=10000, p0=p0)
Yc = func1_2var(X,*coefs)
r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
A,b = coefs
print(A,b,r2)


def func2(x, *params):
    A,b = params
    return A*np.exp(b*x)

X2 = mins.zf
Y2 = mins.Prcv
p0 = (1., 0.05)
coefs, covariance = spo.curve_fit( func2, X2, Y2, maxfev=10000, p0=p0)
Yc2 = func2(X2,*coefs)
r2 = 1 - (np.sum((Y2 - Yc2)**2) / np.sum((Y2-np.mean(Y2))**2))
A2,b2 = coefs
print(A2,b2,r2)


def func3(x, *params):
    A,b,c = params
    # return c+A/x**b
    return c+A*np.exp(-x*b)

X3 = mins.zf
Y3 = mins.LCOH
p0 = (10., 0.05, 20.)
coefs_LCOH, covariance = spo.curve_fit( func3, X3, Y3, maxfev=10000, p0=p0)
Yc3 = func3(X3,*coefs_LCOH)
r2 = 1 - (np.sum((Y3 - Yc3)**2) / np.sum((Y3-np.mean(Y3))**2))
A3,b3,c3 = coefs_LCOH
print(A3,b3,c3,r2)

Qavs = mins.Q_av
msizes = 100*mins.Q_av**2
smin = 30
smax = 300
Qmin = 0.5
Qmax = 1.25
msizes = [smin+(smax-smin)*(aux-Qmin)/(Qmax-Qmin) for aux in Qavs]

fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twinx()
# ax1.plot(mins.zf,mins.LCOH,lw=3, c='mediumblue',marker='o',markersize=15,label='LCOH')
ax1.scatter(mins.zf, mins.LCOH,lw=3, c=cLCOH, marker='o', s=150, label='LCOH')
ax1.plot([],[],lw=3, c=cPrcv, marker='s',markersize=15,label='Receiver Power')
ax2.scatter(mins.zf,mins.Prcv, c=cPrcv, marker='s', s=150)

ax2.plot(X2,Yc2,lw=3, c=cPrcv, ls=':')
ax1.plot(X3,Yc3,lw=3, c=cLCOH, ls=':')

ax2.annotate(r'$P_{{rcv}}={:.1f}e^{{{:.3f}z_f}}$'.format(A2,b2),(Xc[-1]-20,Yc.iloc[-1]-2),c=cPrcv,fontsize=fs)
ax1.annotate(r'${:.1f}+{:.1f}e^{{-{:.3f}z_f}}$'.format(c3,A3,b3),(X3.iloc[-1]-20,Yc3.iloc[-1]+1),c=cLCOH,fontsize=fs-2)

# ax2.scatter([],[],s=msizes[0], c=cPrcv,marker='s', label=r'$0.75 MW/m^2$')
# ax2.scatter([],[],s=msizes[1], c=cPrcv,marker='s', label=r'$1.00 MW/m^2$')
# ax2.scatter([],[],s=msizes[3], c=cPrcv,marker='s', label=r'$1.25 MW/m^2$')
# ax2.scatter([],[],s=msizes[11], c=cPrcv, marker='s', label=r'$1.50 MW/m^2$')

ax1.set_xlabel('Tower height $(m)$',fontsize=fs)
ax1.set_ylabel(r'Minimum LCOH $(USD/MW_{th})$',fontsize=fs)
ax2.set_ylabel(r'Optimal Receiver Power $(MW_{th})$',fontsize=fs)

ax2.spines['left'].set_color(cLCOH)
ax2.spines['right'].set_color(cPrcv)
ax1.tick_params(axis='y', colors=cLCOH,size=10)
ax2.tick_params(axis='y', colors=cPrcv,size=10)
ax1.yaxis.label.set_color(cLCOH)
ax2.yaxis.label.set_color(cPrcv)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)

# ax1.set_title(r'Min LCOH for towers heights and average radiation flux. FINAL!',fontsize=fs)
# ax1.set_xlim(25,65)
ax1.set_ylim(y1,y2)
# ax2.set_ylim(0,30)
ax1.legend(bbox_to_anchor=(0.0,-0.25), loc="lower left",fontsize=fs-2,ncol=2)
# ax2.legend(bbox_to_anchor=(0.4,-0.35), loc="lower left",fontsize=fs-4,ncol=2)
ax1.grid()
fig.savefig(fldr_rslts+'FINAL_LCOH_zf_Prcv.png', bbox_inches='tight')
plt.show()


# Function for Q_av
def func4(x, *params):
    A,b,c = params
    # return c+A/x**b
    return c - A*np.exp(-x*b)
X4 = mins.zf
Y4 = mins.Q_av
p0 = (1., 0.05, 1.)
coefs, covariance = spo.curve_fit( func4, X4, Y4, maxfev=10000, p0=p0)
Yc4 = func4(X4,*coefs)
r2 = 1 - (np.sum((Y4 - Yc4)**2) / np.sum((Y4-np.mean(Y4))**2))
A4,b4,c4 = coefs
print(A4,b4,c4,r2)

cQavg = 'darkgreen'
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(mins.zf,mins.Q_av,s=100,c=cQavg)
ax1.plot(X4,Yc4,lw=3, c=cQavg, ls=':',label=r'${:.2f}-{:.2f}e^{{-{:.3f}z_f}}$'.format(c4,A4,b4))
ax1.legend(loc=0,fontsize=fs-2)

ax1.set_xlim(18,77)
ax1.set_xlabel('Tower height $(m)$',fontsize=fs)
ax1.set_ylabel(r'Optimal $Q_{avg} (MW_{th}/m^2)$',fontsize=fs)
ax1.spines['left'].set_color(cQavg)
ax1.tick_params(axis='y', colors=cQavg,size=10)
ax1.yaxis.label.set_color(cQavg)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslts+'FINAL_LCOH_zf_Qavg.png', bbox_inches='tight')
plt.show()

# Function for fzv
def func5(x, *params):
    A,b,c = params
    # return c+A/x**b
    return c+A*np.exp(-x*b)
X5 = mins.zf
Y5 = mins.fzv
p0 = (10., 0.05, 20.)
coefs, covariance = spo.curve_fit( func5, X5, Y5, maxfev=10000, p0=p0)
Yc5 = func5(X5,*coefs)
r2 = 1 - (np.sum((Y5 - Yc5)**2) / np.sum((Y5-np.mean(Y5))**2))
A5,b5,c5 = coefs
print(A5,b5,c5,r2)

cfzv = 'darkviolet'
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(mins.zf,mins.fzv,s=100,c=cfzv)
ax1.plot(X5,Yc5,lw=3, c=cfzv, ls=':',label=r'${:.2f}+{:.2f}e^{{-{:.3f}z_f}}$'.format(c5,A5,b5))
ax1.legend(loc=0,fontsize=fs-2)

ax1.set_xlim(18,77)
ax1.set_xlabel('Tower height $(m)$',fontsize=fs)
ax1.set_ylabel(r'Optimal $f_{zv} (-)$',fontsize=fs)
ax1.spines['left'].set_color(cfzv)
ax1.tick_params(axis='y', colors=cfzv,size=10)
ax1.yaxis.label.set_color(cfzv)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslts+'FINAL_LCOH_zf_fzv.png', bbox_inches='tight')
plt.show()

zf = 50
Prcv = func2(zf, *[A2,b2])
LCOH = func3(zf, *[A3,b3,c3])
Qavg = func4(zf, *[A4,b4,c4])
fzv = func5(zf, *[A5,b5,c5])
print(zf,Prcv,LCOH,Qavg,fzv)

### ADDITIONAL CORRELATIONS
def func6(x, *params):
    A,b = params
    return A*np.exp(b*x)

ylabs = {'Nhel':'$N_{hel} (-)$','S_TOD':'$S_{TOD} (m^2)$','S_HB':'$S_{HB}  (m^2)$','Arcv':'$A_{rcv} (m^2)$','S_land':'$S_{land} (m^2)$'}

for vary in ['Nhel','S_TOD','S_HB','Arcv','S_land']:
    X6 = mins.zf
    Y6 = mins[vary]
    p0 = (1., 0.05)
    coefs, covariance = spo.curve_fit( func6, X6, Y6, maxfev=10000, p0=p0)
    Yc6 = func2(X6,*coefs)
    r2 = 1 - (np.sum((Y6 - Yc6)**2) / np.sum((Y6-np.mean(Y6))**2))
    A6,b6 = coefs
    
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.scatter(mins.zf,mins[vary],s=100,c=cLCOH)
    ax1.plot(X6,Yc6,lw=3, c=cLCOH, ls=':',label=r'${:.2f}e^{{{:.3f}z_f}}$'.format(A6,b6))
    ax1.legend(loc=0,fontsize=fs-2)

    ax1.set_xlim(18,77)
    ax1.set_xlabel('Tower height $(m)$',fontsize=fs)
    ax1.set_ylabel(r'Optimal {}'.format(ylabs[vary]),fontsize=fs)
    ax1.spines['left'].set_color(cLCOH)
    ax1.tick_params(axis='y', colors=cLCOH,size=10)
    ax1.yaxis.label.set_color(cLCOH)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.grid()
    # fig.savefig(fldr_rslts+'FINAL_LCOH_zf_fzv.png', bbox_inches='tight')
    plt.show()
    
    print(vary,A6,b6,r2)

sys.exit()
#####################################
#%%% LCOH MAP

file_rslts = 'Optim_TPR_0D_quick.csv'
df = pd.read_csv(file_rslts,index_col=0)
df = df.round({'Q_av':2})




#%%% INFLUENCE OF RADIATION FLUX (OLD)

file_rslts = 'Optim_TPR_0D_quick.csv'
df = pd.read_csv(file_rslts,index_col=0)
df = df.round({'Q_av':2})
zfs = [50]
for zf in zfs:
    df = pd.read_csv(file_rslts,index_col=0)
    df = df.round({'Q_av':2})
    pd.set_option('display.max_columns', None)
    df = df[(df.zf==zf)]
    df.sort_values(['Q_av','Prcv'],inplace=True)
    
    
    mins = []
    fig, ax1 = plt.subplots(figsize=(9,6))
    Q_avgs = df.Q_avg_i.unique()[:-2]
    for Q_av in Q_avgs:
        df2 = df[(df.Q_avg_i==Q_av)]
        df2.drop_duplicates(subset=['Prcv','zf','Q_avg_i'],inplace=True)
        # df2.drop_duplicates(inplace=True)
        ax1.plot(df2.Prcv,df2.LCOH,label=r'${:.2f} MW/m^2$'.format(Q_av))
        
        idx_min   = df2.LCOH.idxmin()
        Prcv_min  = df2.loc[idx_min]['Prcv']
        LCOH_min  = df2.loc[idx_min]['LCOH']
        Nhel_min  = df2.loc[idx_min]['N_hel']
        etaSF_min = df2.loc[idx_min]['eta_SF']
        fzv_min   = df2.loc[idx_min]['fzv']
        mins.append([zf,Prcv_min,LCOH_min,Nhel_min,etaSF_min,fzv_min,Q_av])
        
        # bounds = (df2.Prcv.min(),df2.Prcv.max())
        # args = (df2.Prcv,df2.LCOH)
        # res = spo.minimize_scalar(f_minPoly, bounds=bounds, args=args, method='bounded')
        # Prcv_min = res.x
        # LCOH_min = f_minPoly(Prcv_min,*args)
        # Nhel_min = spi.interp1d(df2.Prcv, df2.N_hel, kind='cubic',fill_value='extrapolate')(Prcv_min)
        # etaSF_min = spi.interp1d(df2.Prcv, df2.eta_SF, kind='cubic',fill_value='extrapolate')(Prcv_min)
        # fzv_min  = spi.interp1d(df2.Prcv, df2.fzv, kind='cubic',fill_value='extrapolate')(Prcv_min)
        # mins.append([zf,Prcv_min,LCOH_min,Nhel_min,etaSF_min,fzv_min,Q_av])
    
    # mins = df.loc[mins]
    mins = pd.DataFrame(mins,columns=('zf','Prcv','LCOH','Nhel','eta_SF','fzv','Q_av'))
    mins.sort_values(by='Q_av',inplace=True)
    # args = (mins.Prcv,mins.LCOH)
    # Prcvs = np.arange(mins.Prcv.min(),mins.Prcv.max(),0.1)
    # LCOHs = f_minPoly(Prcvs,*args)
    ax1.plot(mins.Prcv,mins.LCOH,c='k',lw=2,marker='s',label='min(LCOH)')
    ax1.scatter(mins.Prcv,mins.LCOH,c='mediumblue',marker='s',s=100)
    
    ax1.set_ylim(20,35)
    # ax1.plot(Prcvs,LCOHs,lw=3,c='mediumblue',ls='--',label='min(LCOH)')
    ax1.set_ylabel(r'LCOH $(USD/MW_t)$',fontsize=fs)
    ax1.set_xlabel('Receiver Power $(MW_t)$',fontsize=fs)
    ax1.legend(loc=1)
    ax1.grid()
    plt.show()
    # fig.savefig('LCOH_Prcv_Qavg_zf_{:.0f}m.png'.format(zf), bbox_inches='tight')

