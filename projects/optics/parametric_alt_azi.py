# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:47:41 2020

@author: z5158936

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import scipy.optimize as spo
import cantera as ct
from matplotlib import cm
import matplotlib.patches as patches

from bdr_csp import bdr as BDR
from antupy.solar import Sun


####################################

def A_rqrd(rO, *args):
#This function is to calculate the required area given some
    A_rcv_rq,Dsgn,CST,Cg = args
    xrc,yrc,zrc = CST['xrc'],CST['yrc'],CST['zrc']
    CPC = {'Dsgn':Dsgn,'rO':rO,'Cg':Cg}
    CPC   = BDR.CPC_Params( CPC, xrc,yrc,zrc)
    A_rcv = CPC['A_rcv']
    return A_rcv - A_rcv_rq


####################################
def Receiver_0D(CST):
    #######################################
    def h_conv_nat_amb(T_s, T_inf, L, f):
        """
        Correlation for natural convection in upper hot surface horizontal plate
        T_s, T_inf          : surface and free fluid temperatures [K]
        L                   : characteristic length [m]
        """
        T_av = ( T_s + T_inf )/2
        mu, k, rho, cp = f.viscosity, f.thermal_conductivity, f.density_mass, f.cp_mass
        alpha = k/(rho*cp)
        beta = 1./T_av
        visc = mu/rho
        Pr = visc/alpha
        g = 9.81
        
        Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
        if Ra > 1e4 and Ra < 1e7:
            Nu = 0.54*Ra**0.25
            h = (k*Nu/L)
        elif Ra>= 1e7 and Ra < 1e9:
            Nu = 0.15*Ra**(1./3.)
            h = (k*Nu/L)
        else:
            h = 1.52*(T_s-T_inf)**(1./3.)
            # print('fuera de Ra range: Ra= '+str(Ra))
        return h
    ###############################################
    Prcv, qi, Tin, Tout  = [CST[x] for x in ['P_rcv', 'Q_av', 'T_in', 'T_out']]
    ###############################################
    rho_b = 1810
    k_b   = 0.7
    Tamb = 300
    air  = ct.Solution('air.xml')
    ab_p = 0.91
    em_p = 0.75
    tz = 0.05

    Tp = 0.5*(Tin+Tout)
    Tsky = Tamb -15.
    air.TP = (Tp+Tamb)/2., ct.one_atm
    hcov = 10.*h_conv_nat_amb(Tp, Tamb, 5.0, air)
    hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
    hrc = hcov + hrad
    qloss = hrc * (Tp - Tamb)
    eta_rcv = (qi*1e6*ab_p - qloss)/(qi*1e6)
    cp = 365*(Tp+273.15)**0.18
    t_res = rho_b * cp * tz * (Tout - Tin ) / (qi*1e6*ab_p - hrc * (Tp - Tamb))
    
    m_p = Prcv*1e6 / (cp*(Tout-Tin))
    P_SF = Prcv / eta_rcv
    Arcv = P_SF / qi
    Ltot = Arcv**0.5
    vx   = Ltot / t_res
    tz   = m_p * t_res / (Arcv * rho_b)
    
    return  [eta_rcv,P_SF,Arcv,t_res,tz,vx]


####################################
def Receiver_0D_v2(qi):
    #######################################
    def h_conv_nat_amb(T_s, T_inf, L, f):
        """
        Correlation for natural convection in upper hot surface horizontal plate
        T_s, T_inf          : surface and free fluid temperatures [K]
        L                   : characteristic length [m]
        """
        T_av = ( T_s + T_inf )/2
        mu, k, rho, cp = f.viscosity, f.thermal_conductivity, f.density_mass, f.cp_mass
        alpha = k/(rho*cp)
        beta = 1./T_av
        visc = mu/rho
        Pr = visc/alpha
        g = 9.81
        
        Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
        if Ra > 1e4 and Ra < 1e7:
            Nu = 0.54*Ra**0.25
            h = (k*Nu/L)
        elif Ra>= 1e7 and Ra < 1e9:
            Nu = 0.15*Ra**(1./3.)
            h = (k*Nu/L)
        else:
            h = 1.52*(T_s-T_inf)**(1./3.)
            # print('fuera de Ra range: Ra= '+str(Ra))
        return h
    ###############################################
    Tin, Tout  = 800., 1000.
    ###############################################
    Tamb = 300
    air  = ct.Solution('air.xml')
    ab_p = 0.91
    em_p = 0.75

    Tp = 0.5*(Tin+Tout)
    Tsky = Tamb -15.
    air.TP = (Tp+Tamb)/2., ct.one_atm
    hcov = 10.*h_conv_nat_amb(Tp, Tamb, 5.0, air)
    hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
    hrc = hcov + hrad
    qloss = hrc * (Tp - Tamb)
    eta_rcv = (qi*1e6*ab_p - qloss)/(qi*1e6)
    
    return eta_rcv

##############################################
#### MAIN SIMULATION
fldr_rslt = 'Grid/'
file_rslt = fldr_rslt+'1-BDR_AltAzi_2.txt'
fldr_dat  = 'Datasets_Thesis/Azi_Alt/'
text_header  = 'alt\tazi\trO\teta_cos\teta_blk\teta_att\teta_hbi\teta_cpi\teta_cpr\teta_CPC\teta_BDR\teta_SF\tQmax\n'
# f = open(file_rslt,'w'); f.write(text_header); f.close()

#### DESIGN CONDITION
# Getting the list of heliostats from the design case
zfs   = [50]
Dsgns = ['A']
Prcvs  = np.arange(25,50,5)

Cgs   = 2.0
zrc   = 25.
Npan = 16
Ah1  = 7.07*7.07
lat  = -23.
#Initial hints for optimisation
zf    = 50
Prcv  = 35
Dsgn  = 'A'
fzv   = 0.88525
rmax  = 19.23
Cg    = 2.0
Q_av  = 0.75219             #[MW/m2]

plot = True
basecase = True

#Getting the final result for the optimal point found. First update CST and CPC dicts
print('Getting final result from Optimisation')
CST = BDR.CST_BaseCase(zf=zf, rmax=rmax, fzv=fzv, P_rcv=Prcv, zrc=zrc, N_pan=Npan, A_h1=Ah1, Dsgn_CPC=Dsgn, Cg_CPC=Cg, Q_av=Q_av)
eta_rcv, P_th, Arcv = Receiver_0D(CST)[:3]
rO = spo.fsolve(A_rqrd, 1.0, args=(Arcv, Dsgn, CST, Cg))[0]
CPC = BDR.CPC_Params({'Dsgn':Dsgn,'rO':rO,'Cg':Cg},0.,0.,zrc)

file_SF = fldr_dat+'Base_height-{:.0f}'.format(zf)
R0, SF = BDR.Rays_Dataset( file_SF, read=True, save=True, N_pan=Npan )
CST = BDR.CST_BaseCase(zf=zf, rmax=rmax, fzv=fzv, P_rcv=Prcv, eta_rc=eta_rcv, zrc=zrc, N_pan=Npan, A_h1=Ah1, Dsgn_CPC=Dsgn, rO_CPC=rO, Cg_CPC=Cg, Q_av=Q_av, Arcv=Arcv)
R2, SF, CST = BDR.Optical_Sim(R0, SF, CST)
C = BDR.LCOH_estimation(SF, CST)

rmin,rmax,S_HB,S_CPC = [CST[x] for x in ['rmin','rmax','S_HB','S_CPC']]
Etas = SF[SF['hel_in']].mean()
hlst = SF[SF.hel_in].index
N_hel = len(hlst)
eta_SF = Etas['Eta_SF']
eta_hbi = Etas['Eta_hbi']
eta_BDR = Etas['Eta_BDR']
eta_CPC = Etas['Eta_cpi']*Etas['Eta_cpr']
result = [Dsgn, zf, Prcv, fzv, Q_av, zrc, rmax, len(hlst), eta_SF, eta_hbi, eta_BDR, eta_CPC, S_HB, S_CPC, C['SCH'], C['LCOH'], C['LCOE']]
print(Dsgn+'\t'+'\t'.join('{:.3f}'.format(x) for x in result[1:]))

######################################################
#### CALCULATIONS
######################################################
for (alt,azi) in [(alt,azi) for alt in np.arange(90,91,10) for azi in np.arange(180,301,10)]:
    break
    case = 'Alt_{:d}-Azi_{:d}'.format(alt,azi)
    
    #Getting all rays with direction cosines
    file_SF = fldr_dat+'alt-{:d}_azi-{:d}'.format(alt,azi)
    
    R0, SF = BDR.Rays_Dataset( file_SF, read=True, save=True )
    R0 = R0[R0['hel'].isin(hlst)]
    R1 = BDR.HB_direct( R0 , CST )
    R1['hel_in'] = R1['hel'].isin(hlst)
    R1['hit_hb'] = (R1['rb']>rmin)&(R1['rb']<rmax)
    
    #Shadowing
    SF = BDR.Shadow_simple(CST,SF)
    #Interceptions with CPC
    R2 = BDR.CPC_direct(R1,CPC,CST)
    
    #Calculating efficiencies
    SF2 = R2.groupby('hel')[['hel_in','hit_hb','hit_cpc','hit_rcv']].sum()
    SF['Eta_att'] = BDR.Eta_attenuation(R1)
    SF['Eta_hbi'] = SF2['hit_hb']/SF2['hel_in']

    SF['Eta_cpi'] = SF2['hit_cpc']/SF2['hit_hb']
    SF['Eta_cpr'] = SF2['hit_rcv']/SF2['hit_cpc']

    Nr_cpc = R2[R2['hit_rcv']].groupby('hel')['Nr_cpc'].mean()
    SF['Eta_cpr'] = SF['Eta_cpr'] * CST['eta_rfl']**Nr_cpc
    
    SF['Eta_hel'] = SF['Eta_cos'] * SF['Eta_blk'] * SF['Eta_att'] * CST['eta_rfl']
    SF['Eta_BDR'] = (CST['eta_rfl'] * SF['Eta_hbi']) * (SF['Eta_cpi'] * SF['Eta_cpr'])
    SF['Eta_SF']  = SF['Eta_hel'] * SF['Eta_BDR']
    
    SF['hel_in'] = SF.index.isin(hlst)
    R2['hel_in'] = R2['hel'].isin(hlst)
    
    Etas = SF[SF['hel_in']].mean()
    eta_SF = Etas['Eta_SF']
    eta_hbi = Etas['Eta_hbi']
    eta_BDR = Etas['Eta_BDR']
    eta_CPC = Etas['Eta_cpi']*Etas['Eta_cpr']
    
    if plot:
        ### RECEIVER APERTURE RADIATION
        Nx = 100; Ny = 100
        CPC = BDR.CPC_Params({'Dsgn':Dsgn,'rO':rO,'Cg':Cg},0.,0.,zrc)
        N_CPC,V_CPC,rO,rA,x0,y0 = [CPC[x] for x in ['N','V','rO','rA','x0','y0']]
        out   = R2[(R2['hel_in'])&(R2['hit_rcv'])].copy()
        xmin = out['xr'].min(); xmax = out['xr'].max()
        ymin = out['yr'].min(); ymax = out['yr'].max()
        xmin=min(x0)-rA/np.cos(np.pi/V_CPC) 
        xmax=max(x0)+rA/np.cos(np.pi/V_CPC)
        ymin=min(y0)-rA
        ymax=max(y0)+rA
        
        dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
        Nrays = len(out)
        Fbin    = Etas['Eta_SF'] * (CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*Nrays)
        Q_BDR,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
        Q_BDR = Fbin * Q_BDR
        Q_max = Q_BDR.max()
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, aspect='equal')
        X, Y = np.meshgrid(X, Y)
        f_s = 16
        vmin = 0
        vmax = (np.ceil(Q_BDR.max()/100)*100)
        surf = ax.pcolormesh(X, Y, Q_BDR.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
        ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
        cb = fig.colorbar(surf, shrink=0.5, aspect=4)
        cb.ax.tick_params(labelsize=f_s-2)
        
        if Dsgn=='F':
            ax.add_artist(patches.Circle((x0[0],y0[0]), rO, zorder=10, color='black', fill=None))
        else:
            for i in range(N_CPC):
                radius = rO/np.cos(np.pi/V_CPC)
                ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_CPC,radius, np.pi/V_CPC, zorder=10, color='black', fill=None))
                radius = rA/np.cos(np.pi/V_CPC)
                ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_CPC,radius, np.pi/V_CPC, zorder=10, color='black', fill=None))
            
        fig.text(0.77,0.27,'Main Parameters',fontsize=f_s-3)
        fig.text(0.77,0.25,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
        fig.text(0.77,0.23,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
        fig.text(0.77,0.21,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
        fig.text(0.77,0.19,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
        ax.set_title('b) Radiation flux in receiver aperture',fontsize=f_s)
        
        fig.text(0.77,0.70,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
        fig.savefig(fldr_rslt+case+'_radmap_out.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        ### SOLAR FIELD ##################
        f_s=18
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        SF2 = SF[SF['hel_in']].loc[hlst]
        N_hel = len(hlst)
        vmin = SF2['Eta_SF'].min()
        vmax = SF2['Eta_SF'].max()
        surf = ax1.scatter(SF2['xi'],SF2['yi'], s=5, c=SF2['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s)
        cb.ax.locator_params(nbins=4)
        
        fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']), fontsize=f_s)
        fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
        # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
        ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
        for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        ax1.grid()
        fig.savefig(fldr_rslt+case+'_SF.png', bbox_inches='tight')
        # plt.show()
        plt.close(fig)
    
    ########### WRITING RESULTS ###############
    text_r = '\t'.join('{:.4f}'.format(x) for x in [alt, azi, rO, Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_cpi'], Etas['Eta_cpr'], Etas['Eta_BDR'], Etas['Eta_SF'], Q_max])+'\n'
    print(text_r[:-2])
    f = open(file_rslt,'a'); f.write(text_r);    f.close();

######################################
#### PLOTING MAP
df = pd.read_table(file_rslt,sep='\t',header=0)
df.sort_values(by=['alt','azi'],axis=0,inplace=True)
print(df)

# for lbl in ['eta_cos','eta_BDR','eta_SF']:
# for lbl in ['eta_SF']: 
lbl='eta_SF'
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
f_s = 16

alt, azi = df['alt'].unique(), df['azi'].unique()
eta = np.array(df[lbl]).reshape(len(alt),len(azi))
f_eta = interp2d(azi, alt, eta, kind='cubic')

alt2,azi2 = np.arange(0,90,0.1), np.arange(180,301,0.1)
eta2 = f_eta(azi2,alt2)
azi2,alt2=np.meshgrid(azi2,alt2)

azi,alt = np.meshgrid(azi,alt)
vmin=np.floor(df[lbl].min()*10)/10
vmax=np.ceil(df[lbl].max()*10)/10
vmin = 0.44
vmax = 0.56
# vmin=0.38;vmax=0.50
lvs = 13; levels = np.linspace(vmin, vmax, lvs)
mapx = ax.contour(azi2, alt2, eta2, colors='w', levels=levels , vmin=vmin , vmax=vmax,extend='both')
ax.clabel(mapx,mapx.levels,inline=True,fmt='%.2f',fontsize=f_s)

mapp = ax.contourf(azi2, alt2, eta2, levels=levels , vmin=vmin , vmax=vmax, cmap=cm.viridis,extend='both')

ax.set_xlabel('Azimuth (dgr)',fontsize=f_s)
ax.set_ylabel('Elevation (dgr)',fontsize=f_s)

xmin, xmax, ymin, ymax = 180,300,10,90

ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks(np.arange(xmin,xmax+1,10))
ax.set_yticks(np.arange(ymin,ymax+1,10))
ax.set_xticklabels(labels=np.arange(xmin,xmax+1,10), fontsize=f_s)
ax.set_yticklabels(labels=np.arange(ymin,ymax+1,10), fontsize=f_s)

plt.subplots_adjust(right=0.8) 
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = plt.colorbar(mapp,cax=cbar_ax)
cbar_ax.set_yticklabels(labels=cbar_ax.get_yticklabels(), fontsize=f_s-2)
# cbar_ax.set_yticklabels(labels=cbar_ax.get_yticklabels(), fontsize=f_s-2)
plt.colorbar(mapp,cax=cbar_ax,ticks=np.linspace(vmin,vmax,int(lvs/2)+1))
cbar.set_label('Optical efficiency ', rotation=-270, fontsize=f_s)


sol = Sun()
dfs = []
days = [80,172,355];
for (N,t) in [(N,t) for N in days for t in np.arange(5,12.1,0.1)]:
    sol = Sun()
    sol.lat = -25.
    sol.lng = 133.9
    sol.update(N=N,t=t,UTC=9.5)
    sol.azim = abs(sol.azim) + 180.
    
    dfs.append([N,t,sol.h,sol.altit,sol.azim,sol.hsunset])

dfs = pd.DataFrame(dfs,columns=['N','t','h','alt','azi','hss'])

lss = [':','--','-']
lbls = ['Eqquinox','Winter Solstice', 'Summer Solstice']

for i in range(len(days)):
    N = days[i]
    ax.plot(dfs[dfs['N']==N]['azi'],dfs[dfs['N']==N]['alt'],lw=3, c='k', ls=lss[i], label=lbls[i])
    
ax.scatter([180],90.+sol.lat,c='darkred',s=120,label='Design Point',zorder=10)

ax.grid()
ax.legend(fontsize=f_s-2)
fig.savefig('Fig_azi-alt.png', bbox_inches='tight')
plt.show()


#####################################################
#### SIMULATING TMY 
#####################################################

N_hel  = len(hlst)

eta_sg = 0.95
eta_pb = 0.50
C_stg = 0.3                    # USD/kg (Kiang et al. 2019). Black alumina
# E_stg = 4198                   # kJ/m3  (Kiang et al. 2019). Black alumina
rho_stg = 3960                 # kg/m3  (Kiang et al. 2019). Black alumina
cp_stg = 1.05                  # kJ/kg-K (Kiang et al. 2019). Black alumina
dT_stg = 200                   # K
HtD_stg = 0.5                  # height to diameter ratio for storage tank

T_stg  = 8.                    # hr storage
SM     = 2.                    # (-) Solar multiple (Prcv/Prcv_nom)

data=[]
for (T_stg,SM) in [(T_stg,SM) for T_stg in np.arange(8,8.1,4) for SM in np.arange(2.0,2.1,0.5)]:
    P_SF   = Prcv/eta_rcv
    P_pb   = Prcv / SM
    P_el   = eta_sg * eta_pb * P_pb
    Q_stg  = T_stg * P_pb
    M_stg  = (P_pb * T_stg / (cp_stg * dT_stg) * 3600 * 1e3)
    V_stg  = M_stg / rho_stg
    H_stg  = (V_stg/np.pi)**(1./3)
    D_stg  = 2*H_stg
    
    file_TMY = 'Weather/Alice_Springs_Real2000_Created20130430.csv'
    TMY  = pd.read_csv(file_TMY,header=1)
    TMY = TMY[['Date (MM/DD/YYYY)','Time (HH:MM)','DNI (W/m^2)','Dry-bulb (C)','Dew-point (C)', 'RHum (%)', 'Wdir (degrees)','Wspd (m/s)']]
    TMY.columns = ['date','time','DNI','Tamb','Tdp','RH','Wdir','Wspd']
    
    #I have to correct the day phase
    TMY['time'] = np.where((TMY['time'] == '24:50'), '00:50', TMY['time'])
    
    TMY['dt'] = TMY['date'] + ', ' + TMY['time']
    TMY['dt'] = pd.to_datetime(TMY['dt'],format='%m/%d/%Y, %H:%M')
    
    ####### Initial estimation for Thermal Capacity Factor
    #!!! ###### COPY THIS CODE INSIDE THE LCOH ESTIMATION FUNCTION
    # TMY['P_SF'] = eta_SF * TMY['DNI'] * N_hel * CST['A_h1'] * 1e-6
    # Q_av = TMY['P_SF'] / Arcv
    # TMY['eta_rcv'] = Receiver_0D_v2(Q_av)
    # TMY['eta_rcv'] = np.where(TMY['eta_rcv']>0,TMY['eta_rcv'],0)
    # TMY['P_rcv'] = TMY['eta_rcv'] * TMY['P_SF']
    # CF_th=sum(TMY['P_rcv'])/(CST['P_rcv']*len(TMY))
    
    ###########################################################
    
    
    #Read the azi-alt file
    df.sort_values(by=['alt','azi'],axis=0,inplace=True)
    
    alt, azi = df['alt'].unique(), df['azi'].unique()
    eta = np.array(df['eta_SF']).reshape(len(alt),len(azi))
    f_eta = interp2d(azi, alt, eta, kind='linear')          #Function
    
    TMY['N']   = TMY['dt'].dt.dayofyear
    
    #Provisional
    TMY['t']   = (TMY['dt'].dt.hour + TMY['dt'].dt.minute/60) - 4*(15*9.5 - sol.lng)/60 -0.5
    print(TMY[TMY['N']==1])
    
    TMY['alt']    = np.nan
    TMY['azi']    = np.nan
    TMY['h']      = np.nan
    TMY['hss']    = np.nan
    TMY['eta_SF'] = 0
    TMY['eta_rcv']= 0
    TMY['P_SF']   = 0.              # Solar energy from Solar Field (receiver aperture)
    TMY['P_rcv']  = 0.              # Thermal energy delivered from receiver
    TMY['P_pb']   = 0.              # Thermal energy delivered to Power block to generate electricity
    TMY['P_el']   = 0.              # Power generated from the power block
    TMY['L_stg']  = 0.              # Capacity on the storage tank
    
    L_stg = 0.      #Storage load
    for idx,row in TMY.iterrows():
        
        if row['DNI']>0:
            sol = Sun()
            sol.lat = -23.8
            sol.lng = 133.8
            sol.update(N=row['N'],t=row['t'],UTC=9.5)
            
            TMY.loc[idx,'alt']  = sol.altit
            TMY.loc[idx,'azi']  = sol.azim + 180.
            TMY.loc[idx,'h']    = sol.h
            TMY.loc[idx,'hss']  = sol.hsunset
            
            eta_SF_h = f_eta(abs(sol.azim),sol.altit)[0]
            P_SF_h  = eta_SF_h  * row['DNI'] * N_hel * CST['A_h1'] * 1e-6
            
            CST['Q_av'] = P_SF_h / Arcv
            eta_rcv_h = Receiver_0D(CST)[0]
            eta_rcv_h = 0 if eta_rcv_h<0. else eta_rcv_h
            P_rcv_h = eta_rcv_h * P_SF
        
        else:
            eta_SF_h, P_SF_h, eta_rcv_h, P_rcv_h = 0,0,0,0
        
        P_pb_h = P_pb
        L_stg_h = L_stg + P_rcv_h - P_pb_h
        
        if L_stg_h > Q_stg:
            P_rcv_h = Q_stg - L_stg + P_pb_h
            L_stg_h = Q_stg
        
        if L_stg_h < 0:
            P_pb_h = L_stg
            L_stg_h = 0
        
        P_el_h = P_pb_h * eta_sg * eta_pb
        L_stg = L_stg_h
        
        # print(str(row['dt'])+'\t'+'\t'.join('{:.4f}'.format(x) for x in [eta_SF_h, eta_rcv_h, P_SF_h, P_rcv_h, L_stg]))
        TMY.loc[idx,['eta_SF','eta_rcv','P_SF','P_rcv']] = eta_SF_h, eta_rcv_h, P_SF_h, P_rcv_h
        TMY.loc[idx,['P_pb','P_el','L_stg']] = P_pb_h, P_el_h, L_stg_h
    
    ###########################################
    TMY2 = TMY[TMY['dt']<'2000-01-05']       #Only january for now
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(212)
    
    ax1.plot(TMY2.index,TMY2['P_rcv'],lw=2,label='P_rcv')
    ax1.plot(TMY2.index,TMY2['P_el'],label='P_el')
    # ax1.plot(TMY2.index,TMY2['L_stg'],label='L_stg')
    ax1.plot([],[],lw=2,label='DNI',c='C3')
    ax2.plot(TMY2.index,TMY2['DNI'],lw=2,label='DNI',c='C3')
    
    ax3.plot(TMY2.index,TMY2['eta_SF'],lw=2,c='C0',label='eta_SF')
    ax3.plot(TMY2.index,TMY2['eta_rcv'],lw=2,c='C1',label='eta_rcv')
    ax3.plot(TMY2.index,TMY2['L_stg']/Q_stg,lw=2,c='C2',label='L_stg')
    
    ax1.set_xlabel(r'Time (hr)',fontsize=f_s)
    ax1.set_ylabel('Energy (MWh)',fontsize=f_s)
    ax2.set_ylabel('DNI (W/m2)',fontsize=f_s)
    ax3.set_xlabel(r'Time (hr)',fontsize=f_s)
    ax3.set_ylabel('Efficiency or Storage level(-)',fontsize=f_s)
    ax3.set_yticks(np.arange(0.0,1.01,0.1))
    ax1.grid()
    ax3.grid()
    ax1.legend()
    ax3.legend()
    # fig.savefig('Fig_ThermaPower_hours.png', bbox_inches='tight')
    plt.show()
        
        # print(N,h,sol.t,sol.altit,sol.azim)
    
    ##### Global numbers ####################
    dff = TMY.groupby(by=[TMY['dt'].dt.month]).sum()
    dff = dff[['DNI','eta_SF','P_rcv','P_el']]
    Nhrs = TMY.groupby(by=[TMY['dt'].dt.month]).count()['dt']
    dff['CF_th'] = dff['P_rcv'] / (Nhrs * CST['P_rcv'] )
    dff['CF_el'] = dff['P_el'] / (Nhrs * P_el )
    dff['r_CF'] = dff['CF_th']/dff['CF_el']
    CF_sf=sum(TMY['P_rcv'])/(CST['P_rcv']*len(TMY))
    CF_pb=sum(TMY['P_el'])/(P_el*len(TMY))
    
    CST['CF_sf'] = CF_sf
    CST['CF_pb'] = CF_pb
    CST['SM']    = SM
    CST['T_stg'] = T_stg
    CST['P_rcv'] = Prcv
    CST['P_el']  = P_el
    C = BDR.LCOH_estimation(SF,CST)
    data.append([T_stg,SM,CF_sf,CF_pb,CF_sf/CF_pb,C['LCOH'],C['LCOE'],P_el])
    print('\t'.join('{:.4f}'.format(x) for x in data[-1]))

data2 = pd.DataFrame(data,columns=['T_stg','SM','CF_sf','CF_pb','r_CF','LCOH','LCOE','P_el'])
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
i=0
for T_stg in data2.T_stg.unique():
    df2=data2[data2.T_stg==T_stg]
    # ax1.plot(df2.T_stg,df2.LCOH,lw=2,c='C'+str(i))
    ax1.plot(df2.SM,df2.LCOE,lw=2,c='C'+str(i),label='T_stg={:.0f}'.format(T_stg))
    ax2.plot(df2.SM,df2.CF_pb,lw=2,ls='--',c='C'+str(i))
    i+=1
ax1.set_xlabel('Solar Multiple (-)',fontsize=f_s)
ax1.set_ylabel('LCOE (USD/MWh)',fontsize=f_s)
ax2.set_ylabel('Capacity Factor (-)',fontsize=f_s)
ax1.set_ylim(60,150)
ax2.set_ylim(0,1)
ax1.legend()
ax1.grid()
plt.show()


# ###########################################
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111)
# hist = ax.hist(TMY['Pth'],bins=8,range=(0,40),rwidth=0.75)
# Nh = sum(hist[0])
# for i in range(len(hist[0])):
#     ax.annotate('{:.1f}%'.format(100*hist[0][i]/Nh),(hist[1][i]+2.5,hist[0][i]-100), ha='center', c='k')

# ax.set_xlabel(r'Solar output power ($MW_{th}$)',fontsize=f_s)
# ax.set_ylabel('Number of hours (hrs) (on-sun hours)',fontsize=f_s)
# ax.grid()
# fig.savefig('Fig_ThermaPower_hours.png', bbox_inches='tight')
# plt.show()

# ################################

# fig = plt.figure(figsize=(9, 6))
# ax1 = fig.add_subplot(111)
# ax1.bar(dff.index,dff['Pth'],width=0.8)
# for i in range(len(dff)):
#     aux = dff.iloc[i]
#     ax1.annotate('{:.1f}%'.format(100*aux['CF']),(aux.name, aux['Pth']-2 ), ha='center', c='w')

# ax1.set_ylim(0,35)
# ax1.set_xticks(np.arange(1,13,1))
# labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# ax1.set_xticklabels( labels=labels, fontsize=f_s)


# ax1.set_xlabel('Month (-)',fontsize=f_s)
# ax1.set_ylabel('Average Therma Power (on-sun hours) (MW)',fontsize=f_s)
# ax1.grid(axis='y')
# plt.show()
# fig.savefig('Fig_AveragePower.png', bbox_inches='tight')

# ###################################
# TMY.to_csv('AliceSprings_results.csv')
# # df.drop('Pth_real',axis=1,inplace=True)
# df.to_csv('BDR_alt-azi_etas.csv')
# print(sum(TMY[TMY['Pth'].notnull()]['Pth'])/(CST['P_th']*8760))
# print(TMY[TMY['N']==1])