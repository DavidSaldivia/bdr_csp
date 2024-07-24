# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:00:57 2020

@author: z5158936
"""
if __name__ ==  '__main__':
    __spec__ = None
    
    import pandas as pd
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as patches
    from importlib import reload
    from multiprocessing import Pool
    from functools import partial
    import gc
    import BDR
    # BDR = reload(BDR)
    
    ################################################
    ################ MAIN PROGRAM ##################
    ################################################
    
    op_mode = 1
    
    zf   = 50                   #Tower height
    fzv  = 0.83                 #vertex ratio
    Pel  = 10.                  #Required Electrical
    N_hel = 1200                #Number of heliostats
    
    Dsgn_CPC = 'A'              #Design of CPC array
    xrc,yrc,zrc = 0,0,10        #Center of CPC aperture plane
    rO = 2.0                    #CPC outlet radious (Aperture Receiver radious)
    Cg = 2.0                    #CPC concentration ratio
    
    fldr_rslt = 'Result'
    case = 'Base_Case'
    

    #First, it is necessary generate a dictionary with the CST plant characteristics.
    #In this case the parameters defined are zf=50, fzv=0.83, Pel=10MWe
    CST = BDR.CST_BaseCase(zf=zf,fzv=fzv,P_el=Pel)
    
    #The initial dataset (from SolarPILOT) is read and converted in a proper DataFrame
    #The resulting dataframe is R0, it also gives Eta_blk and Eta_cos for each heliostat
    file_SF = 'Rays_Dataset'
    R0, Eta_blk, Eta_cos = BDR.Rays_Dataset( file_SF, convert=False )
    
    #The interceptions with Hyperboloid are calculated. It requires R0 and CST
    #If file_HB is given and print_out=True, the resulting DataFrame is saved in that file
    #The resulting DataFrame is called R1
    file_HB = 'Rays_HB'
    R1 = BDR.HB_Intercepts(R0,CST,file_HB=file_HB,print_out=False)
    
    #########################################################
    #The following code gets the intercepts with CPC surface
    #First the CPC parameters are defined
    #It requires the Design, the CPC central position
    #and two parameters from rO, rA, Cg, H, given through a dict
    CPC = BDR.CPC_Params( Dsgn_CPC,xrc,yrc,zrc, {'rO':rO,'Cg':Cg} )
    
    #The different CPC parameters are in the dict CPC.
    #The centers for the CPC array are calculated and given in x0,y0
    
    Dsgn_CPC,N_CPC,V_CPC,H_CPC,rA,rO,A_rcv = [ CPC[x] for x in ['Dsgn','N','V','H','rA','rO','A_rcv'] ]
    x0,y0 = BDR.CPC_Centers(Dsgn_CPC,rA,xrc,yrc)
    
    #The rays that enter the CPC array are calculated
    nRA=[]
    zA = rA**2     # Aperture and output height for CPC coordinate system
    xc,yc  = R1['xc'], R1['yc']
    for i in range(N_CPC):
        zz = BDR.CPC_Z( xc, yc, V_CPC, x0[i],y0[i])    #Checking on aperture plane
        nRA.append((zz<zA)*(i+1))
    R1['hit_cpc'] = np.array(nRA).max(axis=0)

    N_proc = 4
    pool    = Pool( processes = N_proc )
    R2aux   = pool.map(partial(BDR.CPC_worker_outdist, params=[N_CPC,V_CPC,rA,rO,[x0,y0]]), R1[BDR.R1_cols].values.tolist() )
    R2      = pd.DataFrame( R2aux, columns=BDR.R2_cols )
    pool.close()
    del R2aux
    
    #The final positions for the rays are in R2
    
    ###########################################
    ###########################################
    # The optimisation for the case can be calculated directly with 'BDR.Optimisation'        
    # See the function documentation to know the outputs
    R2, Etas, SF, CPC, HB, hlst, Q_rcv, status = BDR.Optimisation(CST,CPC,file_SF,file_HB)


    #####################################################################
    #####################################################################
    ############################ PLOTING ################################
    #####################################################################
    #####################################################################
    
    N_CPC,V_CPC,rO,rA,Cg = [ CPC[x] for x in ['N','V','rO','rA','Cg'] ]    
    xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
    x0,y0 = BDR.CPC_Centers(Dsgn_CPC,rA,xrc,yrc)
    xCA, yCA, xCO, yCO = [],[],[],[]
    for i in range(N_CPC):
        xA,yA = BDR.CPC_XY_R(rA,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)
        xO,yO = BDR.CPC_XY_R(rO,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)    
        xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
    xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
    xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
    
    
    ###### Ploting CPC shape
    Nx = 100; Ny = 100
    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Ny
    dA = dx*dy
    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
    
    fig = plt.figure(figsize=(10,10))
    # plt.scatter(R2['xr'],R2['yr'],c='b',s=0.01)
    for N in range(N_CPC):
        plt.plot(xCA[N],yCA[N],c='k')
        plt.plot(xCO[N],yCO[N],c='k')
    plt.grid()
    plt.xlim(xmin,xmax);plt.ylim(ymin,ymax);
    fig.savefig(fldr_rslt+'/'+case+'_shape.png', bbox_inches='tight')
    plt.close()
    
    ###### Ploting CPC efficiency points
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(R2[(R2['hit_cpc']==0)]['xc'], R2[(R2['hit_cpc']==0)]['yc'],s=0.1)
    ax.scatter(R2[(R2['hit_cpc']>0)&(~R2['hit_rcv'])]['xc'], R2[(R2['hit_cpc']>0)&(~R2['hit_rcv'])]['yc'],s=0.1,c='gray')
    for N in range(N_CPC):
        ax.plot(xCA[N],yCA[N],c='k')
        ax.plot(xCO[N],yCO[N],c='k')
    ax.grid()
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);
    
    ax.annotate(r'$\eta_{{cpo}}={:.2f}$'.format(Etas['Eta_cpr']),(-2.5,-2.5),fontsize=18, backgroundcolor='white')
    ax.annotate(r'$\eta_{{cpi}}={:.2f}$'.format(Etas['Eta_cpi']),(4.,-5.8),fontsize=18, backgroundcolor='white')
    
    ax.annotate('Design A:',(1.05,0.29),xycoords='axes fraction', fontsize=18)
    ax.annotate('3-hexagon',(1.05,0.26),xycoords='axes fraction', fontsize=18)
    ax.annotate(r'$z_{{f}}={:.1f}m$'.format(zf),(1.05,0.20),xycoords='axes fraction', fontsize=18)
    ax.annotate(r'$f_{{v}}={:.2f}$'.format(fzv),(1.05,0.16),xycoords='axes fraction', fontsize=18)
    ax.annotate(r'$P_{{el}}={:.1f}MW_e$'.format(Pel),(1.03,0.12),xycoords='axes fraction', fontsize=18)
    ax.annotate(r'$C_{{CPC}}={:.1f}$'.format(Cg),(1.05,0.08),xycoords='axes fraction', fontsize=18)
    
    
    kw0 = dict(arrowstyle="Simple, tail_width=2.0, head_width=6, head_length=10", color="k")
    ax.add_patch(patches.FancyArrowPatch((-1.5, -2), (-1.0, -0.5), connectionstyle="arc3,rad=-0.2", zorder=10, **kw0))
    ax.add_patch(patches.FancyArrowPatch((3.9, -5.8), (2.5, -4.5), connectionstyle="arc3,rad=-0.2", zorder=10, **kw0))
    
    plt.show()
    fig.savefig(fldr_rslt+'/'+case+'_no_hitting.png', bbox_inches='tight')
    plt.close()
    
    #####################################################
    ######## HYPERBOLOID RADIATION MAP ##################
    
    f_s = 18
    out2  = R2[(R2['hel_in'])&(R2['hit_hb'])]
    xmin = out2['xb'].min(); xmax = out2['xb'].max()
    ymin = out2['yb'].min(); ymax = out2['yb'].max()
    Nx = 100; Ny = 100
    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
    Fbin = CST['eta_rfl']*Etas['Eta_cos']*Etas['Eta_blk']*(CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*len(out2))
    Q_HB,X,Y = np.histogram2d(out2['xb'],out2['yb'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]],density=False)
    fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(111, title='Ray density on hyperboloid surface (upper view)', aspect='equal')
    ax = fig.add_subplot(111, aspect='equal')
    X, Y = np.meshgrid(X, Y)
    
    vmin = 0
    vmax = (np.ceil(Fbin*Q_HB.max()/10)*10)
    surf = ax.pcolormesh(X, Y, Fbin*Q_HB.transpose(),cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
    ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    fig.text(0.77,0.62,r'$Q_{HB}(kW/m^2)$',fontsize=f_s)
    for tick in ax.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
    for tick in ax.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
    
    # from matplotlib import rc
    # rc('text', usetex=True)
    fig.text(0.77,0.35,'Main Parameters',fontsize=f_s-3)
    fig.text(0.77,0.33,r'$z_{f\;}=50 m$',fontsize=f_s-3)
    fig.text(0.77,0.31,r'$f_{zv}=0.83$',fontsize=f_s-3)
    fig.text(0.77,0.29,r'$z_{rc}=10 m$',fontsize=f_s-3)
    fig.text(0.77,0.27,r'$\eta_{hbi}=0.95$',fontsize=f_s-3)
    
    r1 = patches.Circle((0.,0.0), HB['rlims'][0], zorder=10,color='black',fill=None)
    r2 = patches.Circle((0.,0.0), HB['rlims'][1], zorder=10,edgecolor='black',fill=None)
    ax.add_artist(r1)
    ax.add_artist(r2)
    ax.grid(zorder=20)
    # fig.savefig(fldr_rslt+'/'+case+'_QHB_upper.pdf', bbox_inches='tight')
    fig.savefig(fldr_rslt+'/'+case+'_QHB_upper.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    print(Q_HB.sum())
    del out2
    
    #########################################################
    #Q BDR Distribution
    N_CPC,V_CPC,rO,rA,Cg = [ CPC[x] for x in ['N','V','rO','rA','Cg'] ]    
    xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
    x0,y0 = BDR.CPC_Centers(Dsgn_CPC,rA,xrc,yrc)
    xCA, yCA, xCO, yCO = [],[],[],[]
    for i in range(N_CPC):
        #Plotting hexagons
        xA,yA = BDR.CPC_XY_R(rA,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)
        xO,yO = BDR.CPC_XY_R(rO,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)    
        xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
    xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
    xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, title='Ray density map on CPC aperture plane (upper view)', aspect='equal')
    for N in range(N_CPC):
        ax.plot(xCA[N],yCA[N],c='k')
        ax.plot(xCO[N],yCO[N],c='k')
    Q,x,y,surf = ax.hist2d(R2['xc'],R2['yc'], bins=100, range= [[xmin,xmax], [ymin,ymax]] , cmap=cm.YlOrRd)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=4)
    ax.set_xlabel('X axis (m)');ax.set_ylabel('Y axis (m)');
    fig.savefig(fldr_rslt+'/'+case+'_radmap_ap.png', bbox_inches='tight')
    plt.grid()
    # plt.show()
    plt.close()

    ##############################################################
    Nx = 100; Ny = 100
    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Ny
    dA = dx*dy
    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
    out   = R2[(R2['hel_in'])&(R2['hit_rcv'])]
    Nrays = len(out)
    Fbin    = Etas['Eta_SF'] * (CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*Nrays)
    Q_CPC,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
    Q_max   = Fbin * Q_CPC.max()
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for N in range(N_CPC):
        ax.plot(xCA[N],yCA[N],c='k')
        ax.plot(xCO[N],yCO[N],c='k')
    X, Y = np.meshgrid(X, Y)
    f_s = 16
    vmin = 0
    vmax = 2000
    surf = ax.pcolormesh(X, Y, Fbin*Q_CPC.transpose(),cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
    ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s-2)
    fig.text(0.77,0.65,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
    fig.savefig(fldr_rslt+'/'+case+'_radmap_out.png', bbox_inches='tight')
    plt.grid()
    plt.close()
    
    del X,Y
    
    ################## Solar Field #############################
    R2f = pd.merge( R2 , SF.loc[hlst] , how='inner', on=['hel'] )
    f_s=18
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    
    vmin = 0.0
    vmax = 1.0
    
    surf = ax1.scatter(R2f['xi'],R2f['yi'], s=0.5, c=R2f['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    cb.ax.locator_params(nbins=4)
    
    fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']),fontsize=f_s)
    fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
    ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
    # ax1.set_title('No eta_cpi',fontsize=f_s)
    for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
    for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
    ax1.grid()
    fig.savefig(fldr_rslt+'/'+case+'_SF.png', bbox_inches='tight')
    plt.close(fig)


    del R2f, out, R2, SF, Etas, CST, CPC, HB, hlst
    gc.collect()
