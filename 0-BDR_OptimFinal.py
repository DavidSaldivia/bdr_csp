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
    import gc
    import BDR
    # BDR = reload(BDR)
    
    def A_rqrd(rO,*args):
    #This function is to calculate the required area given some
        A_rcv_rq,Dsgn_CPC,xrc,yrc,zrc,Cg = args
        N_CPC, V_CPC = BDR.CPC_Design(Dsgn_CPC)
        CPC   = BDR.CPC_Params( Dsgn_CPC, xrc,yrc,zrc, {'rO':rO,'Cg':Cg} )
        A_rcv = CPC['A_rcv']
        return A_rcv - A_rcv_rq
    
    ################################################
    ################ MAIN PROGRAM ##################
    ################################################
    
    # Operational Modes:
    #  1             #Testing one single value
    #  2             #Testing different heights, fzvs and powers
    #  3             #Testing different designs and concentration ratio
    
    op_mode = 1
    
    if op_mode == 1:
        zfs   = [50]
        fzvs  = [0.83]
        Cgs   = [2.]
        Dsgns = ['A']
        Pels  = [10.]
        
        fldr_rslt = 'BDR_OptimCase'
        fldr_rslt = 'Paper_Results'
        file_rslt =  ''
        write_f = False
        plot    = False
        
    if op_mode == 2:
        zfs   = [50,]
        # fzvs  = np.arange(0.812,0.851,0.002)
        fzvs  = np.arange(0.770,0.810,0.002)
        Cgs   = [2.]
        Dsgns = ['A']
        Pels  = [5,4,3,2]
        
        fldr_rslt = 'BDR_OptimCase'
        file_rslt =  fldr_rslt+'/1-Pvar_final.txt'
        write_f = True
        plot    = False
    
    if op_mode == 3:
        zfs   = [50]
        fzvs  = np.arange(0.75,0.85,0.02)
        Cgs   = [3.0,4.0]
        Dsgns = ['A']
        Pels  = [10]
        plot  = True
        
        fldr_rslt = 'BDR_OptimCase'
        file_rslt =  fldr_rslt+'/1-CPC_final.txt'
        write_f = True
        
    ############################################
    fldr_dat  = 'Datasets_Final'
    txt_header  = 'Pel\tzf\tfzv\tCg\tDsgn\teta_hbi\teta_cos\teta_blk\teta_att\teta_hbi\teta_cpi\teta_cpr\teta_CPC\teta_BDR\teta_SF\tQ_acc\tN_hel\tS_hel\tS_HB\tS_CPC\tH_CPC\trO\tQ_max\tstatus\n'
    # if write_f:   f = open(file_rslt,'w'); f.write(txt_header); f.close()
    
    ########### Running the loop ###############
    for (zf,fzv,Cg,Dsgn_CPC,Pel) in [(zf,fzv,Cg,Dsgn_CPC,Pel) for zf in zfs for fzv in fzvs for Cg in Cgs for Dsgn_CPC in Dsgns for Pel in Pels]:
        
        case = 'zf_{:d}_fzv_{:.3f}_Cg_{:.1f}_Dsgn_{}_Pel_{:.1f}'.format(zf,fzv,Cg,Dsgn_CPC,Pel)
        print(case)

        #Defining the conditions for the plant
        CST = BDR.CST_BaseCase(zf=zf,fzv=fzv,P_el=Pel)
        
        #Files for initial data set and HB intersections dataset
        file_SF = fldr_dat+'/Rays_Data_height_{:.0f}'.format(zf)
        file_HB = fldr_dat+'/Basecase_zf{:.0f}_fzv{:.3f}_R1'.format(CST['zf'],CST['fzv'])
        
        #Getting the receiver area and the characteristics for CPC
        A_rcv_rq = CST['P_th'] / CST['Q_av']      #Area receiver
        rO = fsolve(A_rqrd, 1.0, args=(A_rcv_rq,Dsgn_CPC,CST['xrc'],CST['yrc'],CST['zrc'],Cg))[0]
        CPC = BDR.CPC_Params( Dsgn_CPC, CST['xrc'],CST['yrc'],CST['zrc'], {'rO':rO,'Cg':Cg} )
        
    
    ### Calling the optimisation function
        ######################################################################
        ######################################################################
        
        R2, Etas, SF, CPC, HB, hlst, Q_rcv, status = BDR.Optimisation(CST,CPC,file_SF,file_HB)
        
        ######################################################################
        ######################################################################
        
        #Some postcalculations
        S_CPC,H_CPC,rO = [ CPC[x] for x in ['S_CPC','H','rO'] ]
        N_hel = len(hlst)
        S_hel = N_hel * CST['A_h1']
        Pel_real = SF.loc[hlst]['Q_acc'].max() * (CST['eta_pb']*CST['eta_sg']*CST['eta_rc'])
        
        #Printing the result on file
        text_r = '\t'.join('{:.3f}'.format(x) for x in [Pel,zf, fzv, Cg])
        text_r = text_r + '\t'+Dsgn_CPC+'\t'+ '\t'.join('{:.4f}'.format(x) for x in [CST['eta_hbi'], Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_cpi'], Etas['Eta_cpr'], Etas['Eta_CPC'], Etas['Eta_BDR'], Etas['Eta_SF']])+'\t'
        text_r = text_r + '\t'.join('{:.2f}'.format(x) for x in [ Pel_real, N_hel, S_hel, HB['S_HB'], S_CPC,H_CPC,rO,Q_rcv.max()])+'\t'+status+'\n'
        print(text_r[:-2])
        
        if write_f:     f = open(file_rslt,'a');     f.write(text_r);    f.close()
        
        # R2.to_pickle('R2_basecase.plk')
        # np.save('Q_rcv',Q_rcv)        
    #####################################################################
    #####################################################################
    ############################ PLOTING ################################
    #####################################################################
    #####################################################################
        
        if plot:
            
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
            
            break
        
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
            
            # vmin = ((np.floor(10*R2f['Eta_SF'].min())-1)/10)
            # vmax = (np.ceil(10*R2f['Eta_SF'].max())/10)
            
            vmin = 0.0
            vmax = 1.0
            
            surf = ax1.scatter(R2f['xi'],R2f['yi'], s=0.5, c=R2f['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
            cb = fig.colorbar(surf, shrink=0.25, aspect=4)
            cb.ax.tick_params(labelsize=f_s)
            cb.ax.locator_params(nbins=4)
            
            fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']),fontsize=f_s)
            fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
            # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
            ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
            # ax1.set_title(r'Focal point height: {:.0f}m ($\eta_{{avg}}$={:.2f})'.format(zf,Eta_cos.mean()),fontsize=f_s)
            # ax1.set_title('No eta_cpi',fontsize=f_s)
            for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
            for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
            ax1.grid()
            fig.savefig(fldr_rslt+'/'+case+'_SF.png', bbox_inches='tight')
            # fig.savefig(fldr_rslt+'/'+case+'_SF.pdf', bbox_inches='tight')
            # plt.show()
            plt.close(fig)
            del R2f, out
            
        # break
        del R2, SF, Etas, CST, CPC, HB, hlst
        gc.collect()
        # break