# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:03:14 2020

@author: z5158936

"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
from scipy.misc import derivative
from scipy.interpolate import interp1d,RectBivariateSpline
from os.path import isfile
from multiprocessing import Pool
import gc
from functools import partial

#%% ##############  GENERAL FUNCTIONS ##############################
####################################################################

def CST_BaseCase(**kwargs):
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. Anyone can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    ############### STANDARD CASE ####################
    ##################################################
    # Environment conditions
    CST['Gbn'] = 950                # Design-point DNI [W/m2]
    ##################################################
    # Receiver and Power Block
    CST['P_el']   = 10.0               #[MW] Target for Net Electrical Power
    CST['eta_pb'] = 0.50               #[-] Power Block efficiency target 
    CST['eta_sg'] = 0.95               #[-] Storage efficiency target
    CST['eta_rc'] = 0.75               #[-] Receiver efficiency target
    ##################################################
    # Characteristics of Solar Field
    CST['eta_sfr'] = 0.97*0.95*0.95                # Solar field reflectivity
    CST['eta_rfl'] = 0.95                          # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    CST['A_h1']    = 7.07*7.07                     # Area of one heliostat
    CST['N_pan']   = 16                            # Number of panels per heliostat

    ##################################################
    # Characteristics of BDR and Tower
    CST['zf']       = 50.               # Focal point (where the rays are pointing originally) will be (0,0,zf)
    CST['fzv']      = 0.83              # Position of HB vertix (fraction of zf)
    CST['eta_hbi']  = 0.95              # Desired hbi efficiency
    
    CST['Dsgn_CPC'] = 'A'               # Number of CPCs and number of polygon vertex
    CST['xrc']      = 0.                # Second focal point (CPC receiver)
    CST['yrc']      = 0.                # Second focal point (CPC receiver)
    CST['fzc']      = 0.20              # Second focal point (Height of CPC Aperture, fraction of zf)
    
    CST['Q_av']     = 0.5               # [MW/m2] Desired average radiation flux on receiver
    CST['Q_mx']     = 2.0               # [MW/m2] Maximum radiation flux on receiver
    
    ##### CHANGING SPECIFIC VARIABLES ###########
    for key, value in kwargs.items():
        CST[key] = value
        
    ####### Variables from calculations #########
    if 'zrc' in CST:                            # Second focal point (CPC receiver)
        CST['fzc'] = CST['zrc']/ CST['zf']
    else:
        CST['zrc']  = CST['fzc']*CST['zf']
    
    if 'zv'  in CST:                            # Hyperboloid vertix height
        CST['fzv'] = CST['zv'] / CST['zf']
    else:
        CST['zv']   = CST['fzv']*CST['zf']
        
    if 'P_th' in CST:                           #[MW] Required power energy
        CST['P_el'] = CST['P_th'] * ( CST['eta_pb']*CST['eta_sg']*CST['eta_rc'] )
    else:
        CST['P_th'] = CST['P_el']/( CST['eta_pb']*CST['eta_sg']*CST['eta_rc'] )


    return CST

#%% ############## HYPERBOLOID SUBROUTINES #########################
####################################################################
####################################################################

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

################################################################

def HB_Shape(x,y,z,*params):
    """
    Function that describe the hyperboloid shape.
    This function allows to move the focus in the Y-axis, so it could be changed.

    Parameters
    ----------
    x,y,z : float. Position.
    *params : Parameters, should be, in the following order:
        zf (tower height),
        zv (vertex height),
        zrc, yrc (second focal point position)

    Returns
    -------
    F : The hyperboloid funcion that should be zero for the points belonging the HB

    """
    
    zf, zv, zrc, yrc = params
    zfh,zvh, zh = zf-zrc, zv-zrc, z-zrc
    yo, zo, c   = yrc/2, zfh/2, np.sqrt(zfh**2+yrc**2)/2
    xp, yp, zp  = x , y-yo , zh-zo
    a , b , t   = c*( 2*zvh/zfh - 1 ) , 2*c*np.sqrt(zvh/zfh - (zvh/zfh)**2), (yo - yrc) / (zo-0.)
    F  = ( yp*t + zp )**2/a**2 - ( xp**2 * ( 1+t**2 ) + ( yp-zp*t )**2 ) / b**2 - t**2 - 1
    return F

################################################################

def HB_Eval(z,*params):
    """
    For a given x,y return the value of z that belongs to the HB.
    """
    x,y,zf,zv,zrc,yrc = params
    Z = HB_Shape(x,y,z,zf,zv,zrc,yrc)
    return Z

################################################################

def HB_Kintersect(k,*params):
    """
    Function to get the value of k that intercept a ray with HB.
    """
    zf,zv,zrc,yrc,xi,yi,zi,ux,uy,uz = params
    x, y, z = xi + k*ux , yi + k*uy , zi + k*uz
    K = HB_Shape(x,y,z,zf,zv,zrc,yrc)
    return K

################################################################

def HB_Surface_dif(xb,yb,dx,dy,*param):
    """
    For a given point xb and yb, obtain the differential surface
    """
    zf,zv,zrc,yrc = param
    zb    = fsolve(HB_Eval, 0.75*zf, args=(xb,yb)+(zf,zv,zrc,yrc))[0]
    dfdx,dfdy,dfdz = [partial_derivative(HB_Shape,i,point=[xb,yb,zb,zf,zv,zrc,yrc]) for i in [0,1,2]]
    dS = np.sqrt( dfdx**2 + dfdy**2 + dfdz**2 ) / abs(dfdz) *dx*dy
    return dS

################################################################

def HB_ReceiverPoint(xi,yi,zi,z_hint,uxi,uyi,uzi,zf,zv,zrc,yrc):
    """
    For a ray from the heliostat field, obtain the interception with HB.

    Parameters
    ----------
    xi,yi,zi        : Position of the ray in the solar field.
    z_hint          : Hint of the height solution
    uxi, uyi, uzi   : Cosine directions of the ray
    zf, zv, zrc, yrc: Parameters of the Hyperboloid

    Returns
    -------
    xb,yb,zb        : Position of the intersection with HB.
    xc, yc, zc      : Position of the ray in the CPC aperture plane.
    uxc, uyc, uzc   : Direction of the reflected ray

    """
    khint         = 0.99*(zf-zrc)/uzi
    kb            = fsolve(HB_Kintersect, khint, args=(zf,zv,zrc,yrc,xi,yi,zi,uxi,uyi,uzi))[0]
    xb, yb, zb    = xi+kb*uxi, yi+kb*uyi, zi+kb*uzi
    ddx, ddy, ddz = [partial_derivative(HB_Shape,i,point=[xb,yb,zb,zf,zv,zrc,yrc]) for i in [0,1,2]]
    nn            = (ddx**2+ddy**2+ddz**2)**0.5
    nx,ny,nz      = ddx/nn, ddy/nn, ddz/nn
    sc            = nx*uxi + ny*uyi + nz*uzi
    uxc, uyc, uzc = uxi-2*sc*nx, uyi-2*sc*ny, uzi-2*sc*nz
    kc            = (zrc-zb)/uzc
    xc, yc, zc    = xb+kc*uxc, yb+kc*uyc, zb+kc*uzc
    return xb,yb,zb,xc,yc,zc,uxc,uyc,uzc


################################################################

def HB_Surface_Cal(R1,CST):
    """
    

    Parameters
    ----------
    R1 : TYPE
        DESCRIPTION.
    CST : TYPE
        DESCRIPTION.

    Returns
    -------
    HB_Surface : TYPE
        DESCRIPTION.
    S_HB : TYPE
        DESCRIPTION.
    rlims : TYPE
        DESCRIPTION.
    hit_hb : TYPE
        DESCRIPTION.

    """
    
    zf,zv,zrc,yrc,eta_hbi = [CST[x] for x in ['zf','zv','zrc','yrc','eta_hbi']]
    N              = 100
    Nx,Ny,Nz       = N, N, N
    
    rays           = R1[R1['hel_in']]       #Calculating area only with selected heliostats
    
    if 'rb' not in rays:    rays['rb']  = np.sqrt( rays['xb']**2 + rays['yb']**2 )
    rmin, rmax     = rays['rb'].quantile(0.001), rays['rb'].quantile(eta_hbi+0.001)
    zmin, zmax     = rays['zb'].quantile(0.0001), rays['zb'].quantile(eta_hbi+0.0001)
    
    
    ymins = []; ymaxs = [];
    zdiv = np.linspace(zmin,zmax,Nz+1)
    for i in range(len(zdiv)-1):
        ymins.append(rays[(rays['zb']>zdiv[i]) & (rays['zb']<zdiv[i+1])]['yb'].quantile(0.001))
        ymaxs.append(rays[(rays['zb']>zdiv[i]) & (rays['zb']<zdiv[i+1])]['yb'].quantile(0.999))
    ymins.append(ymins[-1])
    fymin = interp1d(zdiv,ymins,kind='linear',fill_value='extrapolate')
    
    #Creation the hyperboloid sheet
    xx  = np.linspace(-rmax, rmax, Nx); yy  = np.linspace(-rmax, rmax, Ny)
    dx = xx[1]-xx[0]; dy = yy[1]-yy[0]
    zhint=zf
    
    Z = []; S = []
    for x in xx:
        zz = []; ss = []
        for y in yy:
            z = fsolve(HB_Eval, zhint, args=(x,y)+(zf,zv,zrc,yrc))[0]
            s = HB_Surface_dif(x,y,dx,dy,zf,zv,zrc,yrc)
            
            #Limits on radius and y-extension
            if (x**2+y**2)**0.5<rmin:     z,s = zmin, 0.
            if (x**2+y**2)**0.5>rmax:     z,s = zmax, 0.
            if z!=z:                      z,s = zmax, 0.
            if z>zmax:                    z,s = zmax, 0.
            elif y<fymin(z):              z,s = z, 0.
            
            zz.append(z);   ss.append(s)
        Z.append(np.array(zz));     S.append(np.array(ss))
    Z = np.array(Z);    S = np.array(S)

    S_HB   = S.sum()
    rlims  = rmin,rmax
    
    hit_hb = (RectBivariateSpline(xx,yy,S).ev(rays['xb'],rays['yb']) > 0) & (rays['rb'] < rmax)
    hit_hb = hit_hb & R1['hel_in']
    HB_Surface = xx,yy,Z,S
    # rays.drop('rb',axis=1,inplace=True)
    return HB_Surface, S_HB, rlims, hit_hb

################################################################

def HB_Surface_Int(rays,HB_Surface,rlims):
    
    xx,yy,Z,S = HB_Surface
    if 'rb' not in rays:    rays['rb']  = np.sqrt( rays['xb']**2 + rays['yb']**2 )
    # hit_hb = (RectBivariateSpline(xx,yy,S).ev(rays['xb'],rays['yb']) > 0) & (rays['rb'] < rlims[1]) & (rays['hel_in'])
    hit_hb = (RectBivariateSpline(xx,yy,S).ev(rays['xb'],rays['yb']) > 0) & (rays['rb'] < rlims[1])
    
    return hit_hb
################################################################

def HB_worker(R,params):
    xi,yi,zi,uxi,uyi,uzi,hel = R
    zf,zv,zrc,yrc = params
    zhint  = (zf+zv-zrc*2)/2
    nh     = int(hel)
    xb,yb,zb,xc,yc,zc,uxb,uyb,uzb = HB_ReceiverPoint(xi,yi,zi,zhint,uxi,uyi,uzi,zf,zv,zrc,yrc)

    return [nh,xi,yi,zi,xb,yb,zb,xc,yc,zc,uxi,uyi,uzi,uxb,uyb,uzb]

#%% ############# CPC SUBROUTINES #####################################
#######################################################################
#######################################################################
def CPC_Design(Dsgn):
    """
    From a Design label ('A' to 'E'), return the number of CPC and the number of sides.

    """
    if   Dsgn == 'A':       N_CPC = 3 ; V_CPC = 6       # 3 hexagons with centered vertix
    elif Dsgn == 'B':       N_CPC = 7 ; V_CPC = 6       # 7 hexagons, with one centered
    elif Dsgn == 'C':       N_CPC = 4 ; V_CPC = 4       # 4 squares, with centered vertix
    elif Dsgn == 'D':       N_CPC = 4 ; V_CPC = 4       # 4 squares, with two sharing center and two in shorter side
    elif Dsgn == 'E':       N_CPC = 1 ; V_CPC = 8       # 1 octagon centered
    else:                   N_CPC = 0 ; V_CPC = 0       # Wrong Design label
    return N_CPC, V_CPC

####################################################

def CPC_Centers(Dsgn,rA,xrc,yrc):
    """
    From a design, a radius and a CPC position, it returns the centers of all polygon CPCs

    Parameters
    ----------
    Dsgn : label (str)
        CPC Design
    rA : float
        CPC radius.
    xrc, yrc : floats
        Center position (second focal point)

    Returns
    -------
    x0 : list
        x values for center positions.
    y0 : list
        y values for center positions.

    """
    
    V_CPC = CPC_Design(Dsgn)[1]
    phi   = np.radians(360/V_CPC)
    
    if   Dsgn == 'A':   x0, y0 = [(2*rA/3**0.5)*np.cos(2*n*phi)+xrc for n in range(3)], [(2*rA/3**0.5)*np.sin(2*n*phi)+yrc for n in range(3)]
    elif Dsgn == 'B':   x0, y0 = [xrc]+[(2*rA)*np.sin(phi*n)+xrc for n in range(6)], [yrc]+[(2*rA)*np.cos(phi*n)+yrc for n in range(6)]
    elif Dsgn == 'C':   x0, y0 = [(2*rA/2**0.5)*np.cos(n*phi)+xrc for n in range(4)], [(2*rA/2**0.5)*np.sin(n*phi)+yrc for n in range(4)]
    elif Dsgn == 'D':   x0, y0 = [rA/2**0.5, -rA/2**0.5, 2**0.5*rA,-2**0.5*rA ], [-rA/2**0.5, rA/2**0.5, 2**0.5*rA,-2**0.5*rA]
    elif Dsgn == 'E':   x0, y0 = [0.],[0.]
    else:               x0, y0 = [0.],[0.]
    
    return x0, y0

####################################################

def CPC_Z( x, y, V, xo, yo ):
    """
    Function that return the z position in CPC for a given (x,y) position

    Parameters
    ----------
    x,y : floats, lists, numpy arrays
        Position(s) where the correspondent z value is needed.
    V : integer
        Number of sides of polygon CPC
    xo,yo : floats
        Center of the CPC.

    Returns
    -------
    z : floats, lists, numpy arrays
        Position(s) of CPC array, considering zo=0.

    """
    xp, yp = x-xo, y-yo
    phi    = 360./V
    alpha  = np.degrees(np.arctan2(yp,xp)) % 360
    i      = np.floor(alpha/phi)

    m      = np.tan(np.radians((i+0.5)*phi + 90.))
    n      = yp - m*xp

    phi1, phi2 = np.radians((i+1)*phi), np.radians(i*phi)
    xi = (-n/2) * ( 1/(m-np.tan(phi1)) + 1/(m-np.tan(phi2)) )
    yi = m*xi+n
    
    z  = (xi**2+yi**2)
    return z

def CPC_Kintersect(k,*params):
    xc,yc,zc,ux,uy,uz,V,xo,yo = params
    x, y, z = xc + k*ux , yc + k*uy, zc + k*uz
    K = z - CPC_Z(x,y,V,xo,yo)
    return K

def CPC_Derivatives(xi,yi,rA,rO,V,xo,yo):
    dx  = 1e-6
    ddx = ( CPC_Z(xi+dx/2,yi,rA,V,xo,yo) - CPC_Z(xi-dx/2,yi,rA,V,xo,yo) ) / dx
    ddy = ( CPC_Z(xi,yi+dx/2,rA,V,xo,yo) - CPC_Z(xi,yi-dx/2,rA,V,xo,yo) ) / dx
    ddz = -1
    return ddx,ddy,ddz

def CPC_XY_R(ri,H,V,N,xo,yo,zo):
    """
    For a CPC with given parameters, obtain the points to form the CPC curve.

    Parameters
    ----------
    ri : float
        Radius of CPC transverse area (could be rA, rO, etc).
    H : float
        CPC height.
    V : integer
        Number of sides of polygon CPC.
    N : integer
        Number of CPC in the array.
    xo, yo, zo : floats
        Center of polygon CPC.

    Returns
    -------
    xx : list
        Set of x points that form the CPC aperture shape.
    yy : list
        Set of y points that form the CPC aperture shape.

    """
    
    Np     = 100        #Number of points
    phi    = 360./V
    xmax   = ri/np.cos(np.radians(phi/2))
    xmin   = ri/np.cos(np.radians((np.floor(180./phi)+0.5)*phi))
    if xmin<-xmax:   xmin=-xmax         #To avoid nan values on arccos
    
    xx     = np.linspace(xmax,xmin,Np)
    angs   = np.degrees(np.arccos( xx/xmax ) )
    ii     = np.floor(angs/phi);
    
    phii   = np.radians((ii+0.5)*phi)
    xi, yi = ri*np.cos(phii), ri*np.sin(phii)
    mm     = np.tan(phii + np.pi/2)
    nn     = (yi-mm*xi)
    yy     = mm*xx + nn
    
    xx = np.append(xx,np.flip(xx)) + xo
    yy = np.append(yy,-np.flip(yy)) + yo
    
    return xx,yy

###############################################
def CPC_Params( Dsgn, xrc, yrc, zrc, CPC):
    """
    Function that calculates the main geometric parameters for CPC array.

    Parameters
    ----------
    Dsgn            : Type of CPC array (see CPC_Dsgn function)
    xrc, yrc, zrc   : Position of aperture plane (second focal point)
    CPC             : Must be at least two of these combinations on a dict: rA, rO, H, Cg

    Returns CPC (dict) with the following keys
    -------
    rA    : Aperture radius
    rO    : receiver (output) radius
    H     : CPC height
    Cg    : Concentration Ratio
    S_CPC : Surface of CPC Array [m2].
    A_CPC : Aperture CPC Array Area [m2].
    A_rcv : Outlet CPC Array Area (equal to receiver area) [m2].
    rBDR  : Radius of CPC Array [m]
    theta : CPC concentration angle.
    """

    #Depending the parameters received, the others are calculated
    if 'rA' in CPC and 'rO' in CPC:
        rA,rO = CPC['rA'], CPC['rO']
        H     = rA**2 - rO**2
        Cg    = (rA/rO)**2                      #Concentration ratio of each CPC
        
    elif 'rA' in CPC and 'H' in CPC:
        rA,H  = CPC['rA'], CPC['H']
        rO    = (rA**2 - H)**0.5               # Check if it is over limits
        if rO<0:
            print("Height is too much for rA, will be replaced by height for min rO")
            rO = 0.2
            H  = rA**2-rO**2
        Cg = (rA/rO)**2                      #Concentration ratio of each CPC
    
    elif 'rA' in CPC and 'Cg' in CPC:
        rA,Cg = CPC['rA'], CPC['Cg']
        rO    = rA/Cg**0.5
        H     = rA**2 - rO**2
    
    elif 'rO' in CPC and 'H' in CPC:
        rO,H  = CPC['rO'], CPC['H']
        rA    = H + rO**2
        Cg    = (rA/rO)**2                      #Concentration ratio of each CPC
    
    elif 'rO' in CPC and 'Cg' in CPC:
        rO,Cg = CPC['rO'], CPC['Cg']
        rA    = rO*Cg**0.5
        H     = rA**2-rO**2
    
    else:
        raise ValueError("Wrong input parameters")
        return
    
    ## Getting the parameters
    N, V  = CPC_Design(Dsgn)
    phi   = np.radians(360./V)
    S1    = V*np.tan(phi/2)/6 * ( ( 1 + 4*rA**2 )**(3/2) - ( 1 + 4*rO**2)**(3/2))     #Surface area of 1 CPC
    frCPC = 1 if N==1 else 2 if N==3 else 8**0.5 if N==4 else 3    #Only working for N_CPC in 1,3,4,7
    
    ## Creating the new dictionary
    CPC = {'Dsgn'  :  Dsgn,
           'N'     :  N,
           'V'     :  V,
           'rA'    :  rA,
           'rO'    :  rO,
           'H'     :  H,
           'Cg'    :  Cg,
           'S_CPC' :  N*S1 ,
           'A_CPC' :  V * rA**2 * np.tan(phi/2) * N,
           'A_rcv' :  V * rO**2 * np.tan(phi/2) * N,
           'rBDR'  :  frCPC * rA,
           'theta' :  np.degrees(np.arccos(H**0.5/rA))
           }
    
    return CPC

#################################################
def CPC_worker_outdist(ray,params):
    """
    function that calculate the final distribution in the receiver.
    It solves the intersectio from reflected rays in the CPC array. If the ray reflected on the CPC does not reach the final receiver, it calculates a new reflection on the CPC, and so on until the receiver is reached or the ray leaves the CPC array.

    Parameters
    ----------
    ray : list
        A specific row from R1 dataset.
    params : list
        Parameters of CPC array: N_CPC,V_CPC,rA,rO,centres.

    Returns
    -------
    It returns a list with a specific row for a ray on R2 dataframe. The values are:
    hel, xi,yi,zi, xb,yb,zb, xc,yc,zc, xs,ys,zs, xr,yr,zr, uxi,uyi,uzi, uxb,uyb,uzb, uxr,uyr,uzr, hel_in,hit_hb,hit_cpc,hit_rcv,Nr_cpc
    """
    
    N_CPC,V_CPC,rA,rO,centres = params
    hel,xi,yi,zi,xb,yb,zb,xc,yc,zc,uxi,uyi,uzi,uxb,uyb,uzb,hel_in,hit_hb,hit_cpc = ray
    Nrflmax = 20
    
    x0,y0 = centres
    Nr_cpc = 0                    #Number of internal reflections
    hit_rcv = False               #True if it hit the outlet surface, otherwise False
    
    if hit_cpc==0:
        xs,ys,zs,xr,yr,zr,uxr,uyr,uzr,kr,zO = [np.nan for i in range(11)]
        hit_rcv = False
    else:
        zA,zO = rA**2, rO**2  # Aperture plane for CPC
        xo,yo = x0[hit_cpc-1],y0[hit_cpc-1]
        xn,yn,zn,uxn,uyn,uzn = xc, yc, zA, uxb, uyb, uzb
        
        while not(hit_rcv):
            
            #Finding the reflection point
            try:
                ks = brentq(CPC_Kintersect, 0.01, 100, args=(xn,yn,zn,uxn,uyn,uzn,V_CPC,xo,yo))
            except:
                khint         = (zO/uzn)
                ks = fsolve(CPC_Kintersect, khint, args=(xn,yn,zn,uxn,uyn,uzn,V_CPC,xo,yo))[0]
            xs, ys, zs  = xn+ks*uxn, yn+ks*uyn, zn+ks*uzn
            
            #The ray is going out
            if zs>zA:
                xr,yr,zr,uxr,uyr,uzr,kr,zO = [np.nan for i in range(8)]
                hit_rcv = False
                break
            #The ray reaches the OUTLET
            elif zs<zO:
                kr            = (zO - zs)/uzn
                xr, yr, zr    = xs+kr*uxn,  ys+kr*uyn,  zs+kr*uzn
                uxr,uyr,uzr   = uxn,uyn,uzn
                zout  = CPC_Z(xr,yr, V_CPC, xo, yo)
                hit_rcv = True
            #We get the reflected ray direction
            else:
                Nr_cpc += 1
                dx  = 1e-6          #Partial derivatives
                ddx = ( CPC_Z(xs+dx,ys,V_CPC,xo,yo) - CPC_Z(xs-dx,ys,V_CPC,xo,yo) ) / (2*dx)
                ddy = ( CPC_Z(xs,ys+dx,V_CPC,xo,yo) - CPC_Z(xs,ys-dx,V_CPC,xo,yo) ) / (2*dx)
                ddz = -1
                nn            = (ddx**2+ddy**2+ddz**2)**0.5
                nx,ny,nz      = ddx/nn, ddy/nn, ddz/nn
                sc            = nx*uxn + ny*uyn + nz*uzn
                uxr, uyr, uzr = uxn-2*sc*nx, uyn-2*sc*ny, uzn-2*sc*nz
                kr            = (zO - zs)/uzr
                xr, yr, zr    = xs+kr*uxr,  ys+kr*uyr,  zs+kr*uzr
                zout = CPC_Z(xr,yr, V_CPC, xo, yo)
                
                if   zout<zO:       #The reflected ray gets the output
                    zr = zO;    hit_rcv = True
                elif zout>zA:
                    xr,yr,zr,uxr,uyr,uzr,kr,zO = [np.nan for i in range(8)]
                    hit_rcv = False
                    break
                else:           #we need a new calculation
                    xn,yn,zn,uxn,uyn,uzn = xs,ys,zs,uxr,uyr,uzr
            
            if Nr_cpc == Nrflmax:
                break
        
    return hel, xi,yi,zi, xb,yb,zb, xc,yc,zc, xs,ys,zs, xr,yr,zr, uxi,uyi,uzi, uxb,uyb,uzb, uxr,uyr,uzr, hel_in,hit_hb,hit_cpc,hit_rcv,Nr_cpc



#######################################################################
# %% ############# SUBROUTINES OTHER GEOMETRIES #######################
#######################################################################
def CPC_worker_outdist(R2,Rcvr):
    """
    function that calculate the distribution in a cylindrical final receiver.
    
    It calculates the intersection from rays entering the receiver and its inner surface. The interceptions are calculated directly with quadratic equation.

    Parameters
    ----------
    R2 : DataFrame
        A dataframe with R2 columns
    params : list
        Parameters of CPC array: N_CPC,V_CPC,rA,rO,centres.

    Returns
    -------
    It returns ...
    
    """
    
    rA = Rcvr['rA']
    rM = Rcvr['rM']
    H  = Rcvr['H']
    
    xr,yr,zr, uxr,uyr,uzr = R2[['xr','yr','zr','uxr','uyr','uzr']]
    rr = (xr**2+yr**2)**0.5
    
    a = uxr**2 + uyr**2
    b = 2 * (xr*uxr + yr*uyr)
    c = rr**2 - rM**2
    
    km = -b + (b**2 - 4*a*c)**0.5 / (2*a)
    
    zm = zr+km*uzr
    
    zf = zm if zm>0 else 0
    km = km if zf>0 else -zr/uzr
    
    xf, yf = xr+km*uxr, yr+km*uyr

    hit_rcv = 1 if zf>0 else 0    #(-1 outside, 0 bottom, 1 mantle)
    
    return xf, yf, zf, hit_rcv

#######################################################################
# %% ########### SUBROUTINES FOR COUPLED SYSTEM #######################
#######################################################################

def Eta_attenuation(R1):
    """
    Function to obtain the attenuation efficiency.
    It requires the total distance from heliostats to CPC.
    If the distance from HB to CPC is unknown (xc,yc,zc not calculated yet) it is assumed zero.

    Parameters
    ----------
    R1 : pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    Eta_att : pandas Series
        Pandas series with Eta_att grouped by heliostat ID.

    """
    
    # Distance [km] from heliostats to HB (d1) and from HB to CPC (d2)
    d1  = ((R1['xi']-R1['xb'])**2+(R1['yi']-R1['yb'])**2+(R1['zi']-R1['zb'])**2)**0.5
    d2  = ((R1['xb']-R1['xc'])**2+(R1['yb']-R1['yc'])**2+(R1['zb']-R1['zc'])**2)**0.5 if 'xc' in R1 else 0.
    R1['dray'] = d1 + d2
    d = R1.groupby('hel').mean()['dray']/1e3
    Eta_att = 1 - (0.006789 + 0.1046*d - 0.017*d**2 + 0.002845*d**3)
    return Eta_att

####################################################################
def Rays_Dataset(file_SF, **kwargs):
    """
    Function that read a file from SolarPilot datasets and convert it into R0 Dataframe

    Parameters
    ----------
    file_SF : string
        name of the SolarPilot file with the dataSet.
    **kwargs : various types
        Parameters that can change the calculations.
            N_pan:          Number of panels per heliostat, default 16
            convert:        if True convert the .csv file into .plk. Default False.
            N_hel:          Number of heliostat selected. If not given, use all of them

    Returns
    -------
    R0 : pandas DataFrame
        Initial Dataframe containing the ray dataset.
        The cols are: ['x','y','z','ux','uy','uz','hel']
    Eta_blk : pandas Series
        Contain the blocking efficiency per heliostat.
    Eta_cos : pandas Series
        Contain the cosine efficiency per heliostat.
    """
    
    #kwargs can be N_pan, N_hel, convert
    N_pan = kwargs['N_pan'] if 'N_pan' in kwargs else 16
    convert = kwargs['convert'] if 'convert' in kwargs else False
    
    if isfile(file_SF+'.pkl'):
        rays = pd.read_pickle(file_SF+'.pkl')
    else:
        rays = pd.read_csv(file_SF+'.csv', header=0, names=['x','y','z','ux','uy','uz','ele','stg','rayN'])
        if convert: rays.to_pickle(file_SF+'.pkl')
    
    r_sf = rays[rays['stg']==1][['x','y','z','ele','rayN']]          #All rays that hit the solar field
    r_sf['hel']  = np.sign(r_sf['ele'])*((np.abs(r_sf['ele'])-1)//N_pan + 1)
    r_sf  = r_sf[r_sf['ele']>0]                                       # All rays that leave the solar field
    r_tw  = rays[rays['stg']==2][['x','y','z','ux','uy','uz']]        # All rays that go to tower
    
    N_hel = kwargs['N_hel'] if 'N_hel' in kwargs else r_sf['hel'].max()
    
    R0      = pd.merge(r_tw, r_sf, how='inner', on=['x','y','z']).sort_values('hel')
    R0      = R0[R0['hel']<=N_hel][['x','y','z','ux','uy','uz','hel']]            #DF used to HB rays interceptions
    
    Eta_blk = R0.groupby('hel').count()['x'] / r_sf.groupby('hel').count()['x']
    
    #Angle between normal to heliostat and sun
    sun  = -rays[rays['stg']==1][['ux','uy','uz']].mean()            # Sun position (for Cos efficiency)
    av = R0.groupby('hel').mean()[['x','y','z','ux','uy','uz']]
    Eta_cos = np.sqrt( (av['ux']+sun[0])**2 + (av['uy']+sun[1])**2 + (av['uz']+sun[2])**2 )
    Eta_cos = ( (av['ux']+sun[0])*sun[0] + (av['uy']+sun[1])*sun[1] + (av['uz']+sun[2])*sun[2] ) / Eta_cos
    Eta_blk.name = 'Eta_blk';   Eta_cos.name = 'Eta_cos'
    
    return R0, Eta_blk, Eta_cos

####################################################################
def HB_Intercepts(R0,CST,**kwargs):
    """
    Function that receive a R0 DataFrame and a CST dict.
    It calculates the interception with HB and return a new R1 rays datasets.

    Parameters
    ----------
    R0 : pandas DataFrame
        DataFrame containing the ray dataset from the heliostat field with the correct format.
    CST : dict
        CST plant with parameters defined previously.
    **kwargs : various types
        Different parameters that can change the outputs.
            N_proc:     Number of processors used in calculations. Default is 4.
            file_HB:    If given, it look for that file that should already contain the R1
            print_out:  If True, print out the R1 interception as .plk file with name file_HB.plk. Default False

    Returns
    -------
    R1 : pandas DataFrame
        DataFrame with the interceptions on HB. The cols are:
        ['hel','xi','yi','zi', 'xb','yb','zb', 'xc','yc','zc', 'uxi','uyi','uzi', 'uxb','uyb','uzb', 'hel_in', 'hit_hb', 'hit_cpc']

    """
    
    N_proc  = kwargs['N_proc'] if 'N_proc' in kwargs else 4
    file_HB = kwargs['file_HB'] if 'file_HB' in kwargs else ''
    print_out = kwargs['print_out'] if 'print_out' in kwargs else False
    
    zf,zv,zrc,yrc = [CST[x] for x in ['zf','zv','zrc','yrc']]
    
    if isfile(file_HB+'.pkl'):
        # print("Reading file with positions on HB")
        R1 = pd.read_pickle(file_HB+'.pkl')
    else:
        # print("Reading the file from SolarPilot and Calculating positions on HB. This is the most time consuming part")
        pool  = Pool( processes = N_proc )
        R1    = pool.map(partial(HB_worker, params=[zf,zv,zrc,yrc]), R0.values.tolist() )
        R1    = pd.DataFrame(R1, columns=['hel', 'xi', 'yi', 'zi', 'xb', 'yb', 'zb', 'xc', 'yc', 'zc', 'uxi', 'uyi', 'uzi', 'uxb', 'uyb', 'uzb'])
        R1.index.names = ['ray']; R1.columns.names = ['rays']
        pool.close()
        
        if print_out:   R1.to_pickle(file_HB+'.pkl')                   #Saving the file
    return R1
        


###################################################################
###################################################################

def Optimisation(CST,CPC,file_SF,file_HB):
    """
    Description
    The function receive CST parameters and CPC characteristics. 
    Returns the rays final positions in receiver surface, the efficiencies per heliostat, the average efficiencies, the list of selected heliostats, the HB and CPC characteristics, the receiver radiation map, and the convergence status.
    
    Parameters
    ----------
    CST : Dict.
        Characteristics of CST plant.
    CPC : Dict.
        Design of CPC array.
    file_SF : string.
        Name of file with SolarPilot's dataset
    file_HB : string.
        Name of file with HB interesections. If the file does not exist,
        the interceptions are calculated and the file is created.

    Returns
    -------
    R2    : pandas DataFrame with final rays positions.
    Etas  : pandas Series with Average efficiencies.
    SF    : pandas DataFrame with Detailed efficiencies per heliostat.
    CPC   : dict with CPC characteristics
    HB    : dict with HB characteristics.
    hlst  : list with selected heliostats.
    Q_rcv : pandas DataFrame with receiver radiation map.
    stats : string, 'OK' if the solution converged, otherwise an error label.
    """
    
    # R0_cols = ['x','y','z','ux', 'uy','uz','hel','hel_in']
    R1_cols = ['hel','xi','yi','zi', 'xb','yb','zb', 'xc','yc','zc', 'uxi','uyi','uzi', 'uxb','uyb','uzb', 'hel_in', 'hit_hb', 'hit_cpc']
    R2_cols = ['hel','xi','yi','zi', 'xb','yb','zb', 'xc','yc','zc', 'xs','ys','zs', 'xr','yr','zr', 'uxi','uyi','uzi', 'uxb','uyb','uzb', 'uxr','uyr','uzr', 'hel_in','hit_hb','hit_cpc','hit_rcv','Nr_cpc']
    
    #####################################################
    
    xrc, yrc, zrc, zf, zv, fzv, eta_hbi = [ CST[x] for x in ['xrc','yrc','zrc','zf','zv','fzv','eta_hbi'] ]
    Dsgn_CPC,N_CPC,V_CPC,H_CPC,rA,rO,A_rcv = [ CPC[x] for x in ['Dsgn','N','V','H','rA','rO','A_rcv'] ]
    x0,y0 = CPC_Centers(Dsgn_CPC,rA,xrc,yrc)
    
    #####################################################
    # READING THE FILE WITH HELIOSTAT LAYOUT, WHICH DEPENDS ON HEIGHT
    R0, Eta_blk, Eta_cos = Rays_Dataset( file_SF, convert=False )
    N_max   = len(R0['hel'].unique())
    hlst    = R0['hel'].unique()
    # R0 = pd.merge( R0 , Eta_cos , how='inner', on=['hel'] )
    R1      = HB_Intercepts(R0,CST,file_HB=file_HB,print_out=True)
    
    #########################################################
    #Getting the intercepts with CPC surface
    nRA=[]
    zA = rA**2     # Aperture and output height for CPC coordinate system
    xc,yc  = R1['xc'], R1['yc']
    for i in range(N_CPC):
        zz = CPC_Z( xc, yc, V_CPC, x0[i],y0[i])         #Checking on aperture plane
        nRA.append((zz<zA)*(i+1))
    R1['hit_cpc'] = np.array(nRA).max(axis=0)

    #Re-setting the values for R1 hel_in and hit_hb
    R1['hel_in'] = True         #Considering all heliostat at beginning
    R1['hit_hb'] = True         #Considering all rays hitting the HB
    
    N_proc = 4
    pool    = Pool( processes = N_proc )
    R2aux   = pool.map(partial(CPC_worker_outdist, params=[N_CPC,V_CPC,rA,rO,[x0,y0]]), R1[R1_cols].values.tolist() )
    R2      = pd.DataFrame( R2aux, columns=R2_cols )
    pool.close()
    del R2aux
    
    #########################################################
    #Calculating Efficiencies that does not depend on HB (CPC efficiencies)
    Eta_hbr = CST['eta_rfl'] * np.ones(N_max)
    Eta_cpi = R2[(R2['hit_cpc']>0)].groupby('hel').count()['xb'] / R2.groupby('hel').count()['xb']
    N_avg   = R2[(R2['hit_rcv'])&(R2['hit_cpc']>0)].groupby('hel').mean()['Nr_cpc']
    Eta_cpr = (0.95**N_avg) * R2[(R2['hit_rcv'])&(R2['hit_cpc']>0)].groupby('hel').count()['xb'] / R2[(R2['hit_cpc']>0)].groupby('hel').count()['xb']
    
    R1['hel_in'] = True                 #Considering all heliostat at beginning
    R1['hit_hb'] = True                 #Considering all rays hitting the HB
    hlst         = R1['hel'].unique()   #Considering all heliostat at beginning
    
    Nit = 1; N_ant = N_max; N_an2 = 0
    while True:             #loop to make converge the number of heliostats and hyperboloid size
        
        #########################################################
        #Calculating the surface size with selected heliostats
        HB_Surface, S_HB, rlims, hit_hb_ax = HB_Surface_Cal(R1,CST)
    
        #########################################################
        #Calculating the eta_hbi for all heliostats with given surface
        hit_hb = HB_Surface_Int(R1,HB_Surface,rlims)
        R1['hit_hb']   = hit_hb
        Eta_hbi = (R1.groupby('hel').sum()['hit_hb'] / R1.groupby('hel').count()['xb'])
        
        #########################################################
        #Getting the values for efficiencies and radiation fluxes
        Eta_att = Eta_attenuation(R1)
        Eta_hel = Eta_blk * Eta_cos * Eta_att * CST['eta_rfl']
        Eta_CPC = Eta_cpi * Eta_cpr
        Eta_BDR = Eta_hbi * Eta_hbr * Eta_CPC
        Eta_SF  = Eta_hel * Eta_BDR
        Q_h1    = (Eta_SF * CST['Gbn'] * CST['A_h1'] * 1e-6).sort_values(ascending=False)
        Q_acc   = Q_h1.cumsum()
        Q_avg   = Q_acc / A_rcv
        
        #Getting the number of heliostats required and the list of heliostats
        N_hel   = len( Q_acc[ Q_acc < CST['P_th'] ] ) + 1
        suav    = 0.8
        
        N_hel   = int( suav*N_ant + (1-suav)* N_hel )    #Attenuation factor
        
        if N_an2==N_hel:  N_hel = int((N_hel+N_ant)/2)    #In case we are in a loop
            
        hlst    = Q_acc.iloc[:N_hel].index
        
        #Updating the heliostats selected
        R1['hel_in'] = R1['hel'].isin(hlst)
        
        SF = pd.DataFrame( {'Eta_cos':Eta_cos, 'Eta_blk':Eta_blk, 'Eta_att':Eta_att, 'Eta_hbi':Eta_hbi, 'Eta_cpi':Eta_cpi, 'Eta_cpr':Eta_cpr, 'Eta_hel':Eta_hel,'Eta_CPC':Eta_CPC,'Eta_BDR':Eta_BDR, 'Eta_SF':Eta_SF, 'Q_h1':Q_h1, 'Q_acc':Q_acc, 'Q_avg':Q_avg} ).sort_values('Q_acc')
        Etas = SF.loc[hlst].mean()
        
        #########################################################
        # Writing the results for partial iteration
        text_r = '\t'.join('{:.4f}'.format(x) for x in [Nit, eta_hbi, N_hel, S_HB, Etas['Eta_hbi'], Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_cpi'], Etas['Eta_cpr'], Etas['Eta_CPC'], Etas['Eta_BDR'], Etas['Eta_SF']])+'\n'
        print(text_r[:-2])
        
        #Comparing with previous iteration
        if N_ant==N_hel:
            status = 'OK'
            break
        else:
            N_ant, N_an2 = N_hel, N_ant
        
        #Checking if even with max heliostat we do not have enough power
        if N_hel == N_max:
            status = 'Nmx'
            break
    
        #Checking if we reach the maximum number of iterations
        if Nit == 50:
            status = 'NC'
            break
        else:
            Nit+=1
    
    R2['hel_in'] = R2['hel'].isin(hlst)
    HB = {'Surface':HB_Surface, 'S_HB':S_HB, 'rlims':rlims}
    
    N_CPC,V_CPC,rO,rA,Cg = [ CPC[x] for x in ['N','V','rO','rA','Cg'] ]
    xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
    x0,y0 = CPC_Centers(Dsgn_CPC,rA,xrc,yrc)
    xCA, yCA, xCO, yCO = [],[],[],[]
    for i in range(N_CPC):
        #Plotting hexagons
        xA,yA = CPC_XY_R(rA,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)
        xO,yO = CPC_XY_R(rO,H_CPC,V_CPC,N_CPC,x0[i],y0[i],zrc)    
        xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
    xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
    xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
    
    Nx = 100; Ny = 100
    dx = (xmax-xmin)/Nx;    dy = (ymax-ymin)/Nx;   dA=dx*dy
    R2f = R2[(R2['hel_in'])&(R2['hit_rcv'])]
    Q_rcv,X,Y = np.histogram2d(R2f['xr'],R2f['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
    Nrays = len(R2f)
    Fbin    = Etas['Eta_SF'] * (CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*Nrays)
    Q_rcv = Fbin * Q_rcv
    
    del R0, R1, R2f
    gc.collect()
    
    return R2, Etas, SF, CPC, HB, hlst, Q_rcv, status

#############################################
#############################################

def SF_Plots(eta_type,data):
    """
    Function to plot the heliostat layout using different efficiencies as color code.

    Parameters
    ----------
    eta_type : string
        Efficiency to be used as color code.
    data : list
        Required information: R2, Etas_SF, folder_plot,case.

    Returns
    -------
    text_return : string
        Result message.

    """
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    R2, Etas_SF, folder_plot,case = data
    
    s=1
    f_s = 18
    # Heliostat Field
    
    # print('Creating plot for '+eta_type)
    if eta_type == 'eta_acc':   title = 'Heliostat Field. Optical efficiency per heliostat'
    if eta_type == 'eta_hbi':   title = 'Heliostat Field. HB intercept efficiency per heliostat'
    if eta_type == 'eta_cos':   title = 'Heliostat Field. Cosine efficiency per heliostat'
    if eta_type == 'eta_bdr':   title = 'Heliostat Field. BDR efficiency per heliostat'
    if eta_type == 'eta_cpi':   title = 'Heliostat Field. CPC intercept efficiency per heliostat'
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111)
    surf = ax1.scatter(R2['xi'],R2['yi'], s=s, c=R2[eta_type], cmap=cm.YlOrRd, vmax=(np.ceil(10*R2[eta_type].max())/10) )
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
    ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
    ax1.grid()
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(f_s)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(f_s)
    
    fig.savefig(folder_plot+'/'+case+'_'+eta_type+'.pdf', bbox_inches='tight')
    fig.savefig(folder_plot+'/'+case+'_'+eta_type+'.png', bbox_inches='tight')
    # plt.show()
    text_return = 'Creating plot for '+eta_type
    return text_return
