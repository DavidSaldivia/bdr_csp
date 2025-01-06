# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:45:36 2019

@author: z5158936
"""

import numpy as np
import pandas as pd

########################################
###### CONSTANT AND UTILITIES ##########
########################################

######### Calendar constants ###########
yr_days   = [0,31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
yr_dacc   = [ sum(yr_days[:x]) for x in range(len(yr_days))]
yr_names  = ['','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
yr_namel  = ['','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
yr_typd   = [0,17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]

###### Physical Constants ##############
Gsc       = 1367.           #[W/m2] Solar constant
Tbb_sun   = 5777.           #[K] Effective temperature of Sun

###### Conversion Units ##############
DtR       = np.pi / 180.
RtD       = 1. / DtR

#######################################
#######################################
class Sun:
    def __init__( self ):
        
        self.lat   = -38.7359           #[°] Latitude of solar position
        self.lng   = -72.5904           #[°] Longitude of solar position
        self.att   =  0.                #[m] Geogaphic altitude (meters above mean sea level)
        self.N     = 80                 #[-] day of position. By default March Equinox
        self.TSA   = 12.                #[°] [hr] hour of the day. Solar Time (ST). By default noon
        # self.UTC   = -3                 #[hr] Timezone for location. -3 by default
        self.h     = 0.                 #[°] hour of the day. Angle. By default noon
        self.m     = 3                  #[-] day and month of position
        self.d     = 21                 #[-] day of position expressed according to month
        self.Gbn   = 950                #[W/m2]
        self.Tbb   = Tbb_sun            #[K] Effective temperature of Sun

    ###############################################
    ###############################################
    ###############################################
    
    def update(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if 'lat' in kwargs:     self.lat   = kwargs['lat']
        if 'lng' in kwargs:     self.lng   = kwargs['lng']
        if 'att' in kwargs:     self.att   = kwargs['att']
        if 'N'   in kwargs:     self.N     = kwargs['N']
        if 'm'   in kwargs:     self.m     = kwargs['m']
        if 'd'   in kwargs:     self.d     = kwargs['d']
        if 't'   in kwargs:     self.t     = kwargs['t']
        if 'LT'  in kwargs:     self.LT    = kwargs['LT']
        if 'ST'  in kwargs:     self.ST    = kwargs['ST']
        if 'h'   in kwargs:     self.h     = kwargs['h']
        if 'UTC' in kwargs:     self.UTC   = kwargs['UTC']
        if 'Gbn' in kwargs:     self.Gbn   = kwargs['Gbn']
        if 'Tbb' in kwargs:     self.Tbb   = kwargs['Tbb']
        
        ##### Getting the day and the month, or the day of the year ######
        if 'N' in kwargs:
            self.m = 1      #Incomplete
            self.d = 1      #Incomplete
            
        elif 'd' in kwargs and 'm' in kwargs:
            self.N = yr_dacc[kwargs['m']] + kwargs['d']
            
        else:
            self.N, self.d, self.m = 80, 21, 3
            print('Values for d and m provided are not right to get N')
        
        
        ##### Getting the hour or the time. If nothing provided, set as noon ######
        if 't' in kwargs:
            self.t = kwargs['t']
            self.h = (self.t - 12)*15
        elif 'h' in kwargs:
            self.h = kwargs['h']
            self.t = self.h/15. + 12.
        else:
            self.t, self.h = 12., 0.
            
        ######## Getting the daily values ######## 
        self.angles_day()
        
        ######## Getting the hourly values ########
        self.angles_hour()
        
        return
    ###############################################
    ###############################################
    ###############################################
    
    def angles_day(self):
        """
        ATTRIBUTES CALCULARED HERE:
        
        B        =   [°] Day Angle
        Gon      =   [W/m2] Extraterrestrial radiation      (from Spencer (1971))
        ET       =   [min] Equation Time                    (from Spencer (1971))
        decl     =   [°] Declination                        (from Spencer (1971))
        hsunset  =   [°] Sunset hour
        hsunrise =   [°] Sunset hour
        
        tsunset  =   [hr] Sunset hour in Solar Time
        tsunrise =   [hr] Sunrise hour in Solar Time
        
        -------
        All angles in Degrees. Also all of them have a radians equivalent with '_r' at the end.
        e.g. decl_r is declination in radians.

        """
        
        self.log = ''
        lat , N = self.lat , self.N
        
        #Daily_Angle [°]
        B   = ((N - 1.) * 360. / 365.)
        B_r = B * DtR
        
        #Extraterrestrial radiation [W/m^2]
        Gon = Gsc * (1.00011 + 0.034221*np.cos(B_r) + 1.28e-3*np.sin(B_r)
                     + 7.19e-4*np.cos(2.*B_r) + 7.7e-5*np.sin(2.*B_r))

        #Equation of Time [min]
        ET = 229.2 * (7.5e-5 + 1.868e-3*np.cos(B_r) - 0.032077*np.sin(B_r)
                      - 0.014615*np.cos(2.*B_r) - 0.04089*np.sin(2.*B_r))
        
        #Declination [°]
        decl = RtD * (6.918e-3 - 0.399912*np.cos(B_r) + 0.070257*np.sin(B_r)
                      - 6.758e-3*np.cos(2*B_r) + 9.07e-4*np.sin(2*B_r)
                      - 2.697e-3*np.cos(3*B_r) + 1.48e-3*np.sin(3*B_r) )
        
        # Sunset and sunrise hours
        hsunset  = np.arccos(-np.tan(decl*DtR) * np.tan(lat*DtR)) *RtD     #[°]
        hsunrise = 0 - hsunset                                             #[°]
        tsunset  = 12. + hsunset/(15.)                               #[hr]
        tsunrise = 24. - tsunset                                           #[hr]
        
        ########### Outputs ##############
        self.B        = B
        self.Gon      = Gon
        self.ET       = ET
        self.decl     = decl
        self.hsunset  = hsunset
        self.hsunrise = hsunrise
        self.tsunset  = tsunset
        self.tsunrise = tsunrise
        
        ###### Angles in Radians #########
        self.B_r        = DtR * B
        self.decl_r     = DtR * decl
        self.hsunset_r  = DtR * hsunset
        self.hsunrise_r = DtR * hsunrise
        self.tsunset_r  = DtR * tsunset
        self.tsunrise_r = DtR * tsunrise

    ###############################################
    ###############################################
    ###############################################    
    
    def angles_hour(self, **kwargs):
        """
        ATTRIBUTES CALCULATED HERE:
          altit   =   [°] Solar altitude angle
          zenit   =   [°] Zenith angle
          azim    =   [°] Solar azimuth angle

        """
        
        #Update of Daily angles if necessary
        self.angles_day(**kwargs)
        # self.__dict__.update(kwargs)
        
        L, D, t = self.lat, self.decl, self.t
        h = (t - 12)*15                         #[°] hour of the day. Angle. By default noon
        L = L*DtR; D = D*DtR; h = h*DtR
        
        #Altitude & Zenith
        altit = np.arcsin( np.sin(L) * np.sin(D) + np.cos(L) * np.cos(D) * np.cos(h) )
        zenit = np.pi/2 - altit
        
        #Azimuth
        az_s0  = np.arcsin(np.cos(D) * np.sin(h) / np.cos(altit))
        azim   = az_s0 if (np.cos(h)>np.tan(D)/np.tan(L)) else (abs(az_s0) - np.pi) if (h<0.) else np.pi - az_s0
        
        # azim = np.sign(h)*abs(np.arccos( ( np.cos(zenit)*np.sin(L) - np.sin(D) ) / ( np.sin(zenit) * np.cos(L) )))
        # azim = abs(np.arccos( ( np.cos(zenit)*np.sin(L) - np.sin(D) ) / ( np.sin(zenit) * np.cos(L) )))
        
        ########### Outputs ##############
        self.altit = altit * RtD
        self.zenit = zenit * RtD
        self.azim  = azim  * RtD
        # self.airmass = 1.
        
        ###### Angles in Radians #########
        self.altit_r = altit
        self.zenit_r = zenit
        self.azim_r  = azim

        
#######################################
#######################################
class Plane:
    def __init__( self , beta = 0 , trk = 0 ):
        self.beta  = 0.            #[°] Inclination angle. By default 0
        self.trk   = 0             # Tracking system. By default 0 (no tracking)


#######################################
#######################################

def Solar_Vars(sun):
    # This function gives you solar angles and vars according to input data.
    # All input must be included in the dict 'sun'm with the following names



    # 3.- Collector: Hourly+
    #   trk = traking, according to number indicated below
    # 
    # OUTPUTS: According to the three levels, the following output are given:


    # 3.- Collector:
    #   az       =  [°] Surface Azimuth Angle
    #   beta     =  [°] Inclination angle
    #   theta    =  [°] Incidence angle
    #   Rb       =  [-] Rb factor (for radiation models)
    #
    #    ! tracking can be:
    #       0 = No tracking, if sun['beta'] not provided, then inclination = 0
    #       10 = full tracking (two axis)
    #       1 = E-W tracking, N-S axis
    #       2 = N-S tracking, E-W axis, daily adjustment for normal radiation at noon
    #       3 = N-S tracking, E-W axis, instantaneous adjustment
    #       4 = fixed tilted plane (tilt=lat), with vertical axis tracking
    #       5 = E-W tracking, paralell to earth N-S axis



    out = {}
    

    
    #####################################################
    if 'n' in sun and 'L' in sun and 't' in sun and 'trk' in sun:       #Collector Vars
        trk = sun['trk']
        
        if trk==0:
            beta = sun['beta'] if 'beta' in sun else abs(L)
            az   = 180.
            theta =    ma.sin(L*DtR) * ma.sin(decl*DtR) * ma.cos(beta*DtR)
            theta += - ma.cos(L*DtR) * ma.sin(decl*DtR) * ma.sin(beta*DtR) * ma.cos(az*DtR)
            theta +=   ma.cos(L*DtR) * ma.cos(decl*DtR) * ma.cos(beta*DtR) * ma.cos(h*DtR)
            theta +=   ma.sin(L*DtR) * ma.cos(decl*DtR) * ma.sin(beta*DtR) * ma.cos(az*DtR) * ma.cos(h*DtR)
            theta +=   ma.sin(h*DtR) * ma.cos(decl*DtR) * ma.sin(beta*DtR) * ma.sin(az*DtR)
            theta = ma.acos(theta) * RtD

        elif trk ==1:
            az    = 90. if az_s > 0. else -90.
            beta  = ma.atan(ma.tan(zen*DtR)*abs( ma.cos((az-az_s)*DtR)))*RtD
            theta = ma.acos(ma.sqrt( ma.cos(zen*DtR)**2 + ma.cos(decl*DtR)**2 * ma.sin(h*DtR)**2 )) * RtD

        elif trk ==2:
            az    = 0. if (L-decl>0.) else 180.
            beta  = abs(L - decl)
            theta = ma.acos(ma.sin(decl*DtR)**2 + ma.cos(decl*DtR)**2 * ma.cos(h*DtR)) * RtD

        elif trk ==3:
            az    = 0. if (az_s<90.) else 180.
            beta = ma.atan( ma.tan(zen*DtR) * abs(ma.cos( az_s*DtR )) ) * RtD
            theta = ma.acos( ma.sqrt( 1. - ma.cos(decl*DtR)**2 * ma.sin(h*DtR)**2 ) ) * RtD

        elif trk ==4:
            az    = az_s
            beta  = abs(L)
            theta = ma.acos( ma.cos(zen*DtR) * ma.cos(beta*DtR) + ma.sin(zen*DtR) * ma.sin(beta*DtR) ) * RtD

        elif trk ==10:
            az   = az_s
            beta = zen
            theta = 0.
        
        else:
            out['err'] = 'Not enough input for Collector angles'
        
        Rb  = ma.cos(theta*DtR) / ma.cos(zen*DtR) if theta < 90. else 0.
        
        out.update({ 'az':az , 'beta':beta , 'theta':theta , 'Rb':Rb})
    
    else:
        out['err'] = 'Not enough input for Collector angles'
    
    ########################################################
    
    return out

########################################################
def RadMods( model, vrs, sun ):
    #Estimation of radiation components and global radiation over surface.
    #The model selection is according this options
    #1 - Isotropic Model
    #2 - Hay & Davies' Anisotropic Model
    #3 - HDKR's Anisotropic Model
    #4 - Perez's Anisotropic Model
    
    #The output is the total radiation and beam radiation in given plane    
    out = {}
    
    Ihb, Ihd, albedo = vrs['Ihb'], vrs['Ihd'], vrs['albedo']
    Rb, beta, zenit  = sun['Rb'] , sun['beta'] , sun['zen'] 
    
    Iho              = 1300.    #!!!!
    
    if model == 1:
        It = Ihb*Rb + Ihd*(1+np.cos(beta))/2 + (Ihb+Ihd)*albedo*(1-np.cos(beta))/2
        Ib = Ihb * Rb
        
    elif model == 2:
        Ai = Ihb / Iho
        It = (Ihb+Ihd*Ai)*Rb + Ihd*(1-Ai)*(1+np.cos(beta))/2 + (Ihb+Ihd)*albedo*(1-np.cos(beta))/2
        Ib = (Ihb + Ihd * Ai) * Rb
    
    elif model == 3:
        Ai = Ihb / Iho
        It = (Ihb+Ihd*Ai)*Rb + Ihd*(1-Ai)*((1+np.cos(beta))/2)*(1+np.sqrt(Ihb/(Ihb+Ihd))*(np.sin(beta/2))**3) + (Ihb+Ihd)*albedo*(1-np.cos(beta))/2
        Ib = (Ihb + Ihd * Ai) * Rb
        
    elif model == 4:
        theta, Gon, dt = sun['theta'] , sun['Gon'] , sun['dt'] 
        m = 1/np.cos(zenit) if zenit < (70*DtR) else 1/( np.cos(zenit) + 0.5057*(96.08 - zenit/DtR )**(-1.634) )
        cte1, cte2 = max( 0., np.cos(theta) ), max( np.cos(85.*DtR), np.cos(zenit) )
        epsi = ( ( (Ihd + Ihb / np.cos(zenit) ) / Ihd ) + 1. + 5.535e-6*(theta*DtR)**3) / (1.+5.535e-6*(theta*DtR)**3) 
        delt = m * Ihd / (Gon * dt * 3600.)
        
        if (1.000 < epsi <= 1.065):     f = [-0.008 ,  0.588 , -0.062 , -0.060 ,  0.072 , -0.022 ]
        elif epsi <= 1.230:             f = [ 0.130 ,  0.683 , -0.151 , -0.019 ,  0.066 , -0.029 ]
        elif epsi <= 1.500:             f = [ 0.330 ,  0.487 , -0.221 ,  0.055 , -0.064 , -0.026 ]
        elif epsi <= 1.950:             f = [ 0.568 ,  0.187 , -0.295 ,  0.109 , -0.152 ,  0.014 ]
        elif epsi <= 2.800:             f = [ 0.873 , -0.392 , -0.362 ,  0.226 , -0.462 ,  0.001 ]
        elif epsi <= 4.550:             f = [ 1.132 , -1.237 , -0.412 ,  0.288 , -0.823 ,  0.056 ]
        elif epsi <= 6.200:             f = [ 1.060 , -1.600 , -0.359 ,  0.254 , -1.127 ,  0.131 ]
        else:                           f = [ 0.678 , -0.327 , -0.250 ,  0.156 , -1.377 ,  0.251 ]
            
        F1 = max( 0., ( f[0] + delt * f[1] + zenit * f[2] ) )
        F2 = f[3] + delt * f[4] + zenit * f[5]
        It = Ihb * Rb + Ihd*(1-F1)*(1+np.cos(beta))/2 + Ihd*F1*cte1/cte2 + Ihd*F2*np.sin(beta) + (Ihb + Ihd)*albedo*(1-np.cos(beta))/2
        Ib = Ihb * Rb + Ihd * F1 * cte1/cte2    
    else:
        print("You choose bad your options")
    out = {'It':It, 'Ib':Ib}
    
    return out
########################################################

#######################################
#######################################
def PClarityIndexes(Kt_av, Kt):
    #Inputs: KT_av, Kt are monthly average and daily clarity indexes
    
    Ktmin, Ktmax = 0.05, ( 0.6313 + 0.267*Kt_av - 11.9 * (Kt_av - 0.75)**8 )
    
    x   = ( Ktmax - Ktmin ) / ( Ktmax - Kt_av )
    g   = -1.498 + ( 1.184*x - 27.182 * ma.exp( -1.5*x ) ) / ( Ktmax - Ktmin )
    fKt = min( max( 1. , ( ma.exp(g*Ktmin) - ma.exp(g*Kt) ) / ( ma.exp(g*Ktmin) - ma.exp(g*Ktmax) ) ) , 0. )
    
    return fKt