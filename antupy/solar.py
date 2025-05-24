# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:45:36 2019

@author: z5158936
"""
from dataclasses import dataclass
import numpy as np

# Calendar constants
MONTHS_DAYS   = [0,31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_ACCDAYS   = [ sum(MONTHS_DAYS[:x]) for x in range(len(MONTHS_DAYS))]
MONTHS_NAMES_SHORT = ['','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTHS_NAMES_LONG  = [
    '','January', 'February', 'March',
    'April', 'May', 'June',
    'July', 'August', 'September',
    'October', 'November', 'December'
]
MONTHS_TYPICAL_DAY   = [0,17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]

# Physical Constants
Gsc       = 1367.           #[W/m2] Solar constant
Tbb_sun   = 5777.           #[K] Effective temperature of Sun

# Conversion Units
DtR       = np.pi / 180.
RtD       = 1. / DtR

class Sun:
    def __init__( self ):
        
        self.latitude = -38.7359           #[°] Latitude of solar position
        self.longitude = -72.5904           #[°] Longitude of solar position
        self.att   =  0.                #[m] Geogaphic altitude (meters above mean sea level)
        self.N     = 80                 #[-] day of position. By default March Equinox
        self.TSA   = 12.                #[°] [hr] hour of the day. Solar Time (ST). By default noon
        self.h     = 0.                 #[°] hour of the day. Angle. By default noon
        self.m     = 3                  #[-] day and month of position
        self.d     = 21                 #[-] day of position expressed according to month
        self.Gbn   = 950                #[W/m2]
        self.Tbb   = Tbb_sun            #[K] Effective temperature of Sun
    
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
        if 'lat' in kwargs:
            self.latitude   = kwargs['latitude']
        if 'lng' in kwargs:
            self.longitude = kwargs['longitude']
        if 'att' in kwargs:
            self.att = kwargs['att']
        if 'N'   in kwargs:
            self.N = kwargs['N']
        if 'm'   in kwargs:
            self.m = kwargs['m']
        if 'd'   in kwargs:
            self.d = kwargs['d']
        if 't'   in kwargs:
            self.t = kwargs['t']
        if 'LT'  in kwargs:
            self.LT = kwargs['LT']
        if 'ST'  in kwargs:
            self.ST = kwargs['ST']
        if 'h'   in kwargs:
            self.h  = kwargs['h']
        if 'UTC' in kwargs:
            self.UTC = kwargs['UTC']
        if 'Gbn' in kwargs:
            self.Gbn = kwargs['Gbn']
        if 'Tbb' in kwargs:
            self.Tbb = kwargs['Tbb']
        
        # Getting the day and the month, or the day of the year
        if 'N' in kwargs:
            self.m = 1      #Incomplete
            self.d = 1      #Incomplete
        elif 'd' in kwargs and 'm' in kwargs:
            self.N = MONTHS_ACCDAYS[kwargs['m']] + kwargs['d']
        else:
            self.N = 80
            self.d = 21
            self.m = 3
        
        
        # Getting the hour or the time. If nothing provided, set as noon
        if 't' in kwargs:
            self.t = kwargs['t']
            self.h = (self.t - 12)*15
        elif 'h' in kwargs:
            self.h = kwargs['h']
            self.t = self.h/15. + 12.
        else:
            self.t, self.h = 12., 0.
            
        # Getting the daily values
        self.angles_day
        
        # Getting the hourly values
        self.angles_hour
        
        return
    
    @property
    def angles_day(self) -> dict[str,float]:
        """
        Atributes calculated here:
        -------
        
        B =   [°] Day Angle
        Gon =   [W/m2] Extraterrestrial radiation      (from Spencer (1971))
        ET  =   [min] Equation Time                    (from Spencer (1971))
        declination =   [°] Declination                        (from Spencer (1971))
        hsunset =   [°] Sunset hour
        hsunrise =   [°] Sunset hour
        
        tsunset =   [hr] Sunset hour in Solar Time
        tsunrise =   [hr] Sunrise hour in Solar Time
        
        -------
        All angles in Degrees. Also all of them have a radians equivalent with '_r' at the end.
        e.g. decl_r is declination in radians.

        """
        
        self.log = ''
        latitude = self.latitude
        N = self.N
        
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
        declination = RtD * (6.918e-3 - 0.399912*np.cos(B_r) + 0.070257*np.sin(B_r)
                      - 6.758e-3*np.cos(2*B_r) + 9.07e-4*np.sin(2*B_r)
                      - 2.697e-3*np.cos(3*B_r) + 1.48e-3*np.sin(3*B_r) )
        
        # Sunset and sunrise hours
        hsunset  = np.arccos(-np.tan(declination*DtR) * np.tan(latitude*DtR)) *RtD     #[°]
        hsunrise = 0 - hsunset                                             #[°]
        tsunset  = 12. + hsunset/(15.)                               #[hr]
        tsunrise = 24. - tsunset                                           #[hr]
        
        return {
            "B": B,
            "Gon": Gon,
            "ET": ET,
            "declination": declination,
            "hsunset": hsunset,
            "hsunrise": hsunrise,
            "tsunset": tsunset,
            "tsunrise": tsunrise,
            "B_r": B_r,
            "declination_r": declination * DtR,
            "hsunset_r": hsunset * DtR,
            "hsunrise_r": hsunrise * DtR,
            "tsunset_r": tsunset * DtR,
            "tsunrise_r": tsunrise * DtR,
        }
    
    @property
    def B(self):
        return self.angles_day['B']
    @property
    def Gon(self):
        return self.angles_day['Gon']
    @property
    def ET(self):
        return self.angles_day['ET']
    @property
    def declination(self):
        return self.angles_day['declination']
    @property
    def hsunset(self):
        return self.angles_day['hsunset']
    @property
    def hsunrise(self):
        return self.angles_day['hsunrise']
    @property
    def tsunset(self):
        return self.angles_day['tsunset']
    @property
    def tsunrise(self):
        return self.angles_day['tsunrise']
    @property
    def B_r(self):
        return self.angles_day['B_r']
    @property
    def declination_r(self):
        return self.angles_day['declination_r']
    @property
    def hsunset_r(self):
        return self.angles_day['hsunset_r']
    @property
    def hsunrise_r(self):
        return self.angles_day['hsunrise_r']
    @property
    def tsunset_r(self):
        return self.angles_day['tsunset_r']
    @property
    def tsunrise_r(self):
        return self.angles_day['tsunrise_r']
    
    @property
    def angles_hour(self) -> dict[str,float]:
        """
        ATTRIBUTES CALCULATED HERE:
          altitude   =   [°] Solar altitude angle
          zenit   =   [°] Zenith angle
          azimuth    =   [°] Solar azimuth angle

        """
        
        latitude = self.latitude
        declination = self.declination
        time = self.t
        hour = (time - 12)*15                         #[°] hour of the day. Angle. By default noon
        L = latitude*DtR
        D = declination*DtR
        h = hour*DtR
        
        altitude = np.arcsin( np.sin(L) * np.sin(D) + np.cos(L) * np.cos(D) * np.cos(h) )
        zenit = np.pi/2 - altitude
        az_s0  = np.arcsin(np.cos(D) * np.sin(h) / np.cos(altitude))
        azimuth   = az_s0 if (np.cos(h)>np.tan(D)/np.tan(L)) else (abs(az_s0) - np.pi) if (h<0.) else np.pi - az_s0

        return {
            "altitude": altitude * RtD,
            "zenit": zenit * RtD,
            "azimuth": azimuth * RtD,
            "altitude_r": altitude,
            "zenit_r": zenit,
            "azimuth_r": azimuth,
        }
    
    @property
    def altitude(self):
        return self.angles_hour['altitude']
    @property
    def zenit(self):
        return self.angles_hour['zenit']
    @property
    def azim(self):
        return self.angles_hour['azim']
    @property
    def altitude_r(self):
        return self.angles_hour['altitude_r']
    @property
    def zenit_r(self):
        return self.angles_hour['zenit_r']
    @property
    def azim_r(self):
        return self.angles_hour['azim_r']

@dataclass
class Plane:
    traking: int = 0                     # Tracking system. By default 0 (no tracking)
    beta: float | None = None            # [°] Inclination angle. By default 0
    azimuth: float | None = None         # [°] Surface Azimuth Angle
    Rb: float = 0.                       # Rb factor (for radiation models)
    theta: float | None = None           # [°] Incidence angle
    Gon: float | None = None             # [W/m2] Global radiation on the plane

def solar_variables(
        sun: Sun,
        plane: Plane
    ):
    """
    Solar angles and variables for a given time and location defined in Sun.
    It requires the tracking type and the beta angle
    Tracking can be:
            0 = No tracking, if beta not in plane, then inclination = latitude
            1 = E-W tracking, N-S axis
            2 = N-S tracking, E-W axis, daily adjustment for normal radiation at noon
            3 = N-S tracking, E-W axis, instantaneous adjustment
            4 = fixed tilted plane (tilt=lat), with vertical axis tracking
            5 = E-W tracking, paralell to earth N-S axis
            10 = full tracking (two axis)

        Output Collector:
        az       =  [°] Surface Azimuth Angle
        beta     =  [°] Inclination angle
        theta    =  [°] Incidence angle
        Rb       =  [-] Rb factor (for radiation models)
    """    



    declination = sun.declination
    L = sun.latitude
    h = sun.h
    az_s = sun.azim
    zenith = sun.zenit

    tracking = plane.traking

    beta = None
    azimuth = None
    theta = None
    Rb = None
    
    if sun.N is None and sun.latitude is None and sun.t is None:       #Collector Vars
        raise ValueError('Not enough input for Collector angles')
        
    if tracking==0:
        beta = plane.beta if plane.beta is not None else abs(L)
        azimuth = 180.
        theta = (
            np.acos(
                np.sin(L*DtR) * np.sin(declination*DtR) * np.cos(beta*DtR)
                - np.cos(L*DtR) * np.sin(declination*DtR) * np.sin(beta*DtR) * np.cos(azimuth*DtR)
                + np.cos(L*DtR) * np.cos(declination*DtR) * np.cos(beta*DtR) * np.cos(h*DtR)
                + np.sin(L*DtR) * np.cos(declination*DtR) * np.sin(beta*DtR) * np.cos(azimuth*DtR) * np.cos(h*DtR)
                + np.sin(h*DtR) * np.cos(declination*DtR) * np.sin(beta*DtR) * np.sin(azimuth*DtR)
            ) * RtD
        )

    elif tracking ==1:
        azimuth = 90. if az_s > 0. else -90.
        beta  = np.atan(np.tan(zenith*DtR)*abs( np.cos((azimuth-az_s)*DtR)))*RtD
        theta = np.acos(np.sqrt( np.cos(zenith*DtR)**2 + np.cos(declination*DtR)**2 * np.sin(h*DtR)**2 )) * RtD

    elif tracking ==2:
        azimuth = 0. if (L-declination>0.) else 180.
        beta  = abs(L - declination)
        theta = np.acos(np.sin(declination*DtR)**2 + np.cos(declination*DtR)**2 * np.cos(h*DtR)) * RtD

    elif tracking ==3:
        azimuth = 0. if (az_s<90.) else 180.
        beta = np.atan( np.tan(zenith*DtR) * abs(np.cos( az_s*DtR )) ) * RtD
        theta = np.acos( np.sqrt( 1. - np.cos(declination*DtR)**2 * np.sin(h*DtR)**2 ) ) * RtD

    elif tracking ==4:
        azimuth = az_s
        beta  = abs(L)
        theta = np.acos( np.cos(zenith*DtR) * np.cos(beta*DtR) + np.sin(zenith*DtR) * np.sin(beta*DtR) ) * RtD

    elif tracking ==10:
        azimuth= az_s
        beta = zenith
        theta = 0.
        Rb  = np.cos(theta*DtR) / np.cos(zenith*DtR) if theta < 90. else 0.
    else:
        raise ValueError('No correct tracking type provided')
    
    return { 
        'azimuth':azimuth , 
        'beta':beta ,
        'theta':theta ,
        'Rb':Rb
    }

def radiative_models(
        model: int,
        sun: Sun,
        plane: Plane,
        params: dict[str,float],
    ) -> dict:
    """
    Estimation of radiation components and global radiation over surface.
    The model selection is according this options
        1 - Isotropic Model
        2 - Hay & Davies' Anisotropic Model
        3 - HDKR's Anisotropic Model
        4 - Perez's Anisotropic Model
    
    The output is the total radiation and beam radiation in given plane    
    """
    Ihb = params['Ihb']
    Ihd = params['Ihd']
    albedo = params['albedo']
    dt = params['dt']

    Rb  = plane.Rb
    beta = plane.beta if plane.beta is not None else np.nan
    zenit = sun.zenit 
    
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
        It = (
            (Ihb+Ihd*Ai)*Rb 
            + Ihd*(1-Ai)*((1+np.cos(beta))/2)*(1+np.sqrt(Ihb/(Ihb+Ihd))*(np.sin(beta/2))**3) 
            + (Ihb+Ihd)*albedo*(1-np.cos(beta))/2
            )
        Ib = (Ihb + Ihd * Ai) * Rb
        
    elif model == 4:
        theta = plane.theta if plane.theta is not None else np.nan
        Gon = plane.Gon if plane.Gon is not None else np.nan
        
        m = 1/np.cos(zenit) if zenit < (70*DtR) else 1/( np.cos(zenit) + 0.5057*(96.08 - zenit/DtR )**(-1.634) )
        c1 = max( 0., np.cos(theta) )
        c2 = max( np.cos(85.*DtR), np.cos(zenit) )
        epsi = ( ( (Ihd + Ihb / np.cos(zenit) ) / Ihd ) + 1. + 5.535e-6*(theta*DtR)**3) / (1.+5.535e-6*(theta*DtR)**3) 
        delt = m * Ihd / (Gon * dt * 3600.)
        
        if (1.000 < epsi <= 1.065):
            f = [-0.008 ,  0.588 , -0.062 , -0.060 ,  0.072 , -0.022 ]
        elif epsi <= 1.230:
            f = [ 0.130 ,  0.683 , -0.151 , -0.019 ,  0.066 , -0.029 ]
        elif epsi <= 1.500:
            f = [ 0.330 ,  0.487 , -0.221 ,  0.055 , -0.064 , -0.026 ]
        elif epsi <= 1.950:
            f = [ 0.568 ,  0.187 , -0.295 ,  0.109 , -0.152 ,  0.014 ]
        elif epsi <= 2.800:
            f = [ 0.873 , -0.392 , -0.362 ,  0.226 , -0.462 ,  0.001 ]
        elif epsi <= 4.550:
            f = [ 1.132 , -1.237 , -0.412 ,  0.288 , -0.823 ,  0.056 ]
        elif epsi <= 6.200:
            f = [ 1.060 , -1.600 , -0.359 ,  0.254 , -1.127 ,  0.131 ]
        else:
            f = [ 0.678 , -0.327 , -0.250 ,  0.156 , -1.377 ,  0.251 ]
            
        F1 = max( 0., ( f[0] + delt * f[1] + zenit * f[2] ) )
        F2 = f[3] + delt * f[4] + zenit * f[5]
        It = (
            Ihb * Rb 
            + Ihd*(1-F1)*(1+np.cos(beta))/2 + Ihd*F1*c1/c2 + Ihd*F2*np.sin(beta) 
            + (Ihb + Ihd)*albedo*(1-np.cos(beta))/2
            )
        Ib = Ihb * Rb + Ihd * F1 * c1/c2    
    else:
        print("Wrong model selected")
        It = None
        Ib = None
    return {'It':It, 'Ib':Ib}

def probability_clarity_indexes(Kt_av, Kt):
    """Inputs: KT_av, Kt are monthly average and daily clarity indexes"""
    
    Ktmin = 0.05
    Ktmax = ( 0.6313 + 0.267*Kt_av - 11.9 * (Kt_av - 0.75)**8 )
    
    x = ( Ktmax - Ktmin ) / ( Ktmax - Kt_av )
    g = -1.498 + ( 1.184*x - 27.182 * np.exp( -1.5*x ) ) / ( Ktmax - Ktmin )
    return min(
        max(
            1.,
            (np.exp(g*Ktmin) - np.exp(g*Kt))/(np.exp(g*Ktmin) - np.exp(g*Ktmax))
        )
        , 0. 
        )