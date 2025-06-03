from __future__ import annotations

from dataclasses import dataclass

from typing import Any, TYPE_CHECKING

from os.path import isfile
import pickle

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import fsolve
from scipy.integrate import quad
from pvlib.location import Location

from antupy.units import (Variable, conversion_factor as CF)

if TYPE_CHECKING:
    from bdr_csp.PowerCycle import PlantCSPBeamDownParticle


R0_COLS = ['xi','yi','zi', 'uxi','uyi','uzi', 'hel']
R1_COLS = ['hel','xi','yi','zi', 
           'xb','yb','zb', 'xc','yc','zc', 
           'uxi','uyi','uzi', 'uxb','uyb','uzb', 
           'hel_in', 'hit_hb', 'hit_tod']
R2_COLS = ['hel','xi','yi','zi', 
           'xb','yb','zb', 'xc','yc','zc', 
           'xs','ys','zs', 'xr','yr','zr', 
           'uxi','uyi','uzi', 'uxb','uyb','uzb', 
           'uxr','uyr','uzr', 
           'hel_in','hit_hb','hit_tod','hit_rcv','Nr_tod']

@dataclass
class SolarField():
    # Characteristics of Solar Field
    zf: Variable = Variable(50., "m")
    eta_sfr: Variable = Variable(0.97*0.95*0.95, "-")   # Solar field reflectivity
    eta_rfl: Variable = Variable(0.95, "m")             # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    err_x: Variable = Variable(0.001, "rad")            # Reflected error mirror in X direction
    err_y: Variable = Variable(0.001, "rad")            # Reflected error mirror in X direction
    A_h1: Variable = Variable(2.97**2,"m2")             # Area of one heliostat
    N_pan: int = 1                                      # Number of panels per heliostat
    file_SF: str|None = None                            # File with the rays data
    helios: pd.DataFrame|None = None                  # heliostats DataFrame

    def load_dataset(self, **kwargs) -> tuple[pd.DataFrame,pd.DataFrame]:
        """
        Function that read a file from SolarPilot datasets and convert it into R0 Dataframe

        Parameters
        ----------
        file_SF : string
            name of the SolarPilot file with the dataSet.
        **kwargs : various types
            Parameters that can change the calculations.
                N_pan:          Number of panels per heliostat, default 1
                save_plk:       if True convert the .csv file into .plk. Default False.
                read_plk:       if True, try to read .plk file. If not, look for .csv. Default True.
                N_hel:          Number of heliostat selected. If not given, use all of them

        Returns
        -------
        R0 : pandas DataFrame
            Initial Dataframe containing the ray dataset.
            Columns are: ['x','y','z','ux','uy','uz','hel']
        SF: pandas DataFrame
            Helisotat field DataFrame, indexes are hels.
            Columns are: ['xi','yi','zi','Eta_cos','Eta_blk']
        """
        
        #kwargs can be N_pan, N_hel, convert
        file_SF = self.file_SF
        N_pan = kwargs['N_pan'] if 'N_pan' in kwargs else 1
        save_plk = kwargs['save_plk'] if 'save_plk' in kwargs else False
        read_plk = kwargs['read_plk'] if 'read_plk' in kwargs else True
        
        if file_SF is None:
            raise ValueError("file_SF must be defined, it is the name of the file with the rays data.")

        if isfile(file_SF+'.plk') and read_plk:
            R0,SF = pickle.load(open(file_SF+'.plk','rb'))
            self.helios = SF
            return R0, SF
    
        rays = pd.read_csv(
            file_SF+'.csv',
            header=0,
            names=['xi','yi','zi','uxi','uyi','uzi','ele','stg','rayN']
        )

        #All rays that hit the solar field
        r_sf = rays[rays['stg']==1][['xi','yi','zi','ele','rayN']]
        r_sf['hel']  = np.sign(r_sf['ele'])*((np.abs(r_sf['ele'])-1)//N_pan + 1)
        # All rays that leave the solar field
        r_sf  = r_sf[r_sf['ele']>0]
        # All rays that go to tower
        r_tw  = rays[rays['stg']==2][['xi','yi','zi','uxi','uyi','uzi']]
        
        N_hel = kwargs['N_hel'] if 'N_hel' in kwargs else r_sf['hel'].max()
        
        #DF used to HB rays interceptions
        R0 = pd.merge(r_sf,r_tw, how='inner', on=['xi','yi','zi']).sort_values('hel')
        R0 = pd.DataFrame(R0[R0['hel']<=N_hel][['xi','yi','zi','uxi','uyi','uzi','hel']])
        R0['ri'] = (R0['xi']**2 + R0['yi']**2)**0.5
        
        #Getting the solar field DataFrame
        SF = R0.groupby('hel').mean()[['xi','yi','zi','ri','uxi','uyi','uzi']]
        SF['Eta_blk'] = (
            R0.groupby('hel').count()['xi'] 
            / r_sf.groupby('hel').count()['xi']
        )

        #Angle between normal to heliostat and sun
        sun  = -rays[rays['stg']==1][['uxi','uyi','uzi']].mean()    # Sun position (for Cos efficiency)
        aux1 = (
            (SF['uxi']+sun.iloc[0])*sun.iloc[0] 
            + (SF['uyi']+sun.iloc[1])*sun.iloc[1] 
            + (SF['uzi']+sun.iloc[2])*sun.iloc[2]
        )
        aux2 = np.sqrt(
            (SF['uxi']+sun.iloc[0])**2 
            + (SF['uyi']+sun.iloc[1])**2 
            + (SF['uzi']+sun.iloc[2])**2
        )
        SF['Eta_cos'] =  aux1 / aux2
        SF.drop(['uxi','uyi','uzi'], axis=1, inplace=True)
        if save_plk:
            pickle.dump((R0,SF),open(file_SF+'.plk','wb'))
        
        self.helios = SF
        return R0, SF


@dataclass
class HyperboloidMirror():
    zf: Variable = Variable(50., "m")
    fzv: Variable = Variable(0.83, "-")
    xrc: Variable = Variable(0.0, "m")
    yrc: Variable = Variable(0.0, "m")
    zrc: Variable = Variable(0.0, "m")
    eta_hbi: Variable = Variable(0.95, "-")
    eta_rfl: Variable = Variable(0.95, "-")

    #output
    rmin: Variable = Variable(None,"m")
    rmax: Variable = Variable(None,"m")
    zmin: Variable = Variable(None,"m")
    zmax: Variable = Variable(None,"m")
    surface_area: Variable = Variable(None,"m2")

    mass_total = Variable(None,"kg")
    mass_mirror = Variable(None,"kg")
    mass_structure = Variable(None,"kg")
    mass_fin = Variable(None,"kg")

    @property
    def zv(self) -> Variable:
        return Variable(self.zf.get_value("m") * self.fzv.get_value("-"), "m")

    def height_range(self) -> tuple[float, float]:       # HB_zrange
        zf = self.zf.get_value("m")
        zrc = self.zrc.get_value("m")
        fzv = self.fzv.get_value("-")

        if isinstance(self.rmin, Variable) and isinstance(self.rmax, Variable):
            rmin = self.rmin.get_value("m")
            rmax = self.rmax.get_value("m")
        else:
            raise ValueError("rmin and/or rmax has/have not been defined.")
        
        zfh = zf-zrc
        zvh = fzv*zf-zrc
        fvh = zvh/zfh
        c = zfh/2
        zo = zrc + (zf-zrc)/2
        a = c*( 2*fvh - 1 )
        b = 2*c*np.sqrt(fvh - fvh**2)
        zmin = (rmin**2/b**2 + 1 )**0.5*a + zo
        zmax = (rmax**2/b**2 + 1 )**0.5*a + zo
        self.zmin: Variable = Variable(zmin,"m")
        self.zmax: Variable = Variable(zmax,"m")
        return (zmin,zmax)


    def update_geometry(          # HB_Surface_Direct
            self,
            R1: pd.DataFrame,
        ) -> None:
        
        zf = self.zf.get_value("m")
        zv = self.zv.get_value("m")
        zrc = self.zrc.get_value("m")
        yrc = self.yrc.get_value("m")
        eta_hbi = self.eta_hbi.get_value("-")
        
        zfh = zf - zrc
        zvh = zv - zrc
        fvh = zvh / zfh
        c = zfh/2
        a = c*( 2*fvh - 1 )
        b = 2*c*np.sqrt(fvh - fvh**2)
        
        R1 = R1[R1['hel_in']].copy()
        R1['rb'] = (R1['xb']**2+R1['yb']**2)**0.5
        
        def S_int(r):
            return (1+a**2*r**2/(b**2*(r**2+b**2)))**0.5 * 2*np.pi * r
        
        rmin = R1['rb'].quantile(0.001)
        qmin = len( R1[R1['rb']<=rmin] ) / len(R1)
        if (qmin+eta_hbi) > 1:
            rmax = R1['rb'].max()
        else:
            rmax = R1['rb'].quantile(qmin+eta_hbi)
        
        surface_area = quad(S_int,rmin,rmax)[0]
        self.rmin = Variable(rmin,"m")
        self.rmax = Variable(rmax,"m")
        self.surface_area = Variable(surface_area,"m2")
        return None


    def mcrt_direct(      # HB_direct
            self,
            R0: pd.DataFrame,
            refl_error: bool = True
        ) -> pd.DataFrame:
        
        zf = self.zf.get_value("m")
        fzv = self.fzv.get_value("-")
        zrc = self.zrc.get_value("m")
        yrc = self.yrc.get_value("m")
        
        #hyperboloid geometry calculations
        zv = fzv*zf
        zfh = zf-zrc
        zvh = zv-zrc
        fvh = zvh/zfh
        (xo,yo,zo) = (0, yrc/2, zrc + zfh/2)
        c = np.sqrt(zfh**2+yrc**2)/2
        t = (yo - yrc) / (zo - 0.)
        a = c*( 2*fvh - 1 )
        b = 2*c*np.sqrt(fvh - fvh**2)
        
        # pandas
        R1 = R0.copy()
        m1 = R1['uyi'] * t + R1['uzi']
        m2 = R1['uxi']
        m3 = R1['uyi'] - R1['uzi'] * t
        n1  = (R1['yi']-yo)*t + (R1['zi']-zo)
        n2  = (R1['xi']-xo)
        n3  = (R1['yi']-yo) - (R1['zi']-zo) * t
        
        p2  = m1**2/a**2 - m2**2*(1+t**2)/b**2 - m3**2/b**2
        p1  = 2*(m1*n1/a**2 - m2*n2*(1+t**2)/b**2 - m3*n3/b**2)
        p0  = n1**2/a**2 - n2**2*(1+t**2)/b**2 - n3**2/b**2 - t**2 - 1
        kb  = ( -p1 + (p1**2 - 4*p2*p0)**0.5 ) / (2*p2)
        R1['xb'] = R1['xi'] + kb*R1['uxi']
        R1['yb'] = R1['yi'] + kb*R1['uyi']
        R1['zb'] = R1['zi'] + kb*R1['uzi']
        R1['rb'] = (R1['xb']**2+R1['yb']**2)**0.5
        
        # Partial differentiation
        ddx = -2*(R1['xb'] - xo)/b**2
        ddy = 2*t*((R1['yb']-yo)*t+(R1['zb']-zo))/a**2 - 2*((R1['yb']-yo) - t*(R1['zb']-zo))/b**2
        ddz = 2*((R1['yb']-yo)*t+(R1['zb']-zo))/a**2 + 2*t*((R1['yb']-yo) - t*(R1['zb']-zo))/b**2
        
        # Calculating reflected ray (perfect mirror)
        nn = (ddx**2+ddy**2+ddz**2)**0.5
        nx, ny, nz = ddx/nn, ddy/nn, ddz/nn
        sc = nx*R1['uxi'] + ny*R1['uyi'] + nz*R1['uzi']
        uxr = R1['uxi'] - 2*sc*nx
        uyr = R1['uyi'] - 2*sc*ny
        uzr = R1['uzi'] - 2*sc*nz
        
        #Including reflection errors
        if refl_error:
            R1['uxb'],R1['uyb'],R1['uzb'] = add_reflection_error(uxr,uyr,uzr)
        else:
            R1['uxb'],R1['uyb'],R1['uzb'] = uxr,uyr,uzr
        
        # Getting interception with second focal point
        kc = (zrc-R1['zb'])/R1['uzb']
        R1['xc'] = R1['xb'] + kc*R1['uxb']
        R1['yc'] = R1['yb'] + kc*R1['uyb']
        R1['zc'] = R1['zb'] + kc*R1['uzb']
        R1['rc'] = (R1['xc']**2+R1['yc']**2)**0.5
        
        return R1


    def mcrt_polars(
            self,
            R0: pl.DataFrame,
            refl_error: bool = True
        ) -> pl.DataFrame:
        
        zf = self.zf.get_value("m")
        fzv = self.fzv.get_value("-")
        zrc = self.zrc.get_value("m")
        yrc = self.yrc.get_value("m")
        
        #hyperboloid geometry calculations
        zv = fzv*zf
        zfh = zf-zrc
        zvh = zv-zrc
        fvh = zvh/zfh
        (xo,yo,zo) = (0, yrc/2, zrc + zfh/2)
        c = np.sqrt(zfh**2+yrc**2)/2
        t = (yo - yrc) / (zo - 0.)
        a = c*( 2*fvh - 1 )
        b = 2*c*np.sqrt(fvh - fvh**2)
        
        # Calculating interceptions with surface (explicit quadratic expression)
        R1_pl = (
            R0.with_columns(
                (pl.col("uyi") * t + pl.col("uzi")).alias("m1"),
                (pl.col("uxi")).alias("m2"),
                (pl.col("uyi") - pl.col("uzi") * t ).alias("m3"),
                ( (pl.col("uyi")-yo) * t + (pl.col("uzi")-zo) ).alias("n1"),
                ( pl.col("uxi")-xo ).alias("n2"),
                ( (pl.col("uyi")-yo) - (pl.col("uzi")-zo)*t ).alias("n3"),
            )
            .with_columns(
                (pl.col("m1")**2/a**2 - pl.col("m2")**2*(1+t**2)/b**2 - pl.col("m3")**2/b**2)
                .alias("p2"),
                (
                    2*(pl.col("m1")*pl.col("n1")/a**2 
                    - pl.col("m2")*pl.col("n2")*(1+t**2)/b**2
                    - pl.col("m3")*pl.col("n3")/b**2)
                ).alias("p1"),
                (
                    pl.col("n1")**2/a**2 
                    - pl.col("n2")**2*(1+t**2)/b**2 
                    - pl.col("n3")**2/b**2 - t**2 - 1
                ).alias("p0"),
            )
            .with_columns(     # calculating kb
                (
                    (- pl.col("p1") + (pl.col("p1")**2- 4*pl.col("p2")*pl.col("p0"))**0.5) / (2*pl.col("p2"))
                ).alias("kb")
            )
            .with_columns(     # intersection with mirror
                [
                    (pl.col(f'{j}i') + pl.col("kb")*pl.col(f"u{j}i"))
                    .alias("{j}b") for j in ["x", "y", "z"]
                ]
            )
            .with_columns( #partial differentiation
                (-2*(pl.col('xb') - xo)/b**2).alias("ddx"),
                (
                    2*t*( (pl.col('yb')-yo)*t+(pl.col('zb')-zo) )/ a**2 
                    - 2*( (pl.col('yb')-yo) - t*(pl.col('zb')-zo) ) / b**2
                ).alias("ddy"),
                (
                    2*((pl.col('yb')-yo)*t+(pl.col('zb')-zo))/a**2 
                    + 2*t*((pl.col('yb')-yo) - t*(pl.col('zb')-zo))/b**2
                ).alias("ddz"),
            )
            .with_columns(
                ((pl.col("xb")**2 + pl.col("yb")**2)**0.5).alias("rb"),
                ((pl.col("ddx")**2+pl.col("ddy")**2+pl.col("ddz")**2)**0.5).alias("nn")
            )
            .with_columns(
                [(pl.col(f"dd{j}")/pl.col("nn")).alias(f"n{j}") for j in ["x", "y", "z"]]
            )
            .with_columns(
                (
                    pl.col("nx")*pl.col("uxi") + pl.col("ny")*pl.col("uyi") + pl.col("nz")*pl.col("uzi")
                ).alias("sc")
            )
            .with_columns(      # (perfect) reflected rays
                [(pl.col(f"u{j}i") - 2*pl.col("sc")*pl.col(f"n{j}")).alias(f"u{j}r") for j in ["x", "y", "z"]]
            )
        )
        return R1_pl


    def shadow_point(
            self,
            alt: float,
            azi: float,
            hels: pd.DataFrame
        ) -> pd.DataFrame:
        
        rmax = self.rmax.get_value("m")
        self.height_range()
        zmin = self.zmin.get_value("m")
        zmax = self.zmax.get_value("m")

        alt_r, azi_r = np.radians(alt), np.radians(azi)
        shd_mx = zmax / np.tan(alt_r) + rmax
        shd_c  = zmin / np.tan(alt_r)
        
        shd_x  = -np.sin(azi_r) * shd_c
        shd_y  = -np.cos(azi_r) * shd_c
        shd_r  = (shd_mx - shd_c)
        
        #Shadowing
        Shadow = ((shd_x - hels['xi'])**2+(shd_y - hels['yi'])**2)**0.5 < shd_r
        hels['sh'] = np.where(Shadow,1,0)
        hels['f_sh'] = np.where(Shadow,0,1)
        
        return hels


    def shadow_simple(
            self,
            lat: float,
            lng: float,
            type_shdw: str,
            SF: pd.DataFrame
        ) -> pd.DataFrame:
        
        if type_shdw=='None':
            SF['r_sh'] = 1
            SF['f_sh'] = 1

        rmax = self.rmax.get_value("m")
        self.height_range()
        zmin = self.zmin.get_value("m")
        zmax = self.zmax.get_value("m")
        
        Ns = [4,34,64,95,125,156,186,217,246,277,307,338]
        tdelta = 0.25
        tz = 'Australia/Darwin'
        times = pd.date_range('2021-01-01','2021-12-31 23:59:00', tz=tz, freq=str(tdelta)+'h')
        times = times[times.dayofyear.isin(Ns)]
        sol_pos = Location(lat, lng, tz=tz ).get_solarposition(times)
        sol_pos = pd.DataFrame(sol_pos[sol_pos["elevation"]>0])
        
        alt_r  = np.radians(sol_pos["elevation"])
        azi_r  = np.radians(sol_pos["azimuth"])
        shd_mx = zmax / np.tan(alt_r) + rmax
        shd_c  = zmin / np.tan(alt_r)
        sol_pos['shd_x']  = - np.sin(azi_r) * shd_c
        sol_pos['shd_y']  = - np.cos(azi_r) * shd_c
        sol_pos['shd_r']  = (shd_mx - shd_c)
        
        SF['sh'] = 0
        for idx,row in sol_pos.iterrows():
            Shadow = (
                ((row["shd_x"] - SF['xi'])**2+(row["shd_y"] - SF['yi'])**2)**0.5 
                < row["shd_r"]
            )
            SF['sh'] = np.where(Shadow, SF['sh']+1, SF['sh'])
        
        ft = len(sol_pos)
        f_days = 365/len(Ns)
        SF['r_sh'] = SF['sh'] / ft
        SF['sh']   = SF['sh'] * tdelta * f_days
        
        if type_shdw=='simple':
            SF['f_sh'] = np.where(SF['r_sh']>0.10,0,1)
        elif  type_shdw=='fraction':
            SF['f_sh'] = SF['r_sh']

        return SF
    
    def calculate_mass(
        self,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        Gbn: float = 950,
        A_h1: float = 2.92**2,
    ) -> None:

        rmin = self.rmin.get_value("m")
        rmax = self.rmax.get_value("m")
        eta_rfl = self.eta_rfl.get_value("-")

        Etas = SF[SF['hel_in']].mean()
        hlst = SF[SF.hel_in].index
        N_hel = len(hlst)
        
        out  = R2[(R2['hel_in'])&(R2['hit_hb'])]
        xmin = out['xb'].min()
        xmax = out['xb'].max()
        ymin = out['yb'].min()
        ymax = out['yb'].max()
        Nx = 100
        Ny = 100
        dx = (xmax-xmin)/Nx
        dy = (ymax-ymin)/Nx
        dA = dx*dy
        Q_HB,X,Y = np.histogram2d(
            out['xb'], out['yb'], 
            bins = [Nx,Ny],
            range = [[xmin, xmax], [ymin, ymax]],
            density = False
        )
        X2, Y2 = np.meshgrid(X, Y)
        Xc = np.array(
            [ [(X2[i,j] + X2[i,j+1])/2 for j in range(Ny)] for i in range(Nx) ]
        )
        Yc = np.array(
            [ [(Y2[i,j]+Y2[i+1,j])/2 for j in range(Ny)] for i in range(Nx) ]
        )
        
        def HB_dAs(
                X: np.ndarray,
                Y: np.ndarray,
                HB: HyperboloidMirror
            ) -> np.ndarray:
            
            zf = HB.zf.get_value("m")
            fzv = HB.fzv.get_value("-")
            zrc = HB.zrc.get_value("m")
            yrc = HB.yrc.get_value("m")
            zfh,zvh = zf-zrc, fzv*zf-zrc
            fvh = zvh/zfh
            c = zfh/2
            a = c*( 2*fvh - 1 )
            b = 2*c*np.sqrt(fvh - fvh**2)
            R = np.sqrt(X**2+Y**2)
            Z = np.sqrt( R**2/b**2 + 1 )*a
            dA = np.sqrt( (a/b)**4 * (R/Z)**2 + 1)
            return dA
        
        dS_HB = HB_dAs(Xc, Yc, self) * dx * dy
        Rc = np.sqrt( Xc**2 + Yc**2 )
        S_HB = dS_HB[(Rc>rmin)&(Rc<rmax)].sum()
        Fbin = (
            eta_rfl*Etas['Eta_cos']*Etas['Eta_blk']
            * Gbn * A_h1 * N_hel
            /(1e3*dS_HB*len(out))
        ) 
        Q_HB = Fbin * Q_HB
        
        Q_HB_tot = (Q_HB*dS_HB).sum()               #kW
        Q_HB_abs = Q_HB_tot*(1-eta_rfl)      #kw
        Q_HB_cool_avg = Q_HB_abs/S_HB               #kW/m2
        Q_HB_worst = Q_HB.max()*(1-eta_rfl)  #kW/m2
        dT_mirr   = 70.                             #K
        
        HTC_max = 1e3 * Q_HB_worst / dT_mirr             #W/m2-K
        HTC_avg = 1e3 * Q_HB_cool_avg / dT_mirr          #W/m2-K
        
        [N_fin,M_fin,M_mirror,M_beams,M_total,rAf,rMf] = HB_cooling_calc(Q_HB,eta_rfl)

        N_fin_tot = (N_fin*dS_HB)[(Rc>rmin) & (Rc<rmax)].sum()
        M_fin_tot = (M_fin*dS_HB)[(Rc>rmin) & (Rc<rmax)].sum() /1e3   #[tonnes]
        M_mirr_tot = M_mirror*S_HB / 1e3                                          #[tonnes]
        M_str_tot = M_beams*S_HB / 1e3                                            #[tonnes]
        M_HB_tot  = M_fin_tot + M_mirr_tot + M_str_tot                            #[tonnes]
        
        self.mass_total = Variable(M_HB_tot, "ton")
        self.mass_mirror = Variable(M_mirr_tot, "ton")
        self.mass_structure = Variable(M_str_tot, "ton")
        self.mass_fin = Variable(M_fin_tot, "ton")

        return None

@dataclass
class TertiaryOpticalDevice():
    geometry: str | None = None
    array: str | None = "A"
    xrc: Variable = Variable(0.0,"m")
    yrc: Variable = Variable(0.0,"m")
    zrc: Variable = Variable(0.0,"m")
    
    radius_ap: Variable= Variable(None, "m")
    radius_out: Variable= Variable(None, "m")
    height: Variable= Variable(None, "m")
    Cg: Variable= Variable(None, "-")
    receiver_area: Variable= Variable(None, "m2")

    def __post_init__(self):
        self.update_params()

    def update_params(self):

        #Depending the parameters received, the others are calculated
        geometry  = self.geometry
        array = self.array
        N = self.n_tods
        V = self.n_sides
        
        #For Paraboloid, it requires two of the following parameters: rA, rO, H, Cg, Arcv
        if geometry == 'PB':
            if self.radius_ap.v is not np.nan and self.radius_out.v is not np.nan:
                rA = self.radius_ap.get_value("m")
                rO = self.radius_out.get_value("m")
                H  = rA**2 - rO**2
                Cg = (rA/rO)**2                      #Concentration ratio of each TOD
                
            elif self.radius_ap.v is not np.nan and self.height.v is not np.nan:
                rA = self.radius_ap.get_value("m")
                H = self.height.get_value("m")
                
                rO    = (rA**2 - H)**0.5               # Check if it is over limits
                if rO<0:
                    print("Height is too much for rA, will be replaced by height for min rO")
                    rO = 0.2
                    H  = rA**2-rO**2
                Cg = (rA/rO)**2
            
            elif self.radius_ap.v is not np.nan and self.Cg.v is not np.nan:
                rA = self.radius_ap.get_value("m")
                Cg = self.Cg.get_value("-")
                rO    = rA/Cg**0.5
                H     = rA**2 - rO**2
            
            elif self.radius_out.v is not np.nan and self.height.v is not np.nan:
                rO = self.radius_out.get_value("m")
                H = self.height.get_value("m")
                rA    = (H + rO**2)**0.5
                Cg    = (rA/rO)**2
            
            elif self.radius_out.v is not np.nan and self.Cg.v is not np.nan:
                rO = self.radius_out.get_value("m")
                Cg = self.Cg.get_value("-")
                rA    = rO*Cg**0.5
                H     = rA**2-rO**2
            
            elif self.receiver_area.v is not np.nan and self.Cg.v is not np.nan:
                Arcv = self.receiver_area.get_value("m2")
                Cg = self.Cg.get_value("-")
                rO = (Arcv / ( V*N*np.tan(np.pi/V) ) )**0.5
                rA    = rO*Cg**0.5
                H     = rA**2-rO**2
            
            else:
                raise ValueError("Wrong input parameters")
            
            phi   = np.radians(360./V)
            S1    = (
                V*np.tan(phi/2)/6 
                * ( ( 1 + 4*rA**2 )**(3/2) - ( 1 + 4*rO**2)**(3/2))
            )
            S_TOD = N * S1
            tht   = np.arccos(H**0.5/rA)
            theta = np.degrees(np.arccos(H**0.5/rA))
            zmin  = 0.
            zmax  = rA**2
            fl    = np.nan
            
        #For CPC, for now, it only accepts rO and Cg as initial parameters
        elif geometry =='CPC':
            rO = self.radius_out.get_value("m")
            Cg = self.Cg.get_value("-")

            RtD = 180./np.pi
            rA = rO * Cg**0.5
            tht_mx = np.arcsin(1/Cg**0.5)
            fl = rO * (1+np.sin(tht_mx))
            phi_i = np.pi #/2+tht_mx
            phi_f = 2*tht_mx
            
            Ndz = 1000
            phi1 = np.linspace(phi_i,phi_f,Ndz)
            R  = 2 * fl/ ( 1- np.cos(phi1) )
            r1 = 2 * fl * np.sin(phi1-tht_mx) / ( 1- np.cos(phi1) ) - rO
            z1 = 2 * fl * np.cos(phi1-tht_mx) / ( 1- np.cos(phi1) )
            
            def func_Fzmin(zs,*args):
                r_out, Cg = args
                tht = np.arcsin(1/Cg**0.5)
                rs = 0.
                a1 = ((rs+r_out)*np.cos(tht) + zs*np.sin(tht))**2
                a2 = 4*r_out*(1+np.sin(tht))
                a3 = zs*np.cos(tht) - rs*np.sin(tht) + r_out
                return a1 - a2*a3
            zmin = fsolve(func_Fzmin,0,args=(rO,Cg))[0]
            zmax = 2 * fl * np.cos(phi_f-tht_mx) / ( 1- np.cos(phi_f) )
            H = zmax
            tht   = tht_mx
            theta = np.degrees(tht)
            phi   = np.radians(360./V)
            
            zo = 0.
            dz = (zmax-zo)/Ndz
            dr = np.ediff1d(r1)
            S1 = 2*np.pi*sum(r1[:-1]*((dz/dr)**2+1)**0.5*dr)    #Only valid for Full CPC
            S1 = 2*V*np.tan(np.pi/V)*sum(r1[:-1]*((dz/dr)**2+1)**0.5*dr)    #For polygon
            S_TOD = N * S1
            
        elif (geometry is None) or (geometry == 'N'):
            array = 'N'
            N,V,phi,S1,H,Cg, = 0,0,0,0,0,1
            rO = self.radius_out.get_value("m")
            rA = self.radius_ap.get_value("m")
            S_TOD = np.nan
            theta = np.nan
            tht = np.nan
            fl = np.nan
            zmin = 0.
            zmax = np.nan
        else:
            raise ValueError("Wrong geometry type, it must be 'PB', 'CPC' or None")
        
        ## Creating the object
        self.radius_ap = Variable(rA,"m")
        self.radius_out = Variable(rO,"m")
        self.height = Variable(H,"m")
        self.Cg = Variable(Cg,"-")
        self.surface_area = Variable(S_TOD,"m2")
        self.aperture_area = Variable( (V*rA**2*np.tan(phi/2)*N), "m2" )
        self.receiver_area = Variable( V * rO**2 * np.tan(phi/2) * N, "m2" )
        self.x0 = self.array_centers()[0]
        self.y0 = self.array_centers()[1]
        self.theta = Variable(theta,"rad")
        self._tht = tht
        self.focal_length = Variable(fl,"m")
        self.zmin = Variable(zmin,"m")
        self.zmax = Variable(zmax,"m")

        return None
    
    @property
    def n_tods(self) -> int:       #N_TOD
        return self._get_array_values()[0]
    
    @property
    def n_sides(self) -> int:      #V_TOD
        return self._get_array_values()[1]

    def _get_array_values(self) -> tuple[int,int]:
        array_values = {
            "A": (3, 6),           # 3 hexagons with centered vertix
            "B": (7, 6),           # 7 hexagons, with one centered
            "C": (4, 4),           # 4 squares, with centered vertix
            "D": (4, 4),           # 4 squares, with two sharing center and two in shorter side
            "E": (1, 8),           # 1 octagon centered
            "F": (1,int(1e6)),     # Full circle
            "N": (1,0),            # Non-TOD
        }
        if self.array not in array_values:
            raise ValueError(f"Invalid array type: {self.array}. Must be one of {list(array_values.keys())}.")
        return array_values[self.array]

    def array_centers(self) -> tuple[list[float], list[float]]:
        """
        Returns
        -------
        x0 : list
            x values for center positions.
        y0 : list
            y values for center positions.

        """
        
        V_TOD = self.n_sides
        rA = self.radius_ap.get_value("m")
        xrc = self.xrc.get_value("m")
        yrc = self.yrc.get_value("m")

        phi   = np.radians(360/V_TOD) if (V_TOD > 0) else 2*np.pi
        match self.array:
            case "A":
                x0 = [ (2*rA/3**0.5)*np.cos(2*n*phi) + xrc for n in range(3) ]
                y0 = [ (2*rA/3**0.5)*np.sin(2*n*phi) + yrc for n in range(3) ]
            case "B":
                x0 = [xrc] + [ (2*rA)*np.sin(phi*n) + xrc for n in range(6) ]
                y0 = [yrc] + [ (2*rA)*np.cos(phi*n) + yrc for n in range(6) ]
            case "C":
                x0 = [ (2*rA/2**0.5)*np.cos(n*phi) + xrc for n in range(4) ]
                y0 = [ (2*rA/2**0.5)*np.sin(n*phi) + yrc for n in range(4) ]
            case "D":
                x0 = [rA/2**0.5, -rA/2**0.5, 2**0.5*rA, -2**0.5*rA]
                y0 = [-rA/2**0.5, rA/2**0.5, 2**0.5*rA, -2**0.5*rA]
            case "E":
                x0 = [0.]
                y0 = [0.]
            case "F":
                x0 = [0.]
                y0 = [0.]
            case "N":
                x0 = [0.]
                y0 = [0.]
            case _:
                x0 = [0.]
                y0 = [0.]
        return (x0, y0)

    def perimeter_points(       # TOD_XY_R
                self,
                ri: float,
                Np: int = 100,
                tod_index: int = 1,
            ) -> tuple[np.ndarray, np.ndarray]:
            
            xo = self.array_centers()[0][tod_index]
            yo = self.array_centers()[1][tod_index]
            V = self.n_sides

            phi = 360./V
            xmax = ri/np.cos(np.radians(phi/2))
            xmin = ri/np.cos(np.radians((np.floor(180./phi)+0.5)*phi))
            if xmin<-xmax:
                xmin=-xmax         #To avoid nan values on arccos
            
            xx = np.linspace(xmax,xmin,Np)
            angles = np.degrees(np.arccos( xx/xmax ) )
            ii = np.floor(angles/phi);
            
            phii = np.radians((ii+0.5)*phi)
            xi, yi = ri*np.cos(phii), ri*np.sin(phii)
            slopes = np.tan(phii + np.pi/2)
            intercepts = (yi-slopes*xi)
            yy     = slopes*xx + intercepts
            
            xx = np.append(xx,np.flip(xx)) + xo
            yy = np.append(yy,-np.flip(yy)) + yo
            
            return xx,yy

    def limits(self) -> tuple[float,float,float,float]:
        xCA, yCA = [],[]
        for i in range(self.n_tods):
            xA, yA = self.perimeter_points(self.radius_ap.get_value("m"), tod_index=i)
            xCA.append(xA)
            yCA.append(yA)
        xCA=np.array(xCA)
        yCA=np.array(yCA)
        (xmin,xmax,ymin,ymax) = (xCA.min(), xCA.max(), yCA.min(), yCA.max())

        return xmin,xmax,ymin,ymax
    
    def radiation_flux(self, R2, total_rad) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        xmin,xmax,ymin,ymax = self.limits()
        Nx = 100
        Ny = 100
        dx = (xmax-xmin)/Nx
        dy = (ymax-ymin)/Ny
        dA = dx*dy
        R2f = R2[(R2['hel_in'])&(R2['hit_rcv'])]
        Q_rcv,X,Y = np.histogram2d(
            R2f['xr'],R2f['yr'],
            bins=[Nx,Ny],
            range=[[xmin, xmax], [ymin, ymax]],
            density=False
        )
        Nrays = len(R2f)
        Fbin = total_rad/(1e3*dA*Nrays)
        Q_rcv = Fbin * Q_rcv.transpose()
        X, Y = np.meshgrid(X, Y)

        return Q_rcv, X, Y

    def mcrt_solver(
        self,
        R1: pd.DataFrame,
        refl_error: bool = True
    ) -> pd.DataFrame:
        
        geometry = self.geometry
        array = self.array
        N_TOD = self.n_tods
        V_TOD = self.n_sides
        H_TOD = self.height.get_value("m")
        rA = self.radius_ap.get_value("m")
        rO = self.radius_out.get_value("m")
        x0 = self.x0
        y0 = self.y0
        zmin = self.zmin.get_value("m")
        zmax = self.zmax.get_value("m")
        Cg = self.Cg.get_value("-")
        zrc = self.zrc.get_value("m")

        #If there is no TOD, the function return R2==R1 with some extra labels
        if array=='N':
            R2 = R1.copy()
            R2['hit_rcv'] = (R2['rc'] < rA) & (R2['hit_hb'])
            R2['hit_tod'] = R2['hit_rcv']
            R2['Nr_tod'] = 0
            R2['xr'] = R2['xc']
            R2['yr'] = R2['yc']
            R2['zr'] = R2['zc']
            R2['uxr'] = R2['uxb']
            R2['uyr'] = R2['uyb']
            R2['uzr'] = R2['uzb']
            return R2
        
        # INITIAL CALCUALTIONS
        #Getting the rays that enter the mirror and local-global coordinate variables
        # CPCs
        if (geometry == 'CPC' and array == 'F'):
            zV = zrc - H_TOD
            zA = H_TOD
            zO = 0.
            R1['xo'] = x0[0]
            R1['yo'] = y0[0];
            R1['hit_tod'] = (
                CPC_enter(R1['xc'], R1['yc'], rA, array, V_TOD, R1['xo'], R1['yo'])
                & R1['hit_hb']
            )
            R1['Npolygon'] = np.where( R1['hit_tod'], 1, 0)
        
        elif (geometry =='CPC' and array in ['A','B','C','D','E']):
            zV = zrc - H_TOD
            zA = H_TOD
            zO = 0.
            R1['Npolygon'] = 0
            R1['xo'] = np.nan
            R1['yo'] = np.nan
            xc = R1[R1['hit_hb']]['xc']
            yc = R1[R1['hit_hb']]['yc']
            xo,yo,Npolygon = R1['xo'], R1['yo'], R1['Npolygon']
            for i in range(N_TOD):
                x0i, y0i = x0[i], y0[i]
                hit = CPC_enter(R1['xc'],R1['yc'],rA,array,V_TOD,x0i,y0i)
                aux = hit[hit]
                xo.update( x0i*aux )
                yo.update( y0i*aux )
                Npolygon.update((i+1)*aux)
                xc = xc[~hit]
                yc = yc[~hit]
            R1['hit_tod'] = np.where(R1['Npolygon']>0,True,False)
        
        # Paraboloids
        elif (geometry == 'PB' and array == 'F'):
            zV = zrc - rA**2
            zA = rA**2
            zO = rO**2
            R1['hit_tod'] = (R1['rc']<rA)&(R1['hit_hb'])
            R1['Npolygon'] = np.where(R1['hit_tod'],1,0)
            R1['xo'] = x0[0]
            R1['yo'] = y0[0];
            
        elif (geometry =='PB' and array in ['A','B','C','D','E']):
            zV = zrc - rA**2
            zA = rA**2
            zO = rO**2
            
            R1['Npolygon'] = 0
            R1['xo'] = np.nan
            R1['yo'] = np.nan
            xc = R1[R1['hit_hb']]['xc']
            yc = R1[R1['hit_hb']]['yc']
            xo,yo,Npolygon = R1['xo'], R1['yo'], R1['Npolygon']
            for i in range(N_TOD):
                x0i, y0i = x0[i], y0[i]
                hit = PB_Z( xc, yc, V_TOD, x0i, y0i) < zA
                aux = hit[hit]
                xo.update( x0i*aux )
                yo.update( y0i*aux )
                Npolygon.update((i+1)*aux)
                xc = xc[~hit]
                yc = yc[~hit]
            R1['hit_tod'] = np.where(R1['Npolygon']>0,True,False)
        else:
            zV = np.nan
            zA = np.nan
            zO = np.nan
        
        #INITIAL DF
        R2 = R1[R1['hit_tod']][[
            'xc','yc','zc','uxb','uyb','uzb','xo','yo','hit_tod','Npolygon'
        ]]
        R2.rename(
            columns={
                'xc':'xn', 'yc':'yn', 'zc':'zn',
                'uxb':'uxn', 'uyb':'uyn', 'uzb':'uzn'
                },
            inplace = True
        )
        R2['xs'] = 0.
        R2['ys'] = 0.
        R2['zs'] = 0.
        R2['hit_rcv'] = False
        R2['Nr_tod']  = 0
        R2['zn'] = zA           #moving everything to TOD coordinates
        R2f = R2.copy()         #this will be the final df
        R2f['xr'] = np.nan
        R2f['yr'] = np.nan
        R2f['zr'] = np.nan
        R2f['rr'] = np.nan
        R2f['uxr'] = np.nan
        R2f['uyr'] = np.nan
        R2f['uzr'] = np.nan
        
        # STARTING THE LOOP
        Nrfl = 1
        rays_ant = 0
        method = 'NR'
        tol=1e-4
        h = 1e-4
        while True:
        
            # CALCULATING FOR INTERSECTIONS
            N_ini=len(R2)
            # For PB-F it is possible to solve the equations directly.
            # All other PBs and CPCs require non-linear solver
            
            if (geometry == 'PB' and array == 'F'):
                p2 = R2['uxn']**2 + R2['uyn']**2
                p1 = 2*(R2['xn']*R2['uxn'] + R2['yn']*R2['uyn']) - R2['uzn']
                p0 = R2['xn']**2 + R2['yn']**2 - R2['zn']
                ks   = np.where(
                    abs(p2)>1e-8,
                    ( -p1 + (p1**2 - 4*p2*p0)**0.5 ) / (2*p2),
                    -p0/p1
                )
                R2['zs'] = R2['zn'] + ks*R2['uzn']
                
                no_sol = ( (R2['zs']>zA) | R2['zs'].isnull() )
                R2rjct = R2[no_sol]
                R2f.update(R2rjct)
                R2 = R2[~no_sol]
                ks = ks[~no_sol]
                
            elif method == 'NR':
                
                k_a = pd.Series(h/2,index=R2.index,dtype='float64')
                k_b = (zmin - R2['zn']) / R2['uzn']
                
                if geometry == 'CPC':
                    args = (R2, array, rO, Cg, V_TOD)
                    Fk_a = CPC_Fk(k_a,args)
                    Fk_b = CPC_Fk(k_b,args)
                elif geometry == 'PB':
                    Fk_a = PB_Fk(k_a, R2, V_TOD)
                    Fk_b = PB_Fk(k_b, R2, V_TOD)
                
                sol_1 = (Fk_a*Fk_b < 0)
                R2rjct = R2[~sol_1]
                R2f.update(R2rjct)
                R2 = R2[sol_1]
                
                ki = (zmin - R2['zn']) / R2['uzn']
                R2i = R2.copy()
                ks = ki.copy()
                for it in range(40):
                    if geometry == 'CPC':
                        args = (R2i, array, rO, Cg,V_TOD)
                        dFk = ( CPC_Fk(ki+h/2,args) - CPC_Fk(ki-h/2,args) ) / h
                        kj  = ki - CPC_Fk(ki,args)/dFk
                    elif geometry == 'PB':
                        dFk = (PB_Fk(ki+h/2,R2i,V_TOD) - PB_Fk(ki-h/2,R2i,V_TOD)) / h
                        kj  = ki - PB_Fk(ki,R2i,V_TOD)/dFk
                        
                    err_k = (ki-kj).abs()
                    errmax = err_k.max()
                    ks.update(kj[err_k<tol])
                    R2i = R2i[err_k>tol]
                    ki = kj[err_k>tol]
                    if errmax<tol or len(R2i)==0: break
                
            R2['ks'] = ks
            R2['xs'] = R2['xn'] + ks*R2['uxn']
            R2['ys'] = R2['yn'] + ks*R2['uyn']
            R2['zs'] = R2['zn'] + ks*R2['uzn']
            R2['rs'] = (R2['xs']**2+R2['ys']**2)**0.5

            if geometry == 'CPC':
                R2['hit_rcv'] = (
                    CPC_enter(R2['xs'], R2['ys'], rA, array, V_TOD, R2['xo'], R2['yo']) 
                    & (R2['zs']<zO)
                )
            elif geometry == 'PB':
                R2['hit_rcv'] = (R2['zs']<=zO)
            
            # UPDATING RAYS INTO RECEIVER
            #Rays that already hit the outlet are updated
            R2rcv = R2[R2['hit_rcv']].copy()
            if (len(R2rcv)>0):
                kr = (zO - R2rcv['zs'])/R2rcv['uzn']
                R2rcv['xr'] = R2rcv['xs']+kr*R2rcv['uxn']
                R2rcv['yr'] = R2rcv['ys']+kr*R2rcv['uyn']
                R2rcv['zr'] = R2rcv['zs']+kr*R2rcv['uzn']
                R2rcv['rr'] = (R2rcv['xr']**2+R2rcv['yr']**2)**0.5
                R2rcv['uxr'] = R2rcv['uxn']
                R2rcv['uyr'] = R2rcv['uyn']
                R2rcv['uzr'] = R2rcv['uzn']
                
                R2rcv['xs'] = R2rcv['xn']
                R2rcv['ys'] = R2rcv['yn']
                R2rcv['zs'] = R2rcv['zn']
                R2f.update(R2rcv)
            
            # CALCULATING REFLECTED DIRECTION FOR NEXT ITERATION
            # Rays that goes to next iteration
            R2 = R2[~R2['hit_rcv']]
            R2 = R2[(R2['zs']>zO)&(R2['zs']<zA)]
            R2['Nr_tod'] = Nrfl
            
            if geometry == 'CPC':
                args = (array, V_TOD, rO, Cg, R2['xo'], R2['yo'])
                ddx = (
                    CPC_Fxyz(R2['xs']+h, R2['ys'], R2['zs'],args) 
                    - CPC_Fxyz(R2['xs']-h, R2['ys'], R2['zs'], args)
                    )/(2*h)
                ddy = (
                    CPC_Fxyz(R2['xs'],R2['ys']+h,R2['zs'],args) 
                    - CPC_Fxyz(R2['xs'], R2['ys']-h, R2['zs'], args)
                    )/(2*h)
                ddz = (
                    CPC_Fxyz(R2['xs'],R2['ys'],R2['zs']+h,args) 
                    - CPC_Fxyz(R2['xs'],R2['ys'], R2['zs']-h, args)
                    )/(2*h)
            
            elif geometry == 'PB' and array=='F':
                ddx = 2*R2['xs']
                ddy = 2*R2['ys']
                ddz = -1
            
            elif geometry == 'PB' and array in ['A','B','C','D','E']:
                ddx = (
                    PB_Z(R2['xs']+h,R2['ys'],V_TOD,R2['xo'],R2['yo']) 
                    - PB_Z(R2['xs']-h,R2['ys'], V_TOD,R2['xo'],R2['yo'])
                    ) / (2*h)
                ddy = (
                    PB_Z(R2['xs'],R2['ys']+h,V_TOD,R2['xo'],R2['yo']) 
                    - PB_Z(R2['xs'],R2['ys']-h,V_TOD,R2['xo'],R2['yo'])
                    ) / (2*h)
                ddz = -1
            else:
                ddx = np.nan
                ddy = np.nan
                ddz = np.nan
                
            #Calculating perfect reflections
            nn = (ddx**2+ddy**2+ddz**2)**0.5
            (nx, ny, nz) = (ddx/nn, ddy/nn, ddz/nn)
            sc = nx*R2['uxn'] + ny*R2['uyn'] + nz*R2['uzn']
            uxrp = R2['uxn'] - 2*sc*nx
            uyrp = R2['uyn'] - 2*sc*ny
            uzrp = R2['uzn'] - 2*sc*nz
            
            #Adding errors to reflections
            if refl_error:
                R2['uxr'],R2['uyr'],R2['uzr'] = add_reflection_error(uxrp,uyrp,uzrp)
            else:
                R2['uxr'],R2['uyr'],R2['uzr'] = uxrp, uyrp, uzrp          #No reflection errors
            
            #Update for next iteration
            R2['xn'] = R2['xs']
            R2['yn'] = R2['ys']
            R2['zn'] = R2['zs']
            R2['uxn'] = R2['uxr']
            R2['uyn'] = R2['uyr']
            R2['uzn'] = R2['uzr']
            
            rays_in = sum(R2f['hit_rcv'])
            # if (rays_in==rays_ant)or(Nrfl==10)or(abs(rays_in-rays_ant)/rays_in < 0.001):
            if ((rays_in==rays_ant) and (Nrfl>4)) or (Nrfl==10):
                break
            else:
                Nrfl+=1
                rays_ant = rays_in
                
        # Getting the result back
        R2f['zs'] = R2f['zs'] + zV
        R2f['zr'] = R2f['zr'] + zV
        R2 = R1.copy()
        for x in ['xs','ys','zs','xr','yr','zr','uxr','uyr','uzr','hit_rcv','Nr_tod']:
            R2[x]=R2f[x]

        R2["hit_rcv"] = R2["hit_rcv"].convert_dtypes()
        R2.fillna({'hit_rcv':False},inplace=True)
        R2.fillna({'Nr_tod': 0}, inplace=True)

        return R2


def add_reflection_error(
        uxi: pd.Series,
        uyi: pd.Series,
        uzi: pd.Series,
        sigma_se: float = 2.02e-3
    ) -> tuple[pd.Series,pd.Series, pd.Series]:
    """
    This function is used to add reflection errors to any ray/mirror interaction
    
    Parameters
    ----------
    uxi,uyi,uzi : [-] Series containing the reflected vector (assuming perfect mirror)
    sigma_se    : [rad] Standard desviation of reflected rays.
    Returns
    -------
    uxf,uyf,uzf : [-] Series containing the reflected vector (including error)

    """    

    #Sigma in rad
    N_rays = len(uxi)
    
    #Generating the random values for errors
    R_theta = np.random.uniform(size=N_rays)
    R_phi   = np.random.uniform(size=N_rays)
    
    phi_se   = 2*np.pi*R_phi
    theta_se = ((-2*sigma_se**2)*np.log(1-R_theta))**0.5
    tan_se = np.tan(theta_se)
    sinphi = np.sin(phi_se)
    cosphi = np.cos(phi_se)
    
    #Cross product between (uxb,uyb,uzb) x (1,0,0) to get an arbitrary perpendicular vector to ub
    #uxt, uyt, uzt = (0, uzi, -uyi)
    
    #Rotated vector that is perpendicular to ub
    uxr = -sinphi * (uyi**2 + uzi**2)
    uyr =  uzi*cosphi + uxi*uyi*sinphi
    uzr = -uyi*cosphi + uxi*uzi*sinphi
    
    #Vector including mirror errors
    uxe, uye, uze = uxr*tan_se, uyr*tan_se, uzr*tan_se
    uf_mod = ( (uxi+uxe)**2 + (uyi+uye)**2 + (uzi+uze)**2 )**0.5
    
    #Final vector including errors and normalized
    uxf = (uxi+uxe) / uf_mod
    uyf = (uyi+uye) / uf_mod
    uzf = (uzi+uze) / uf_mod
    
    return (uxf,uyf,uzf)


def get_eta_attenuation(R1: pd.DataFrame) -> pd.Series:
    """
    Function to obtain the attenuation efficiency.
    It requires the total distance from heliostats to TOD.
    If the distance from HB to TOD is unknown (xc,yc,zc not calculated yet) it is assumed zero.

    Parameters
    ----------
    R1 : pandas DataFrame
    Returns
    -------
    Eta_att : pandas Series
        Pandas series with Eta_att grouped by heliostat ID.

    """
    
    # Distance [km] from heliostats to HB (d1) and from HB to TOD (d2)
    d1  = ((
        (R1['xi']-R1['xb'])**2
        +(R1['yi']-R1['yb'])**2
        +(R1['zi']-R1['zb'])**2)**0.5
    )
    d2  = ((
        (R1['xb']-R1['xc'])**2
        +(R1['yb']-R1['yc'])**2
        +(R1['zb']-R1['zc'])**2)**0.5 if 'xc' in R1 else 0.
    )
    R1['dray'] = d1 + d2
    d = R1.groupby('hel').mean()['dray']/1e3
    Eta_att = 1 - (0.006789 + 0.1046*d - 0.017*d**2 + 0.002845*d**3)
    return Eta_att



def heliostat_selection(
        CST: dict,
        HSF: SolarField,
        HB: HyperboloidMirror,
        TOD: TertiaryOpticalDevice,
        verbose: bool = True
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:

    type_shdw = CST['type_shdw'] if 'type_shdw' in CST else 'None'
    eta_hbi = HB.eta_hbi.get_value("-")
    A_h1 = CST["A_h1"].get_value("m2")
    lat = CST["lat"]
    lng = CST["lng"]

    # loading dataset
    R0, SF = HSF.load_dataset(save_plk=True)
    N_max   = len(R0['hel'].unique())
    if verbose:
        print("Rays dataset loaded")

    R1 = HB.mcrt_direct(R0, refl_error=True)
    if verbose:
        print("Interceptions with HB calculated")
    
    R1['hel_in'] = True         #Considering all heliostat at beginning
    R1['hit_hb'] = True         #Considering all rays hitting the HB
    R2 = TOD.mcrt_solver(R1)
    if verbose:
        print("Interceptions with TOD calculated")
    
    #Calculating efficiencies that does not depend on HB (TOD efficiencies)
    SF['Eta_hbr'] = eta_hbi * np.ones(N_max)
    SF['Eta_tdi'] = (
        R2[R2['hit_tod']].groupby('hel').count()['xb'] 
        / R2.groupby('hel').count()['xb']
    )
    N_avg   = R2[(R2['hit_rcv'])&(R2['hit_tod'])].groupby('hel').mean()['Nr_tod']
    SF['Eta_tdr'] = (
        (CST["eta_rfl"]**N_avg) 
        * R2[(R2['hit_rcv'])&(R2['hit_tod'])].groupby('hel').count()['xb'] 
        / R2[R2['hit_tod']].groupby('hel').count()['xb']
    )
    
    #Calculating the efficiencies that depend on HB
    R1['hel_in'] = True                 #Considering all heliostat at beginning
    R1['hit_hb'] = True                 #Considering all rays hitting the HB
    
    suav    = 0.7
    Nit = 1
    N_ant = N_max
    N_an2 = 0
    while True:     #loop to make converge the number of heliostats and hyperboloid size

        HB.update_geometry(R1)
        hb_rmin = HB.rmin.get_value("m")
        hb_rmax = HB.rmax.get_value("m")
        R1['hit_hb'] = ( (R1['rb']>hb_rmin) & (R1['rb']<hb_rmax) )

        # Altitude = (90° - lat). Azimuth = 0.
        if type_shdw == 'None' or type_shdw == 'point':
            SF = HB.shadow_point( (90. - lat), 0., SF )
        else:
            SF = HB.shadow_simple(
                lat=lat,
                lng=lng,
                type_shdw=type_shdw,
                SF=SF
            )
        
        #Getting the values for efficiencies and radiation fluxes
        SF['Eta_hbi'] = (R1.groupby('hel').sum()['hit_hb'] / R1.groupby('hel').count()['xb'])
        SF['Eta_att'] = get_eta_attenuation(R1)
        SF['Eta_hel'] = SF['Eta_blk'] * SF['Eta_cos'] * SF['Eta_att'] * CST['eta_rfl']
        SF['Eta_TOD'] = SF['Eta_tdi'] * SF['Eta_tdr']
        SF['Eta_BDR'] = SF['Eta_hbi'] * SF['Eta_hbr'] * SF['Eta_TOD']
        SF['Eta_SF']  = SF['Eta_hel'] * SF['Eta_BDR']
        SF['Q_h1']    = (SF['f_sh'] * SF['Eta_SF'] * CST['Gbn'] * A_h1 * 1e-6)
        SF.sort_values(by='Q_h1',ascending=False,inplace=True)
        Q_acc    = SF['Q_h1'].cumsum()
        
        #Getting the number of heliostats required and the list of heliostats
        N_hel0  = len( Q_acc[ Q_acc < CST['P_SF'] ] ) + 1
        N_hel   = int(np.ceil( suav*N_ant + (1-suav)* N_hel0 ))    #Attenuation factor
        if N_an2==N_hel:
            N_hel = int((N_hel+N_ant)/2)    #In case we are in a loop
            
        hlst    = Q_acc.iloc[:N_hel].index
        
        #Updating the heliostats selected
        SF['hel_in'] = SF.index.isin(hlst)
        R1['hel_in'] = R1['hel'].isin(hlst)
        Etas = SF[SF['hel_in']].mean()

        # Writing the results for partial iteration
        if verbose:
            hb_area = HB.area.get_value("m2")
            
            text_r = '\t'.join('{:.4f}'.format(x) for x in [
                Nit, eta_hbi, N_hel, hb_area,
                Etas['Eta_hbi'], Etas['Eta_cos'], Etas['Eta_tdi'], Etas['Eta_tdr'],
                Etas['Eta_TOD'], Etas['Eta_BDR'], Etas['Eta_SF'], hb_rmin, hb_rmax
            ])+'\n'
            print(text_r[:-2])
        
        #Checking if even with max heliostat we do not have enough power
        if N_hel == N_max:
            status = 'Nmx'
            break
        
        #Comparing with previous iteration
        if N_ant==N_hel:
            status = 'OK'
            break
        else:
            N_ant, N_an2 = N_hel, N_ant
        
        #Checking if we reach the maximum number of iterations
        if Nit == 50:
            status = 'NC'
            break
        else:
            Nit += 1
    
    R2['hel_in'] = R2['hel'].isin(hlst)
    
    return R2, Etas, SF, status


def optical_efficiencies(
        plant: PlantCSPBeamDownParticle,
        R2: pd.DataFrame,
        SF: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Function to obtain the optical efficiencies after a BDR simulation.

    Parameters
    ----------
    CST : dict.
        Characteristics of CST plant
    R2 : pandas DataFrame
        Ray dataset AFTER TOD simulation
    SF : pandas DataFrame
        Solar field heliostats AFTER simulations

    Returns
    -------
    SF : pandas DataFrame
        Same as SF input but including efficiencies

    """
    
    Gbn = plant.Gbn.get_value("W/m2")
    A_h1 = plant.Ah1.get_value("m2")
    eta_rfl = plant.eta_rfl.get_value("-")
    
    SF2 = R2.groupby('hel')[['hel_in','hit_hb','hit_tod','hit_rcv']].sum()
    SF['Eta_att'] = get_eta_attenuation(R2)
    SF['Eta_hbi'] = SF2['hit_hb']/SF2['hel_in']
    SF['Eta_tdi'] = SF2['hit_tod']/SF2['hit_hb']
    Nr_tod = R2[R2['hit_rcv']].groupby('hel')['Nr_tod'].mean()
    SF['Eta_tdr'] = (SF2['hit_rcv']/SF2['hit_tod']) * eta_rfl**Nr_tod
    SF['Eta_hel'] = SF['Eta_cos'] * SF['Eta_blk'] * SF['Eta_att'] * eta_rfl
    SF['Eta_BDR'] = (eta_rfl * SF['Eta_hbi']) * (SF['Eta_tdi'] * SF['Eta_tdr'])
    SF['Eta_SF']  = SF['Eta_hel'] * SF['Eta_BDR']
    SF['Q_h1']  = ( SF['Eta_SF']*Gbn*A_h1*1e-6 )
    SF['Q_pen'] = SF['Q_h1']*SF['f_sh']
    SF.sort_values(by='Q_pen',ascending=False,inplace=True)
    return SF

#  OLD FUNCTIONS #########

def CST_BaseCase(**kwargs) -> dict[str, Any]:
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. They can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    # Environment conditions
    CST['Gbn']   = 950                # Design-point DNI [W/m2]
    CST['day']   = 80                 # Design-point day [-]
    CST['omega'] = 0.0                # Design-point hour angle [rad]
    CST['lat']   = -23.               # Latitude [°]
    CST['lng']   = 115.9              # Longitude [°]
    CST['T_amb'] = 300.               # Ambient Temperature [K]
    CST['type_weather'] = 'TMY'       # Weather source (for CF) Accepted: TMY, MERRA2, None
    CST['file_weather'] = None        # Weather source (for CF)

    # Characteristics of Solar Field
    CST['eta_rfl'] = 0.95                     # Includes mirror refl, soiling and refl. surf. ratio. Used also for HB and TOD
    CST['A_h1']    = 2.92**2                  # Area of one heliostat
    CST['N_pan']   = 1                        # Number of panels per heliostat
    CST['err_tot'] = 0.002                    # [rad] Total reflected error
    CST['type_shdw'] = 'simple'              # Type of shadow modelling

    # Characteristics of BDR and Tower
    CST['zf']       = 35.               # Focal point (where the rays are pointing originally)
    CST['fzv']      = 0.88              # Position of HB vertix (fraction of zf)
    CST['rmin']     = 0.                # Inner radius of HB mirror
    CST['rmax']     = 20.               # Outer radius of HB mirror
    
    CST['Type']     = 'PB'              # Type of TOD, could be CPC, PB or None
    CST['Array']    = 'A'               # Number of TODs and number of polygon vertex
    CST['Arcv']     = 20.               # Area of receiver (outlet of TOD)
    CST['Cg_TOD']   = 2.0               # Concentration ratio on TOD
    CST['xrc']      = 0.                # Second focal point (TOD receiver)
    CST['yrc']      = 0.                # Second focal point (TOD receiver)
    CST['zrc']      = 10.0              # Second focal point (Height of TOD aperture)

    ### Receiver and Storage Tank characteristics
    CST['P_rcv']  = 10.               # [MWth] Initial target for Receiver nominal power
    CST['type_rcvr'] = 'HPR_0D'       # [-] model for Receiver
    CST['Q_mx']   = 3.0               # [MW/m2] Maximum radiation flux on receiver
    CST['Qavg']   = 0.5               # [MW/m2] Average radiation flux on receiver (initial guess)
    CST['T_pC']   = 950               # [K] Particle temperature in cold tank
    CST['T_pH']   = 1200              # [K] Particle temperature in hot tank
    CST['TSM']    = 'CARBO'           # [-] Thermal Storage Material
    CST['tz']     = 0.05              # [m] Thickness of material on conveyor belt
    CST['T_stg']   = 8.               # [hrs] Hours of storage
    CST['SM']      = 2.0              # [-] Initial target for solar multiple
    CST['HtD_stg'] = 0.5              # [-] height to diameter ratio for storage tank
    
    # Receiver and Power Block efficiencies
    # CST['P_el']   = 10.0               #[MW] Target for Net Electrical Power
    CST['T_pb_max'] = 875 + 273.15     #[K] Maximum temperature un power block cycle
    CST['Ntower'] = 1                  #[-] Number of towers feeding one power block 
    CST['eta_pb'] = 0.50               #[-] Power Block efficiency target (initial value) 
    CST['eta_sg'] = 1.00               #[-] Storage efficiency target (assumed 1 for now)
    CST['eta_rcv'] = 0.75              #[-] Receiver efficiency target (initial value)   
    
    for key, value in kwargs.items():
        CST[key] = value
    
    return CST



def HB_cooling_calc(
        qin: float,
        eta_rfl: float = 0.95,
    ) -> list[float]:
    
    FoS      = 2.0                           #Factor of Safety
    A_s      = 1.                            #Calculations per m2
    dT_adm   = 100-30                        #Admisible delta T [°C]
    
    #Glass properties
    t_glss   = 4e-3
    rho_glss = 2500
    k_glass  = 1000
    #Silver properties
    k_silver = 429
    rho_silver = 10490
    t_silver = 0.16e-3
    #Aluminum properties
    k_al   = 235
    rho_al = 2700.
    t_al   = 1e-3                       #[m] thickness of aluminium back sheet
    #Glue properties (SYLGARD™ 170 silicone elastomer)
    k_glue = 0.4                        #[W/m2K] Ben-Zvi, Seagal, Epstein, 2009
    t_glue = 0.5e-3                     #[m] Assumed
    rho_glue = 1370                     #[kg/m3]. Specific gravity of 1.37 indicated in technical data
    #Fin properties
    hc       = 5                        #Natural HTC, conservative value, no wind
    k_fin    = 235                      #Fin conductivity (Aluminium)
    rho_fin  = 2700                     #Fin density (Aluminium)
    D_fin  = 0.01                        #[m]
    L_fin  = 0.20                        #[m]
    
    m = (4*hc/(k_fin*D_fin))**0.5
    Lc = L_fin + D_fin/4.
    eta_fin = np.tanh(m*Lc) / (m*Lc)    #Fin Efficiency
    
    q_dis   = FoS * qin * (1-eta_rfl) * 1e3      #Heat to dissipate [W/m2]
    
    dT_glue = q_dis * t_glue / k_glue
    dT_al   = q_dis * t_al / k_al
    dT_fin  = dT_adm - dT_glue - dT_al
    
    A_fin_b = np.pi*D_fin**2/4.                     #Base area of fin
    A_fin1  = np.pi*D_fin*(L_fin + D_fin/4.)        #Dissipating heat fin
    V_fin1  = A_fin_b * L_fin                       #Volume of fin
    
    #Fins per sqm
    N_fin = (q_dis/(hc*dT_fin) - A_s) / (eta_fin*A_fin1 - A_fin_b)
    #If number of fins is less than zero, it means no fins are required, so set to zero
    if hasattr(N_fin, "__len__"):
        N_fin[N_fin<0] = 0
    else:
        if N_fin<0:
            N_fin = 0
    
    rAf = N_fin * A_fin_b
    M_fin    = V_fin1*N_fin*rho_fin         # Extra weight from fins per m2
    M_mirror = t_glss*rho_glss + t_silver*rho_silver + t_glue*rho_glue + t_al*rho_al  # Mirror weight per m2
    M_beams  = 0.025*0.025*2 * rho_al         # Assuming 2.5x2.5cm2 aluminium beams every meter
    M_total  = (M_mirror + M_fin + M_beams)   # [kg/m2]
    
    rMf = M_fin/M_total
    # rM2 = M_fin/(M_mirror+M_beams)          #Percentage of extra material
    return [N_fin,M_fin,M_mirror,M_beams,M_total,rAf,rMf]



def HB_mass_cooling(
        HB: HyperboloidMirror,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        Gbn: float = 950,
        A_h1: float = 2.92**2,
        full_results: bool = False
    ) -> tuple[tuple,tuple] | tuple:

    rmin = HB.rmin.get_value("m")
    rmax = HB.rmax.get_value("m")
    eta_rfl = HB.eta_rfl.get_value("-")

    Etas = SF[SF['hel_in']].mean()
    hlst = SF[SF.hel_in].index
    N_hel = len(hlst)
    
    out  = R2[(R2['hel_in'])&(R2['hit_hb'])]
    xmin = out['xb'].min()
    xmax = out['xb'].max()
    ymin = out['yb'].min()
    ymax = out['yb'].max()
    Nx = 100
    Ny = 100
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Nx
    dA = dx*dy
    Q_HB,X,Y = np.histogram2d(
        out['xb'], out['yb'], 
        bins = [Nx,Ny],
        range = [[xmin, xmax], [ymin, ymax]],
        density = False
    )
    X2, Y2 = np.meshgrid(X, Y)
    Xc = np.array(
        [ [(X2[i,j] + X2[i,j+1])/2 for j in range(Ny)] for i in range(Nx) ]
    )
    Yc = np.array(
        [ [(Y2[i,j]+Y2[i+1,j])/2 for j in range(Ny)] for i in range(Nx) ]
    )
    
    def HB_dAs(
            X: np.ndarray,
            Y: np.ndarray,
            HB: HyperboloidMirror
        ) -> np.ndarray:
        
        zf = HB.zf.get_value("m")
        fzv = HB.fzv.get_value("-")
        zrc = HB.zrc.get_value("m")
        yrc = HB.yrc.get_value("m")
        zfh,zvh = zf-zrc, fzv*zf-zrc
        fvh = zvh/zfh
        c = zfh/2
        a = c*( 2*fvh - 1 )
        b = 2*c*np.sqrt(fvh - fvh**2)
        R = np.sqrt(X**2+Y**2)
        Z = np.sqrt( R**2/b**2 + 1 )*a
        dA = np.sqrt( (a/b)**4 * (R/Z)**2 + 1)
        return dA
    
    dS_HB = HB_dAs(Xc, Yc, HB) * dx * dy
    Rc = np.sqrt( Xc**2 + Yc**2 )
    S_HB = dS_HB[(Rc>rmin)&(Rc<rmax)].sum()
    Fbin = (
        eta_rfl*Etas['Eta_cos']*Etas['Eta_blk']
        * Gbn * A_h1 * N_hel
        /(1e3*dS_HB*len(out))
    ) 
    Q_HB = Fbin * Q_HB
    
    Q_HB_tot = (Q_HB*dS_HB).sum()               #kW
    Q_HB_abs = Q_HB_tot*(1-eta_rfl)      #kw
    Q_HB_cool_avg = Q_HB_abs/S_HB               #kW/m2
    Q_HB_worst = Q_HB.max()*(1-eta_rfl)  #kW/m2
    dT_mirr   = 70.                             #K
    
    HTC_max = 1e3 * Q_HB_worst / dT_mirr             #W/m2-K
    HTC_avg = 1e3 * Q_HB_cool_avg / dT_mirr          #W/m2-K
    
    [N_fin,M_fin,M_mirror,M_beams,M_total,rAf,rMf] = HB_cooling_calc(Q_HB,eta_rfl)

    N_fin_tot = (N_fin*dS_HB)[(Rc>rmin) & (Rc<rmax)].sum()
    M_fin_tot = (M_fin*dS_HB)[(Rc>rmin) & (Rc<rmax)].sum() /1e3   #[tonnes]
    M_mirr_tot = M_mirror*S_HB / 1e3                                          #[tonnes]
    M_str_tot = M_beams*S_HB / 1e3                                            #[tonnes]
    M_HB_tot  = M_fin_tot + M_mirr_tot + M_str_tot                            #[tonnes]
    
    
    if full_results:
        return [M_fin_tot,M_mirr_tot,M_str_tot,M_HB_tot], [Q_HB,N_fin,M_fin]
    else:
        return [M_fin_tot,M_mirr_tot,M_str_tot,M_HB_tot]



def PB_Z(
        x: float | np.ndarray,
        y: float | np.ndarray,
        V: int,
        xo: float,
        yo: float
    ) -> float | np.ndarray:
    """
    Function that return the z position in PB concentrator for a given (x,y) position

    Parameters
    ----------
    x,y : floats, lists, numpy arrays
        Position(s) where the correspondent z value is needed.
    V : integer
        Number of sides of polygon PB
    xo,yo : floats
        Center of the PB.

    Returns
    -------
    z : floats, lists, numpy arrays
        Position(s) of PB array, considering zo=0.

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


def TOD_XY_R(
        ri: float,
        H: float,
        V: int,
        N: int,
        xo: float,
        yo: float,
        zo: float
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    For a TOD with given parameters, obtain the points to form the TOD curve.

    Parameters
    ----------
    ri : float
        Radius of TOD transverse area (could be rA, rO, etc).
    H : float
        PB height.
    V : integer
        Number of sides of polygon TOD.
    N : integer
        Number of TOD in the array.
    xo, yo, zo : floats
        Center of polygon TOD.

    Returns
    -------
    xx : list
        Set of x points that form the PB aperture shape.
    yy : list
        Set of y points that form the PB aperture shape.

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


def PB_direct(
        R1: pd.DataFrame,
        TOD: dict,
        CST: dict
    ) -> pd.DataFrame:
    
    #This function is only useful for Polygon Paraboloid.
    def Fk(ki,R2,V):
        xs = R2['xn'] + ki*R2['uxn']
        ys = R2['yn'] + ki*R2['uyn']
        zs = R2['zn'] + ki*R2['uzn']
        zp = PB_Z(xs,ys,V,R2['xo'],R2['yo'])
        return zp - zs
       
    Array,N_PB,V_PB,H_PB,rA,rO,x0,y0 = [ TOD[x] for x in ['Array', 'N', 'V', 'H', 'rA', 'rO', 'x0', 'y0'] ]
    
    if Array=='F':
        x0,y0 = x0[0],y0[0]
    
    # zV = CST['zrc'] - rA**2         #Height of Paraboloid focus
    # zA = CST['zrc']
    # zO = zV + rO**2
    
    #!!!
    zV = CST['zrc'] - rA**2
    zA = rA**2
    zO = rO**2
    
    #If there is no TOD, the function return R2
    if Array=='N':
        R2 = R1.copy()
        R2['hit_rcv'] = (R2['rc'] < rA) & (R2['hit_hb'])
        R2['hit_tod'] = R2['hit_hb']
        R2['Nr_tod'] = 0
        R2['xr'] = R2['xc'];   R2['yr'] = R2['yc'];   R2['zr'] = R2['zc'];
        R2['uxr'] = R2['uxb']; R2['uyr'] = R2['uyb']; R2['uzr'] = R2['uzb'];
        return R2
    
    ###########################################
    #Getting the rays that enter the TOD
    if Array=='F':
        R1['hit_tod'] = (R1['rc']<rA)&(R1['hit_hb'])
        R1['Npolygon'] = np.where(R1['hit_tod'],1,0)
        R1['xo'] = x0; R1['yo'] = y0;
        
    elif Array in ['A','B','C','D','E']:
        R1['Npolygon'] = 0
        R1['xo'] = np.nan
        R1['yo'] = np.nan
        
        xc = R1[R1['hit_hb']]['xc']
        yc = R1[R1['hit_hb']]['yc']
        xo,yo,Npolygon = R1['xo'], R1['yo'], R1['Npolygon']
        for i in range(N_PB):
            hit = PB_Z( xc, yc, V_PB, x0[i],y0[i]) < zA
            aux = hit[hit]
            xo.update(x0[i]*aux)
            yo.update(y0[i]*aux)
            Npolygon.update((i+1)*aux)
            xc = xc[~hit]
            yc = yc[~hit]
        R1['hit_tod'] = np.where(R1['Npolygon']>0,True,False)
    
    ############################################
    #This will be the initial df
    R2 = R1[R1['hit_tod']][['xc','yc','zc','uxb','uyb','uzb','xo','yo','hit_tod','Npolygon']]
    R2.rename(columns={'xc':'xn','yc':'yn','zc':'zn','uxb':'uxn','uyb':'uyn','uzb':'uzn'},inplace=True)
    R2['xs'] = 0;   R2['ys'] = 0;    R2['zs'] = 0
    R2['hit_rcv'] = False
    R2['Nr_tod'] = 0
    R2['zn'] = zA
    #This will be the final df
    R2f = R2.copy()
    R2f['xr'] = np.nan;   R2f['yr'] = np.nan;    R2f['zr'] = np.nan
    R2f['uxr'] = np.nan;  R2f['uyr'] = np.nan;   R2f['uzr'] = np.nan
    # Calculating interceptions with surface (explicit quadratic expression)
    Nrfl = 1
    rays_ant = 0
    
    while True:
        
        # Getting the intercept
        if Array=='F':
            p2 = R2['uxn']**2 + R2['uyn']**2
            p1 = 2*(R2['xn']*R2['uxn'] + R2['yn']*R2['uyn']) - R2['uzn']
            p0 = R2['xn']**2 + R2['yn']**2 - R2['zn']
            ks   = np.where(abs(p2)>1e-8,( -p1 + (p1**2 - 4*p2*p0)**0.5 ) / (2*p2), -p0/p1)
        elif Array in ['A','B','C','D','E']:
            tol=1e-4        #Solving the non-linear equation
            ki = ( zO - R2['zn'] )/R2['uzn']
            R2i = R2
            ks = ki.copy()
            h=1e-6
            for i in range(20):
                dFk = ( Fk(ki+h/2,R2i,V_PB) - Fk(ki-h/2,R2i,V_PB) ) / h
                kj  = ki - Fk(ki,R2i,V_PB)/dFk
                err_k = (ki-kj).abs()
                errmax = err_k.max()
                ks.update(kj[err_k<tol])
                R2i = R2i[err_k>tol]
                ki = kj[err_k>tol]
                
                if errmax<tol or len(R2i)==0: break
                if len(err_k)/len(R2)<0.001: break
        
        #Calculating the position on the surface
        R2['xs'] = R2['xn'] + ks*R2['uxn']
        R2['ys'] = R2['yn'] + ks*R2['uyn']
        R2['zs'] = R2['zn'] + ks*R2['uzn']
        R2['rs'] = (R2['xs']**2+R2['ys']**2)**0.5
        R2['hit_rcv'] = R2['zs']<=zO
        
        #Rays that go out the system are updated
        R2out = R2[(R2['zs']>zA)|R2['zs'].isnull()]
        R2f.update(R2out)

        #Rays that already hit the outlet are updated
        R2rcv = R2[R2['hit_rcv']].copy()
        if len(R2rcv)>0:
            kr = (zO - R2rcv['zs'])/R2rcv['uzn']
            R2rcv['xr'] = R2rcv['xs']+kr*R2rcv['uxn']
            R2rcv['yr'] = R2rcv['ys']+kr*R2rcv['uyn']
            R2rcv['zr'] = R2rcv['zs']+kr*R2rcv['uzn']
            R2rcv['rr'] = (R2rcv['xr']**2+R2rcv['yr']**2)**0.5
            R2rcv['uxr'] = R2rcv['uxn']
            R2rcv['uyr'] = R2rcv['uyn']
            R2rcv['uzr'] = R2rcv['uzn']
            R2f.update(R2rcv)
        
        #We calculate the position for the reflected rays
        R2 = R2[(R2['zs']>zO)&(R2['zs']<zA)]
        R2['Nr_tod'] = Nrfl
        
        if Array=='F':
            ddx = 2*R2['xs'];      ddy = 2*R2['ys'];     ddz = -1
        elif Array in ['A','B','C','D','E']:
            ddx = (PB_Z(R2['xs']+h,R2['ys'],V_PB,R2['xo'],R2['yo']) - PB_Z(R2['xs']-h,R2['ys'], V_PB, R2['xo'], R2['yo'])) / (2*h)
            ddy = (PB_Z(R2['xs'],R2['ys']+h,V_PB,R2['xo'],R2['yo']) - PB_Z(R2['xs'],R2['ys']-h,V_PB, R2['xo'], R2['yo'])) / (2*h)
            ddz = -1
            
        nn = (ddx**2+ddy**2+ddz**2)**0.5
        nx, ny, nz = ddx/nn, ddy/nn, ddz/nn
        sc = nx*R2['uxn'] + ny*R2['uyn'] + nz*R2['uzn']
        R2['uxr'] = R2['uxn'] - 2*sc*nx
        R2['uyr'] = R2['uyn'] - 2*sc*ny
        R2['uzr'] = R2['uzn'] - 2*sc*nz
        kr = (zO - R2['zs'])/R2['uzr']
        R2['xr'] = R2['xs'] + kr*R2['uxr']
        R2['yr'] = R2['ys'] + kr*R2['uyr']
        R2['zr'] = R2['zs'] + kr*R2['uzr']
        
        if Array=='F':
            zout = (R2['xr']**2+R2['yr']**2)
            R2['hit_rcv'] = zout <= zO
        elif Array in ['A','B','C','D','E']:
            zout = PB_Z( R2['xr'] , R2['yr'], V_PB, R2['xo'],R2['yo'])
            R2['hit_rcv'] = zout <= zO
            
        R2rfl = R2[R2['hit_rcv']]  #Checking if the reflected ray hits the outlet
        R2f.update(R2rfl)
        
        #Update for next iteration
        # R2f['hit_rcv'] = (R2f['xr']**2+R2f['yr']**2)<zO
        # R2 = R2[(R2['rr']>zO**0.5)&(R2['rs']<zA**0.5)].copy()
        R2 = R2[(zout>zO)&(zout<zA)].copy()
        # R2 = R2[(zout>zO)&(R2['zs']<zA)].copy()
        R2['xn'] = R2['xs']; R2['yn'] = R2['ys']; R2['zn'] = R2['zs']
        R2['uxn'] = R2['uxr']; R2['uyn'] = R2['uyr']; R2['uzn'] = R2['uzr']
        
        rays_in = sum(R2f['hit_rcv'])
        if (rays_in==rays_ant)or(Nrfl==10)or(abs(rays_in-rays_ant)/rays_in < 0.001):
            break
        else:
            Nrfl+=1
            rays_ant = rays_in
            
    # Getting the result back
    R2f['zs'] = R2f['zs'] + zV
    R2f['zr'] = R2f['zr'] + zV
    
    R2 = R1.copy()
    for x in ['xs','ys','zs','xr','yr','zr','uxr','uyr','uzr','hit_rcv','Nr_tod']:
        R2[x]=R2f[x]
    R2['hit_rcv'].fillna(False,inplace=True)
    R2['Nr_tod'].fillna(0,inplace=True)
    
    return R2


#%% FINAL OPTICAL DEVICE: CPC AND PARABOLOID

def CPC_Fk(ks,args):
    R2, Array, r_out, Cg, V = args
    
    xn, yn, zn  = R2['xn'], R2['yn'], R2['zn']
    uxn, uyn, uzn  = R2['uxn'], R2['uyn'], R2['uzn']
    xo, yo = R2['xo'], R2['yo']
    
    tht = np.arcsin(1/Cg**0.5)
    xs =  xn + ks * uxn - xo
    ys =  yn + ks * uyn - yo
    zs =  zn + ks * uzn
    
    if Array == 'F':
        rs = (xs**2+ys**2)**0.5
    
    elif Array in ['A','B','C','D','E']:
        phi    = 360./V
        alpha  = np.degrees(np.arctan2(ys,xs)) % 360
        i      = np.floor(alpha/phi)                    #Line which the ray belongs
        mi      = np.tan(np.radians((i+0.5)*phi + 90.))  #Slope of line
        ni      = ys - mi*xs                              # Intercept
    
        phi1, phi2 = np.radians((i+1)*phi), np.radians(i*phi)
        xi = (-ni/2) * ( 1/(mi-np.tan(phi1)) + 1/(mi-np.tan(phi2)) )
        yi = mi*xi+ni               #(xi,yi) are the point in the center of the line
        rs = (xi**2+yi**2)**0.5     #rs should belong to CPC curve    
    
    a1 = ((rs+r_out)*np.cos(tht) + zs*np.sin(tht))**2
    a2 = 4*r_out*(1+np.sin(tht))
    a3 = zs*np.cos(tht) - rs*np.sin(tht) + r_out
    
    return a1 - a2*a3

################################################
#Function to check if the rays enter a surface
def CPC_enter( x, y, r, Array, V, xo, yo ):
    
    if Array == 'F':
        return (x-xo)**2+(y-yo)**2 < r**2
    
    elif Array in ['A','B','C','D','E']:
        xp, yp = x-xo, y-yo
        phi    = 360./V
        alpha  = np.degrees(np.arctan2(yp,xp)) % 360
        i      = np.floor(alpha/phi)        #Which line the point belong to
    
        #mi and ni are the slope and intercept of the line
        mi    = np.tan(np.radians((i+0.5)*phi + 90.))
        phii  = np.radians((i+0.5)*phi)
        xi,yi = r*np.cos(phii), r*np.sin(phii)
        ni    = yi - mi*xi
    
        xl = ni / (yp/xp - mi)
        yl = yp*xl/xp
        dl = (xl**2+yl**2)**0.5     #Maximum possible distance (defined by r)
        
        dp = (xp**2+yp**2)**0.5     #Distance from the point to the center
    
        return dl>dp
    
    else:
        return np.nan
    
#############################################
def CPC_Fxyz(x,y,z,args):
    Array, V, r_out, Cg, xo, yo = args
    tht = np.arcsin(1/Cg**0.5)
    
    if Array == 'F':
        r = ((x-xo)**2+(y-yo)**2)**0.5
        
    elif Array in ['A','B','C','D','E']:
        xp, yp = x-xo, y-yo
        phi    = 360./V
        alpha  = np.degrees(np.arctan2(yp,xp)) % 360
        i      = np.floor(alpha/phi)                     #Line which the ray belongs
        mi      = np.tan(np.radians((i+0.5)*phi + 90.))  #Slope of line
        ni      = yp - mi*xp                             # Intercept
    
        phi1, phi2 = np.radians((i+1)*phi), np.radians(i*phi)
        xi = (-ni/2) * ( 1/(mi-np.tan(phi1)) + 1/(mi-np.tan(phi2)) )
        yi = mi*xi+ni               #(xi,yi) are the point in the center of the line
        r = (xi**2+yi**2)**0.5      #r should belong to CPC curve
            
    a1 = ((r+r_out)*np.cos(tht) + z*np.sin(tht))**2
    a2 = 4*r_out*(1+np.sin(tht))
    a3 = z*np.cos(tht) - r*np.sin(tht) + r_out
    
    return a1 - a2*a3

#############################################
#%%% FUNCTIONS FOR PARABOLOID
#This function is only useful for Polygon Paraboloid
def PB_Fk(ki,R2,V):
    xs = R2['xn'] + ki*R2['uxn']
    ys = R2['yn'] + ki*R2['uyn']
    zs = R2['zn'] + ki*R2['uzn']
    zp = PB_Z(xs,ys,V,R2['xo'],R2['yo'])
    return zp - zs



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
    if eta_type == 'eta_acc':
        title = 'Heliostat Field. Optical efficiency per heliostat'
    if eta_type == 'eta_hbi':
        title = 'Heliostat Field. HB intercept efficiency per heliostat'
    if eta_type == 'eta_cos':
        title = 'Heliostat Field. Cosine efficiency per heliostat'
    if eta_type == 'eta_bdr':
        title = 'Heliostat Field. BDR efficiency per heliostat'
    if eta_type == 'eta_tdi':
        title = 'Heliostat Field. TOD intercept efficiency per heliostat'
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111)
    surf = ax1.scatter(
        R2['xi'],R2['yi'],
        s=s, c=R2[eta_type], cmap=cm.YlOrRd,
        vmax=(np.ceil(10*R2[eta_type].max())/10)
    )
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