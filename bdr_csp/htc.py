import numpy as np
import cantera as ct

SIGMA_CONSTANT = 5.67e-8

def temp_sky_simplest(temp_amb: float) -> float:
    """simplest function to estimate sky temperature. It is just temp_amb-15.[K]

    Args:
        temp_amb (float): temperature in K

    Returns:
        float: sky temperature
    """
    return (temp_amb - 15.)

def h_conv_Holman(
        T_s: float,
        T_inf: float,
        L: float,
        fluid: ct.Solution
    ) -> float:
    """
    Correlation for natural convection in upper hot surface horizontal plate Holman
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = ( T_s + T_inf )/2
    fluid.TP = T_av, ct.one_atm
    mu = fluid.viscosity
    k = fluid.thermal_conductivity
    rho = fluid.density_mass
    cp = fluid.cp_mass
    alpha = k/(rho*cp)
    beta = 1./T_s
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


def h_conv_NellisKlein(
        T_s: float,
        T_inf: float,
        L: float,
        fluid: ct.Solution
    ) -> float:
    """
    Correlation for natural convection in upper hot surface horizontal plate (Nellis & Klein, 2012).
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = ( T_s + T_inf )/2
    fluid.TP = T_av, ct.one_atm
    mu = fluid.viscosity
    k = fluid.thermal_conductivity
    rho = fluid.density_mass
    cp = fluid.cp_mass
    alpha = k/(rho*cp)
    beta = 1./T_s
    visc = mu/rho
    Pr = visc/alpha
    g = 9.81
    Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
    C_lam  = 0.671 / ( 1+ (0.492/Pr)**(9/16) )**(4/9)
    Nu_lam = float(1.4/ np.log(1 + 1.4 / (0.835*C_lam*Ra**0.25) ) )
    C_tur  = 0.14*(1 + 0.0107*Pr)/(1+0.01*Pr)
    Nu_tur = C_tur * Ra**(1/3)
    Nu = (Nu_lam**10 + Nu_tur**10)**(1/10)
    h = (k*Nu/L)
    return h


def h_conv_Experiment(
        T_s: float,
        T_inf: float,
        L: float,
        fluid: ct.Solution
    ) -> float:
    """
    Correlation for natural convection in upper hot surface horizontal plate Holman
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = ( T_s + T_inf )/2
    fluid.TP = T_av, ct.one_atm
    mu = fluid.viscosity
    k = fluid.thermal_conductivity
    rho = fluid.density_mass
    cp = fluid.cp_mass
    alpha = k/(rho*cp)
    beta = 1./T_s
    visc = mu/rho
    Pr = visc/alpha
    g = 9.81
    Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
    Nu = 0.26*Ra**0.411*L**(-0.234)
    h = (k*Nu/L)
    return h