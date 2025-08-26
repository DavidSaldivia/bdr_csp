from antupy import Var
from antupy.props import Air

SIGMA_CONSTANT = 5.67e-8


def h_conv_Experiment(
        T_s: float,
        T_inf: float,
        L: float,
        P: float = 101325,
        fluid: Air = Air()
    ) -> float:
    """
    Correlation for natural convection in upper hot surface horizontal plate Holman
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = Var(( T_s + T_inf )/2, "K")
    mu = fluid.viscosity(T_av, P)
    k = fluid.k(T_av, P)
    rho = fluid.rho(T_av, P)
    cp = fluid.cp(T_av, P)
    alpha = k/(rho*cp)
    beta = 1./T_s
    visc = mu/rho
    Pr = (visc/alpha).v
    g = 9.81
    Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc.v**2
    Nu = 0.26*Ra**0.411*L**(-0.234)
    return (k*Nu/L)