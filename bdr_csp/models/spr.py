from dataclasses import dataclass
import cantera as ct
from bdr_csp import htc

@dataclass
class BlackboxModel():
    Fc: float = 2.57
    ab_p: float = 0.91
    em_p: float = 0.85
    air: ct.Solution = ct.Solution('air.yaml')
    HTC: str = 'NellisKlein'
    view_factor: float | None = None

    # former HTM_0D_blackbox
    def run_0D_model(
            self,
            temp: float,
            flux: float,
            temp_amb: float,
    ) -> dict[str,float]:
        """It returns an initial estimation of a BD-SPR's efficiency based on particle temperature and flux

        Args:
            temp (float): Particle temperature in K
            flux (float): Radiation flux in MW/m2
            temp_amb (float): Ambient temperature in K

        Returns:
            dict[str,float]: dictionary with 'eta_rcv', 'h_rad' and 'h_conv'
        """
        #retrieving data
        ab_p = self.ab_p
        em_p = self.em_p
        air = self.air
        HTC = self.HTC
        Fc = self.Fc
        view_factor = self.view_factor

        temp_sky = htc.temp_sky_simplest(temp_amb)
        air.TP = (temp+temp_amb)/2., ct.one_atm
        if HTC == 'NellisKlein':
            h_conv = Fc * htc.h_conv_NellisKlein(temp, temp_amb, 0.01, air)
        elif HTC == 'Holman':
            h_conv = Fc * htc.h_conv_Holman(temp, temp_amb, 0.01, air)
        elif HTC == 'Experiment':
            h_conv = Fc * htc.h_conv_Experiment(temp, temp_amb, 0.1, air)

        if view_factor is None:
            view_factor = 1.0        
        h_rad  = view_factor * em_p * htc.SIGMA_CONSTANT * (temp**4. - temp_sky**4.) / (temp - temp_amb)
        hcond = 0.833
        hrc = h_conv + h_rad + hcond
        q_loss = hrc * (temp - temp_amb)
        if flux<=0.:
            eta_rcv = 0.
        else:
            eta_rcv = (flux*1e6*ab_p - q_loss)/(flux*1e6)
        if eta_rcv<0:
            eta_rcv == 0.

        output = {
            "eta_rcv": eta_rcv,
            "h_rad": h_rad,
            "h_conv": h_conv,
            "q_loss": q_loss,
        }
        return output

