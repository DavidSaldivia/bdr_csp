"""
module with a simple units manager
"""
import numpy as np
from typing import Iterable

CONVERSION_FUNDAMENTALS: dict[str,dict[str|None,float]] = {
    "adim" : {
        "-": 1e0,
        "": 1e0,
        " ": 1e0,
        "(-)": 1e0,
        "()": 1e0,
    },
    "length" : {
        "m": 1e0,
        "mm": 1e3,
        "km": 1e-3,
        "mi": 1e0/1609.34,
        "ft": 3.28084,
        "in": 39.3701,
    },
    "mass": {
        "kg": 1e0,
        "g": 1e3,
        "ton": 1e-3,
        "lb": 2.20462,
        "oz": 35.274,
    },
    "time": {
        "s": 1e0, "sec": 1e0,
        "min": 1e0/60,
        "h": 1e0/3600, "hr": 1e0/3600,
        "d": 1e0/(24*3600), "day": 1e0/(24*3600),
        "wk": 1e0/(24*3600*7), "week": 1e0/(24*3600*7),
        "mo": 1e0/(24*3600*30), "month": 1e0/(24*3600*30),
        "yr": 1e0/(24*3600*365), "year": 1e0/(24*3600*365),
    },
    "temperature": {
        "K": 1.0,
        "C": np.nan
    },
    "current": {
        "A": 1e0,
        "mA": 1e3,
        "kA": 1e-3,
    },
    "substance": {
        "mol": 1e0,
        "mmol": 1e3,
        "kmol": 1e-3,
    },
    "luminous_intensity": {
        "cd": 1e0,
        "lm": 1e0,
    },
}

CONVERSIONS_DERIVED: dict[str,dict[str|None,float]] = {
    "area" : {
        "m2": 1e0,
        "mm2": 1e6,
        "km2": 1e-6,
        "ha": 1e-4,
    },
    "volume": {
        "m3": 1e0,
        "L": 1e3,
        },
    "mass_flowrate": {
        "kg/s": 1e0,
        "g/s": 1e3,
        "kg/min": 60,
        "kg/hr": 3600,
    },
    "volume_flowrate": {
        "L/s": 1e0,
        "m3/s": 1e-3,
        "m3/min": 1e-3*60,
        "m3/hr": 1e-3*3600,
        "L/min": 60,
        "L/hr": 3600,
        "ml/s": 1e3,
    },
    "energy": {
        "J": 1e0,
        "kJ": 1e-3,
        "MJ": 1e-6,
        "Wh": 1e-3/3.6,
        "kWh": 1e-6/3.6,
        "MWh": 1e-9/3.6,
        "cal": 4.184e0,
        "kcal": 4.184e3,
    },
    "energy_flow": {
        "MW/m2": 1e0,
        "kJ/m2": 1e3,
        "J/m2": 1e6,
    },
    "power": {
        "W": 1e0,
        "kW": 1e-3,
        "MW": 1e-6,
        "J/h": 3.6e6, "J/hr": 3.6e6,
        "kJ/h": 3.6e0, "kJ/hr": 3.6e0,
        "MJ/h": 3.6e-3, "MJ/hr": 3.6e-3,
    },
    "pressure": {
        "Pa": 1e0,
        "bar": 1e-5,
        "psi": 1e0/6894.76,
        "atm": 1e0/101300,
    },
    "velocity": {
        "m/s": 1e0,
        "km/hr": 3.6,
        "mi/hr": 2.23694,
        "ft/s": 3.28084,
    },
    "angular": {
        "rad": 1e0,
        "deg": 180./np.pi,
    },
    "cost" : {
        "AUD": 1e0,
        "USD": 1.4e0,
    },
#-------------------
    "density": {
        "kg/m3": 1e0,
        "g/cm3": 1e-3,
    },
    "specific_heat": {
        "J/kgK": 1e0, "J/kg-K": 1e0,
        "kJ/kgK": 1e-3, "kJ/kg-K": 1e-3,
    },
}

CONVERSIONS : dict[str,dict[str|None,float]] = CONVERSIONS_DERIVED | CONVERSION_FUNDAMENTALS
UNIT_TYPES: dict[str|None, str] = dict()
for type_unit in CONVERSIONS.keys():
    for unit in CONVERSIONS[type_unit].keys():
        UNIT_TYPES[unit] = type_unit

class Variable():
    """
    Class to represent parameters and variables in the system.
    It is used to store the values with their units.
    If you have a Variable instance, always obtain the value with the get_value method.
    In this way you make sure you are getting the value with the expected unit.
    get_value internally converts unit if it is possible.
    """
    def __init__(
            self,
            value: float,
            unit: str | None = None,
            type: str = "scalar"
        ):
        self.value = value
        self.unit = unit if unit is not None else "None"
        self.type = type

    def __mul__(self, other):
        """ Overloading the multiplication operator. """
        if isinstance(other, Variable):
            return Variable(self.value * other.value, [self.unit, other.unit])
        elif isinstance(other, (int, float)):
            return Variable(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def __rmul__(self, other):
        """ Overloading the multiplication operator. """
        if isinstance(other, Variable):
            return Variable(self.value * other.value, [other.unit, self.unit])
        elif isinstance(other, (int, float)):
            return Variable(self.value * other, self.unit)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

    def get_value(self, unit: str | None = None):
        """ Get the value of the variable in the requested unit.
        If the unit is not compatible with the variable unit, an error is raised.
        If the unit is None, the value is returned in the variable unit.
        """
        
        if unit is None:
            unit = self.unit

        if self.unit == unit:
            return self.value
        
        if UNIT_TYPES[unit] == UNIT_TYPES[self.unit]:
            return self.value * conversion_factor(self.unit, unit)
        else:
            raise ValueError( f"Variable unit ({self.unit}) and wanted unit ({unit}) are not compatible.")

    def set_unit(self, unit: str | None = None):
        """ Set the unit of the variable. """
        if UNIT_TYPES[unit] == UNIT_TYPES[self.unit]:
            self.value = self.value * conversion_factor(self.unit, unit)
        else:
            raise ValueError(
                f"unit ({unit}) is not compatible with existing primary unit ({self.unit})."
            )

    @property
    def v(self) -> float:
        """ Property to obtain the value of the variable (in its primary unit). """
        return self.value
    
    @property
    def u(self) -> str:
        """ Property to obtain the primary unit of the variable. """
        return self.unit
    
    @property
    def units(self) -> str:
        """ Property to obtain the compatible units of the variable. """
        return UNIT_TYPES[self.u]
    
    def __repr__(self) -> str:
        return f"{self.value:} [{self.unit}]"

#-------------------------
class Array():
    """
    Similar to Variable() but for lists (iterators, actually).
    """
    def __init__(self, values: Iterable, unit: str | None = None):
        self.values = values
        self.unit = unit
        self.type = type

    def get_values(self, unit=None) -> Iterable:
        values = self.values
        if unit is None:
            unit = self.unit

        if self.unit == unit:
            return values
        
        if UNIT_TYPES[unit] == UNIT_TYPES[self.unit]:
            conv_factor = conversion_factor(self.unit, unit)
            values_out = [v*conv_factor for v in values]
            return values_out
        if self.unit != unit:
            raise ValueError(
                f"The variable used have different units: {unit} and {self.unit}"
                )
        return values

    def __repr__(self) -> str:
        return f"{self.values:} [{self.unit}]"


#-------------------------
def conversion_factor(unit1: str|None, unit2: str|None) -> float:
    """ Function to obtain conversion factor between units.
    The units must be in the UNIT_CONV dictionary.
    If they are units from different phyisical quantities an error is raised.
    """

    if UNIT_TYPES[unit1] == UNIT_TYPES[unit2]:
        type_unit = UNIT_TYPES[unit1]
        conv_factor = CONVERSIONS[type_unit][unit2] / CONVERSIONS[type_unit][unit1]
    else:
        raise ValueError(f"Units {unit1=} and {unit2=} do not represent the same physical quantity.")
    return conv_factor


#---------------------
def main():

    #Examples of conversion factors and Variable usage.
    print(conversion_factor("L/min", "m3/s"))
    print(conversion_factor("W", "kJ/hr"))
    print(conversion_factor("W", "kJ/hr"))

    time_sim = Variable(365, "d")
    print(f"time_sim in days: {time_sim.get_value('d')}")
    print(f"time_sim in hours: {time_sim.get_value('hr')}")
    print(f"time_sim in seconds: {time_sim.get_value('s')}")
    return

if __name__=="__main__":
    main()
    pass
