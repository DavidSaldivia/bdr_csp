# BDR_CSP - Concentrated Solar Power with Beam-Down Receivers

Python codes for concentrated solar power (CSP) systems with Beam-Down Receivers (BDR) using Monte Carlo Ray Tracing (MCRT) analysis and thermal simulation.

## Core Modules (`bdr_csp/`)

The `bdr_csp` package contains five core modules, developed as part of my PhD thesis: [Design and Optimisation of Beam-Down Particle Receivers for High-Temperature Concentrating Solar Thermal Applications](https://doi.org/10.26190/unsworks/30144).

### 1. **bdr.py** - Beam Down Receiver Optical Analysis
Primary module for optical simulation of beam-down receiver systems. Handles:
- Solar field layout and heliostat geometry
- Hyperboloid mirror optical properties
- Tertiary optical devices
- Ray tracing and flux calculations
- Integration with `antupy` framework for building CSP plant models

**Publications:**
- [Optical analysis and optimization of a beam-down receiver for advanced cycle concentrating solar thermal plants](https://doi.org/10.1016/j.applthermaleng.2021.117405) (Applied Thermal Engineering, 2021)
- [Effect of heliostat curvature on optical performance of beam-down systems](https://doi.org/10.1063/5.0149309) (AIP Conference Proceedings, 2023)

### 2. **spr.py** - Solid Particle Receiver Components
Implements solid particle receiver models that work with BDR optics:
- **HPR0D** - Horizontal Particle Receiver (0D simplified model)
- **HPR2D** - Horizontal Particle Receiver (2D detailed model)
- **TPR2D** - Tilted Particle Receiver (2D model using granular flow)
- Thermal absorption and simplified particle dynamics
- Convection and radiation heat loss calculations

**Publications:**
- [Thermal simulation of beam-down particle receivers with different configurations](https://doi.org/10.52825/solarpaces.v1i.664) (SolarPACES, 2022)

### 3. **pb.py** - Power Block and Thermodynamic Cycle
Comprehensive thermal and power cycle simulation:
- `ModularCSPPlant` - Integrated plant model combining optics and thermal systems
- Power block thermal analysis
- Energy balance calculations
- Integration of BDR optics with particle receivers

**Publications:**
- [Thermal simulation of beam-down particle receivers with different configurations](https://doi.org/10.52825/solarpaces.v1i.664) (SolarPACES, 2022)
- [Techno-economic assessment of the storage value of a novel modular Beam-Down
Receiver CSP plant in Australia](https://www.ceem.unsw.edu.au/sites/default/files/documents/62.pdf) (APSRC, 2022)

### 4. **htc.py** - Heat Transfer Correlations
Heat transfer coefficient calculations:
- Natural convection on surfaces
- Radiative heat transfer
- Custom correlations for receiver geometries

### 5. **dir.py** - Directory and Path Management
Utility module for managing file paths and data directories across the project.

## Research Projects (`projects/`)

### 1. **optics/** - Optical Analysis and Simulations
Parametric optical analysis of beam-down receiver configurations using MCRT:
- Solar field design for beam-down optics.
- Different types of tertiary optical devices.
- Performance simulations at single and multiple points.
- Flux distribution analysis over receiver surface.

### 2. **spr/** - Solid Particle Receiver Studies
Detailed studies of particle receiver designs and performance:
- HPR (Horizontal Particle Receiver) with conveyor belt analysis
- TPR (Tilted Particle Receiver) with granular flow analysis
- 2D thermal and simplified fluid dynamics modeling
- HPR-TPR performance comparison
- Results analysis and visualization

### 3. **optim/** - Thermal System Optimization
Thermal subsystem optimization focused on LCOH (Levelized Cost of Heat) minimization:
- Annual performance calculations
- Sensitivity analysis for thermal and economic performance
- Bayesian optimization of thermal parameters

### 4. **dispatch/** - Market Integration and Dispatch
Integrated CSP system simulation and operational dispatch:
- Real-world performance simulations in Australian NEM (National Electricity Market)
- Dispatch model integration with real weather data
- Alice Springs case study
- Performance characterization under real operating conditions
- Parametric dispatch analysis

## Usage Notes

- **Unified Framework**: All modules build on the `antupy` framework for consistent plant simulation
- **Vectorized Operations**: Core modules use vectorized NumPy/Pandas operations for efficiency
- **Data Dependencies**: Projects typically require MCRT ray datasets generated via SolarPILOT
- **Module Relationships**: 
  - `bdr` is foundational (used by `spr` and `pb`)
  - `spr` depends on `bdr`
  - `pb` integrates `bdr` and `spr`
  - Projects use various combinations of these modules

## Getting Started

1. Install dependencies from `pyproject.toml`
2. For optical analysis: Use projects in `optics/` with SolarPILOT ray datasets
3. For receiver simulation: See examples in `spr/` project
4. For complete system analysis: See the `optim/` or `dispatch/` modules.

## Legacy Code

Historical versions and earlier implementations are available in `old/` folder for reference.
