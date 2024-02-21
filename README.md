[![CI](https://github.com/fhswf/MLPro-MPPS/actions/workflows/ci.yml/badge.svg)](https://github.com/fhswf/MLPro-MPPS/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/mlpro-mpps.svg)](https://badge.fury.io/py/mlpro-mpps)
[![PyPI Total Downloads](https://static.pepy.tech/personalized-badge/mlpro-mpps?period=total&units=international_system&left_color=blue&right_color=orange&left_text=PyPI%20Total%20Downloads)](https://pepy.tech/project/mlpro-mpps)
[![PyPI Last Month Downloads](https://static.pepy.tech/personalized-badge/mlpro-mpps?period=month&units=international_system&left_color=blue&right_color=orange&left_text=PyPI%20Last%20Month%20Downloads)](https://pepy.tech/project/mlpro-mpps)

# MLPro-MPPS - A Customizable Framework for Multi-Purpose Production Systems in Python

<img src="https://github.com/fhswf/MLPro/blob/main/doc/logo/original/logo.png?raw=True" align="right" width="25%"/>
MLPro-MPPS provides functionalities to design and develop customizable multi-purpose production systems in Python. This framework is integrated with MLPro (https://mlpro.readthedocs.io) and inherits several basic functionalities from MLPro. The user can design a production system simulation from the lowest component level (sensors and actuators) until the combination of them in the form of MPPS. The other possibility is to use the ready-to-use components in the pool of objects. Moreover, since MLPro-MPPS is compatible with MLPro, it is possible to utilize MLPro-MPPS as an environment in MLPro-RL, as a game board in MLPro-GT, or as a state transition function in an MLPro-BF-Systems. Hence, MLPro-MPPS is reusable and powerful.

## Getting Started
To get started with MLPro-MPPS, you can begin with the following tasks:

### Installation from PyPI
```python
pip install mlpro-mpps
```

### Dependencies
Please read [requirements.txt](https://github.com/fhswf/MLPro-MPPS/blob/main/requirements.txt)

### Introduction Video
You can see a video of the introduction of MLPro-MPPS at the 2nd IEEE Industrial Electronics Society Annual Online Conference by clicking [this link](https://youtu.be/zjvDDBpl-bE).

### First Steps
After installing MLPro-MPPS and its dependencies, we suggest starting with the ready-to-run examples:
1. [HOWTO 001 - SETTING UP COMPONENTS AND MODULES](https://github.com/fhswf/MLPro-MPPS/blob/main/src/mlpro_mpps/examples/howto_001_set_up_components_and_modules_in_MPPS.py)
2. [HOWTO 002 - SETTING UP MPPS](https://github.com/fhswf/MLPro-MPPS/blob/main/src/mlpro_mpps/examples/howto_002_set_up_MPPS.py)
3. [HOWTO 003 - MPPS IN REINFORCEMENT LEARNING](https://github.com/fhswf/MLPro-MPPS/blob/main/src/mlpro_mpps/examples/howto_003_run_RL_on_BGLP_using_MPPS.py)
4. [HOWTO 004 - MPPS IN GAME THEORY](https://github.com/fhswf/MLPro-MPPS/blob/main/src/mlpro_mpps/examples/howto_004_run_GT_on_BGLP_using_MPPS.py)

Additionally, the class diagram of the basic function is available in [this directory](https://github.com/fhswf/MLPro-MPPS/tree/main/doc/class_diagram). The ready-to-use components and MPPS samples can be found in [this directory](https://github.com/fhswf/MLPro-MPPS/tree/main/src/mlpro_mpps/pool).


## Key Features and Functionalities

#### a) Providing base classes of components in a clean structure
- Including base classes for a sensor, an actuator, and a component state (required for simulation)

#### b) Versatile and configurable
- MPPS is versatile, which means that it has a high degree of flexibility, where the users can set up a production system with as many sensors, actuators, components, and modules as possible
- Moreover, MPPS is also not restricted to a modular production system, but also applicable in any form of technical systems

#### c) Simplification of measurements and designs of the MPPS-based systems' dynamics
- When MPPS is simulated, the dynamics of the sensors and component states are affected by the actual status of the actuators. In the simulation mode, their dynamics are measured through a mathematical calculation that is defined by the [TransferFunction class of MLPro](https://github.com/fhswf/MLPro/blob/main/src/mlpro/bf/physics/basics.py)
- The mathematical calculation can be attached to the lowest level of the components, such as sensors and component states, which make them reusable and reproducible.

#### d) Well-integrated to [MLPro](https://mlpro.readthedocs.io)
- Possibility to convert MPPS into the reinforcement learning environment in MLPro-RL or game theory game board in MLPro-GT
- Possibility to reuse functionalities from [MLPro](https://mlpro.readthedocs.io)

## Development
- Consequent object-oriented design and programming (OOD/OOP)
- Quality assurance by test-driven development
- Hosted and managed on GitHub
- Agile CI/CD approach with automated test and deployment
- Clean code paradigma

## Citing MLPro-MPPS
- https://doi.org/10.1109/ONCON60463.2023.10431280
- https://doi.org/10.1016/j.simpa.2023.100509

## Project and Team
Project MLPro-MPPS was started in 2022 by the [Group for Automation Technology and Learning Systems at the South Westphalia University of Applied Sciences, Germany](https://www.fh-swf.de/de/forschung___transfer_4/labore_3/labs/labor_fuer_automatisierungstechnik__soest_1/standardseite_57.php).

MLPro is currently designed and developed by [Steve Yuwono](https://github.com/steveyuwono), [Marlon Löppenberg](https://github.com/marlonloeppenberg), and further [contributors](https://github.com/fhswf/MLPro/graphs/contributors). 


## How to contribute
If you want to contribute, please read [CONTRIBUTING.md](https://github.com/fhswf/MLPro-MPPS/blob/main/CONTRIBUTING.md)
