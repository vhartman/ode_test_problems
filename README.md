# Overview

Python implementation of test problems for numerical ODE-solvers

Problems are mainly taken from 
- https://archimede.dm.uniba.it/~testset/CWI_reports/testset2006r23.pdf
- https://tutorials.juliadiffeq.org/html/models/01-classical_physics.html
- https://www.unige.ch/~hairer/testset/testset.html

### Implemented Problems

In no specific order:
- Chemical Akzo Nobel
- Hires
  - 8 non-linear ODEs
- Pollutions
  - Chemical reaction of an air pollution problem
- Ring Modulator
  - Electronics; description of a specific circuit
- Pleiades
  - Non-stiff, medium dimensionality. Celestial mechanics
- Van der Pol Oscillator
  - From electronics, bahviour of vacuum tubes in an electric circuit
- Harmonic Oscillator
- Arenstorf Orbit
  - 3 Body problem, very dependent on adaptive stepsize, periodic
- Robertson
  - 3 Non linear ODEs, chemical reaction
- Simple Pendulum
- Henon Heiles
  - Non-linear motion of a star around a galactic center restricted to a plane

Most of the problems are stiff.

# Todo
- Compute baseline solution for the problems that do not have one yet
- Add more problems
  - Simpler methods for basic correctness testing
  - Methods with more expensive function calls
  - Ricatti-style equations from control theory
