import logging

from __init__ import initialize
from plotting.plot import plot
from object_definitions.BasiliskSimulator_def import BasiliskSimulator
from object_definitions.SkyfieldSimulator_def import SkyfieldSimulator


def simualte_satellite_orbits():

    # Load config and define all neccessary objects
    cfg = initialize('configs/default.yaml')

    # Bypass the simulation if we instead want to plot old datafiles
    if not cfg.bypass_sim_to_plot:
    
        # Initialize Skyfield SGP4 Propagator
        skf = SkyfieldSimulator(cfg)

        # Run Skyfield SGP4 propagator
        skf.run()

        # Extract initial states @ simulation startTime and update cfg.satellites
        skf.extract_initial_states_and_update_satellites(cfg)

        # Initialize Basilisk Dynamic Model Propagator
        bsk = BasiliskSimulator(cfg)

        # Run Basilisk Dynamic Model Propagator
        bsk.run()

    # Plot results
    plot(cfg)


if __name__ == "__main__":
    simualte_satellite_orbits()


# TODO
"""
Disturbance:
* Implement drag with exponentially decaying atmosphere

* Also include J3 and J4 terms in Basilisk gravitational disturbance

* Implement 3rd body pull from the moon (and check sun)

* Implement option to toggle disturbance effect from solar radiation pressure

* Implement functionality to log force from SRP


Plotting:
* Generate nadir projection map

* Change y-axis scaling to km or logarithmic

* Create plot to show altitude

* Change plot colors where the two simulation outputs are shown in the same plot


Simulator Misk:
* Generate timestamped .bin files for Vizard without overwriting old data (like it is already implemented in sim_data)

* Make option available to toggle displaying plots
"""