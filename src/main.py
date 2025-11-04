import logging

from __init__ import initialize
from plotting.plot import plot
from object_definitions.BasiliskSimulator_def import BasiliskSimulator
from object_definitions.SkyfieldSimulator_def import SkyfieldSimulator


def simualte_satellite_orbits():

    # Load config and define all neccessary objects
    cfg = initialize('configs/default.yaml')
    
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