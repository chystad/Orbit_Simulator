import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from pathlib import Path

from object_definitions.Config_def import Config
from object_definitions.SimData_def import SimData
from plotting.DataLoader_def import DataLoader

PLT_SAVE_FOLDER_PATH = Path('data/sim_plt')


def plot(cfg: Config, alt_datafilename_to_plot: Optional[str] = None) -> None:
    """
    Input:
        cfg:                        Simulation config
        alt_datafilename_to_plot:   Only plots the current simulation if this is set to 'None'. 
                                      str containing the timestamp (the part of the filename before '_skf' or '_bsk').
                                      of the datafiles 
    """

    # Initialize data loader object
    data_loader = DataLoader()

    if alt_datafilename_to_plot is None:
        # plot results from this simulation 
        datafiles_to_plot = data_loader.get_datafiles_by_timestamp(cfg.timestamp_str)
        
    else:
        # plot results from a previous simulation
        datafiles_to_plot = data_loader.get_datafiles_by_timestamp(alt_datafilename_to_plot)

    # Check which simulator generated the data
    skf_sim_data = None
    bsk_sim_data = None
    for datafile in datafiles_to_plot:
        sim_data = data_loader.load_sim_data_file(datafile)
        if "skf" in datafile:
            skf_sim_data = sim_data
        elif "bsk" in datafile:
            bsk_sim_data = sim_data
        else:
            raise ValueError(f"Neither 'skf' or 'bsk' is present in datafile name: {datafile}"
                             f"The datafile cannot be categorized as skyfield or basilisk simulation output.")
        
    if skf_sim_data is None or bsk_sim_data is None:
        raise ValueError(f"one or both strings: 'skf', 'bsk' with was not found in a datafile starting with '{cfg.timestamp_str}'")
    
    # TODO: insert check of config parameter to actually run this command.
    # TODO: Consider moving the above operation to a function in DataLoader_def
    plot_pos_comparison(skf_sim_data, bsk_sim_data)



def plot_pos_comparison(skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot position vectors for each satellite from two SimData datasets.

    For satellite i (0..n-1), creates a new figure and plots:
      - SKF position components (x,y,z) as different blue shades
      - BSK position components (x,y,z) as different green shades
    The time axis comes from each dataset's own `time` array.

    Args:
        skf_sim_data: SimData
        bsk_sim_data: SimData
    Returns:
        None
    """
    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Color palettes (light → dark)
    skf_colors = ["#9ecae1", "#3182bd", "#08519c"]  # blues
    bsk_colors = ["#a1d99b", "#31a354", "#006d2c"]  # greens
    comp_labels = ["x", "y", "z"]

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks (won't fail if shapes are as specified)
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf = np.ravel(skf.time)
        t_bsk = np.ravel(bsk.time)

        # Create a new figure for this satellite; no explicit numbering to avoid conflicts
        plt.figure()
        ax = plt.gca()

        # Plot SKF (blue shades)
        for comp in range(3):
            ax.plot(
                t_skf,
                skf.pos[comp, :],
                label=f"SKF {comp_labels[comp]}",
                linewidth=1.8,
                color=skf_colors[comp],
            )

        # Plot BSK (green shades)
        for comp in range(3):
            ax.plot(
                t_bsk,
                bsk.pos[comp, :],
                label=f"BSK {comp_labels[comp]}",
                linewidth=1.8,
                linestyle="--",
                color=bsk_colors[comp],
            )

        sat_name = skf.satellite_name if getattr(skf, "satellite_name", None) else f"Satellite {i+1}"
        ax.set_title(f"Position comparison — {sat_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position component")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        plt.show()


    
def save_plot():
    pass


