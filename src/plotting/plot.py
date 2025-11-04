from typing import Optional
from pathlib import Path

from object_definitions.Config_def import Config
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

    
    


