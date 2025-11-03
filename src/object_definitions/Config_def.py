import yaml
import logging
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.Satellite_def import Satellite
from object_definitions.TwoLineElement_def import TLE

"""
=========================================================================================================
Overview of classes and their methods with high-level functionality

Config
    __init__:       Initializes the Config instance with global config parameters that 
                      apply to all simulations
    read:           Reads the config file using yaml.full_load

BasiliskConfig
    __init__:       Initializes the BasiliskConfig instance with config parameters that 
                      only apply to th basilisk simualtion framework

SkyfiledConfig:
    __init__:
=========================================================================================================
"""


@dataclass_json
@dataclass
class BasiliskSettings:
    useSphericalHarmonics: bool
    orbitCase: str
    show_plots: bool
    show_progress_bar: bool
    deltaT: float
    use_custom_initial_state: bool

@dataclass_json
@dataclass
class SkyfieldSettigns:
    deltaT: float


class Config:
    def __init__(self, config_file_path: str) -> None:
        """
        =========================================================================================================
        [WORK IN PROGRESS]
        Initialize Config instance with attributes from the config file

        INPUTS:
           config_file_path                    
        
        ATTRIBUTES:
            startTime
            simulationDuration
            tle_export_path
            timestamp_str       Used in the naming of data files. str holding the real-world simulation start time.
            satellites          One Satellite instance for each satellite described in the default config.
            b_set               BasiliskSettings instance describing the Basilisk simulation settings
            s_set               SkyfieldSettings instance describing the Skyfield simulation settings     
        =========================================================================================================
        """
        ####################
        # Load cofig files #
        ####################
        d_cfg = self.read(config_file_path)                 # default config
        b_cfg = self.read(d_cfg['BASILISK']['config_path']) # basilisk config
        s_cfg = self.read(d_cfg['SKYFIELD']['config_path']) # skyfield condig
        
        ##################################################
        # Fetch parameters from the various config files #
        ##################################################
        # Fetch from default.yaml
        startTime_str = d_cfg['SIMULATION']['startTime'] # str
        simulationDuration = d_cfg['SIMULATION']['simulationDuration'] # float  
        tle_export_path = d_cfg['SIMULATION']['tle_export_path']  

        # Fetch from basilisk.yaml
        useSphericalHarmonics = b_cfg['BASILISK_SIMULATION']['useSphericalHarmonics']
        orbitCase = b_cfg['BASILISK_SIMULATION']['orbitCase']
        show_plots = b_cfg['BASILISK_SIMULATION']['show_plots']
        show_progress_bar = b_cfg['BASILISK_SIMULATION']['show_progress_bar']
        bsk_deltaT = b_cfg['BASILISK_SIMULATION']['deltaT']
        use_custom_initial_state = b_cfg['BASILISK_SIMULATION']['use_custom_initial_state']

        # Fetch from skyfield.yaml
        skf_deltaT = s_cfg['SKYFIELD_SIMULATION']['deltaT']

        # Create Satellite intstances
        satellites = self.generate_satellite_instances_from_config(d_cfg)
        
        ##############################
        # Assign instance attributes #
        ##############################
        self.startTime = startTime_str
        self.simulationDuration = simulationDuration
        self.tle_export_path = tle_export_path
        self.timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.satellites = satellites

        # Assign BasiliskSettings instance to b_set attribute
        self.b_set = BasiliskSettings(
            useSphericalHarmonics,
            orbitCase,
            show_plots,
            show_progress_bar,
            bsk_deltaT,
            use_custom_initial_state
        )

        # Assign SkyfieldSettings instance to s_set attribute
        self.s_set = SkyfieldSettigns(
            skf_deltaT
        )


    def read(self, config_file_path: str):
        # Get full path to the target config file
        config_path = Path(config_file_path)
        
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.full_load(f)

        return config
    
        
    
    @staticmethod
    def generate_satellite_instances_from_config(loaded_default_cfg) -> list[Satellite]:
        """
        TODO: Write docstring...

        TODO: Add functionality to handle the case where initial pos/vel is not provided in default.yaml:
                -> Okay if 'use_custom_initial_state' == false (from basilisk.yaml)
                    -> Assign None for satellite atributes related to the initial custom pos/vel
                -> Error otherwise
        """

        def _parse_inital_states_from_config(initial_pos: str, initial_vel: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            # [DEPRECIATED AS OF 23.10]
            """
            INPUT:
                initial_pos             3 element initial satellite position list. Elements can be str, int or float
                initial_vel             3 element initial satellite velocity list. Elements can be str, int or float

            OUTPUT:
                parsed_ndarray_pos      3 element np.array position vector 
                parsed_ndarray_vel      3 element np.array velocity vector 
            """
            

            if len(initial_pos) != 3 or len(initial_vel) != 3:
                raise ValueError("Wrong number of elements in 'initial_pos' or 'initial_vel' from default.yaml")
            
            parsed_pos = [_parse_element(x) for x in initial_pos]
            parsed_vel = [_parse_element(v) for v in initial_vel]

            parsed_ndarray_pos = np.array(parsed_pos, dtype=float)
            parsed_ndarray_vel = np.array(parsed_vel, dtype=float)
            
            #parsed_initial_state = np.array(parsed_pos + parsed_vel, dtype=float)

            return parsed_ndarray_pos, parsed_ndarray_vel
        

        def _parse_element(val) -> float:
            """
            Parse input value from type str or int to type float that can later be used in np.array([], dtype=float)
            """
            if isinstance(val, str):
                try:
                    return float(val.replace("E", "e"))
                except:
                    raise ValueError(f"Invalid numeric string: {val}")
            elif isinstance(val, (float, int)):
                return float(val)
            else:
                raise TypeError(f"Unsupported type {type(val)} for element {val}")



        # Use raw config for satellite information
        all_sat_info = loaded_default_cfg['SATELLITES']
        
        satellite: Satellite
        satellites: list[Satellite] = []
        for sat, sat_info in all_sat_info.items():
            # Check if all satellite attributes are compatible with the Satellite object
            allowed = {'name', 'tle_line1', 'tle_line2', 'custom_initial_pos', 'custom_initial_vel'} # NOTE: This must be updated of the config-format changes!
            unknown = set(sat_info) - allowed
            if unknown:
                raise ValueError(f"{sat}: unknown keys for {unknown}")
            
            # Extract tle strings
            tle_line1 = sat_info['tle_line1']
            tle_line2 = sat_info['tle_line2']
            
            # Generate TLE instance from tle lines
            tle = TLE(
                tle_line1,
                tle_line2
            )
            
            # Create Satellite instance form current config satellite
            satellite = Satellite(
                sat_info['name'],
                tle_line1,
                tle_line2,
                tle,
                init_pos = None, # This field will be populated by data from skyfield later
                init_vel = None  # This field will be populated by data from skyfield later
            )
            
            logging.debug(f"Appending {sat_info['name']} to 'satellites'")
            satellites.append(satellite)

        return satellites
        


        
