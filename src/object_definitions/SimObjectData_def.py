import numpy as np
from numpy.typing import NDArray
from skyfield import timelib


class SimObjectData:
    def __init__(self, 
                 satellite_name: str, 
                 time: NDArray[np.float64],
                 pos: NDArray[np.float64], 
                 vel: NDArray[np.float64]) -> None:
        """
        =========================================================================================================
        ATTRIBUTES:
            satellite_name (str):           Name of the satellite
            time (NDArray[np.float64]):     1-by-n Array containing the sample times [seconds] after simulation 
                                              start time (simulation offset)
            pos (NDArray[np.float64]):      3-by-n Position Array of the satellite
            vel (NDArray[np.float64]):      3-by-n Velocity Array of the satellite
        =========================================================================================================
        """
        
        self.satellite_name: str = satellite_name
        self.time: NDArray[np.float64] = time
        self.pos: NDArray[np.float64] = pos
        self.vel: NDArray[np.float64] = vel