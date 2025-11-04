import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

# from object_definitions.BaseSimulator_def import BaseSimulator
from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite

from Basilisk import __path__
from Basilisk.simulation import spacecraft
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport)

"""
=========================================================================================================
TODO docstring
=========================================================================================================
"""

class BasiliskSimulator:
    """
    =========================================================================================================
    ATTRIBUTES:
        cfg             Config instance
        simTaskName     Name of the task
        simProcessName  Name of the process
        scSim           Simulation module container
        dynProcess      Simulation process
        scObjects       List containing all simulation objects (satellites)
        dataRecorders   List containing all simulation recorders (one for each scObject)
        planet(temp)    planet (always earth)
    =========================================================================================================
    """
    def __init__(self, cfg: Config) -> None:
        logging.debug("Setting up Basilisk simulation...")
        
        ###############
        # Load config #
        ###############
        
        self.cfg = cfg     # Assign config to self.cfg attribute
        d_set = cfg        # default config
        b_set = cfg.b_set  # basilisk config
        


        ###################################
        # Configure simulation parameters #
        ###################################

        # Set fixed simulation integration time step
        simulationTimeStep = macros.sec2nano(b_set.deltaT)

        # Set simulation duration
        simualtionDuration_sec = d_set.simulationDuration * 60 * 60
        simulationDuration = macros.sec2nano(simualtionDuration_sec)

        # Set number of data points
        numDataPoints = simulationDuration // simulationTimeStep

        # Set sample time (same as 'deltaT' in basilisk simulation config)
        samplingTime = unitTestSupport.samplingTime(simulationDuration, simulationTimeStep, numDataPoints)

        # path to basilisk. Used to fetch predesigned models
        bskPath = __path__[0]
        fileName = os.path.basename(os.path.splitext(__file__)[0])

        # # set the simulation time
        # n = np.sqrt(mu / oe.a / oe.a / oe.a)
        # P = 2. * np.pi / n
        # if b_set.useSphericalHarmonics:
        #     simulationTime = macros.sec2nano(3. * P)
        # else:
        #     simulationTime = macros.sec2nano(.8 * P)

        # # Setup data logging before the simulation is initialized
        # if b_set.useSphericalHarmonics:
        #     numDataPoints = 400
        # else:
        #     numDataPoints = 100
        # # The msg recorder can be told to sample at an with a minimum hold period in nano-seconds.
        # # If no argument is provided, i.e. msg.recorder(), then a default 0ns minimum time period is used
        # # which causes the msg to be recorded on every task update rate.
        # # The recorder can be put onto a separate task with its own update rate.  However, this can be
        # # trickier to do as the recording timing must be carefully balanced with the module msg writing
        # # to avoid recording an older message.
        # samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)


        
        #############################
        # Set up simulation objects #
        #############################

        # Select task and process names
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"

        # Create a sim module as an empty container
        self.scSim = SimulationBaseClass.SimBaseClass()

        # Configure the use of simulation progress bar (shown in terminal)
        self.scSim.SetProgressBar(b_set.show_progress_bar)

        # Create the simulation process
        dynProcess = self.scSim.CreateNewProcess(self.simProcessName)

        # create the dynamics task and specify the integration update time
        dynProcess.addTask(self.scSim.CreateNewTask(self.simTaskName, simulationTimeStep))

        # setup Gravity Body and initialize Earth as the central celestial body
        gravFactory = simIncludeGravBody.gravBodyFactory()
        self.planet = gravFactory.createEarth()
        self.planet.isCentralBody = True          # ensure this is the central gravitational body
        if b_set.useSphericalHarmonics:
            # If extra customization is required, see the createEarth() macro to change additional values.
            # For example, the spherical harmonics are turned off by default.  To engage them, the following code
            # is used
            self.planet.useSphericalHarmonicsGravityModel(bskPath + '/supportData/LocalGravData/GGM03S-J2-only.txt', 2)

            # The value 2 indicates that the first two harmonics, excluding the 0th order harmonic,
            # are included.  This harmonics data file only includes a zeroth order and J2 term.
        mu = self.planet.mu

        

        ##########################################
        # Initialize scObjects and dataRecorders #
        ##########################################

        # Initialize empty containers for to-be-defined Spacecraft/Recorder objects
        self.scObjects: list[spacecraft.Spacecraft] = []
        self.dataRecorders: list = [] # list of what?

        # get satellites from config
        satellites = self.cfg.satellites

        # setup the orbit using classical orbit elements
        oe = orbitalMotion.ClassicElements()
        rLEO = 7000. * 1000      # meters
        rGEO = 42000. * 1000     # meters
        if b_set.orbitCase == 'GEO':
            oe.a = rGEO
            oe.e = 0.00001
            oe.i = 0.0 * macros.D2R
        elif b_set.orbitCase == 'GTO':
            oe.a = (rLEO + rGEO) / 2.0
            oe.e = 1.0 - rLEO / oe.a
            oe.i = 0.0 * macros.D2R
        else:                   # LEO case, default case 0
            oe.a = rLEO
            oe.e = 0.001
            oe.i = 33.3 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        oe = orbitalMotion.rv2elem(mu, rN, vN)      # this stores consistent initial orbit elements
        # with circular or equatorial orbit, some angles are arbitrary

        # Define angle between satellites following the same orbit
        separationAng = 90.0 * macros.D2R

        # Populate object containers
        for i, sat in enumerate(satellites):
            # Initialize spacecraft object
            scObj = spacecraft.Spacecraft()
            scObj.ModelTag = sat.name

            # Add spacecraft object to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scObj)

            # The gravitational body must be added to the spacecraft object
            gravFactory.addBodiesTo(scObj)

            if not b_set.use_custom_initial_state: # default case
                # Modify the orbital elements to get the wanted satellite separation
                if i > 0:
                    oe.f = oe.f - separationAng
                    rN, vN = orbitalMotion.elem2rv(mu, oe)

            else: # use initial state
                rN = sat.init_pos # [m]   In N frame (inertial = ECI)
                vN = sat.init_vel # [m/s] in N frame (inertial = ECI)

            ########################## MANUAL OVERRIDE ##########################
            # r_BN_N = np.array([10000e3, 0.0, 0.0])     # Position vector [m]
            # v_BN_N = np.array([0.0, 1e3, 0.0])      # Velocity vector [m/s]

            # scObj.hub.r_CN_NInit = r_BN_N  # m   - r_BN_N
            # scObj.hub.v_CN_NInit = v_BN_N  # m/s - v_BN_N

            #######################################################################

            scObj.hub.r_CN_NInit = rN  # m   - r_BN_N
            scObj.hub.v_CN_NInit = vN  # m/s - v_BN_N
            

            # Create object state recorders
            dataRec = scObj.scStateOutMsg.recorder(samplingTime)

            # Add recorder to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, dataRec)
                        
            # Append defined spacecraft object and dataRec to scObjects and dataRecorders, respectively
            self.scObjects.append(scObj)
            self.dataRecorders.append(dataRec)


        
        ##################
        # Data recording #
        ##################
        

        #
        #   setup orbit and simulation time
        #
        # # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
        # scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
        # scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N

        # # Calculate the initial state for the second spacecraft with a lag angle wrt the first spacecraft
        # lagAng = 20.0 * macros.D2R
        # oe2 = copy(oe)
        # oe2.f = oe.f - lagAng
        # rN2, vN2 = orbitalMotion.elem2rv(mu, oe2)

        # # Set initial state for the second spacecraft
        # scObject2.hub.r_CN_NInit = rN2  # m   - r_BN_N
        # scObject2.hub.v_CN_NInit = vN2  # m/s - v_BN_N


        # These vectors specify the inertial position and velocity vectors relative to the planet of the
        # spacecraft center of mass location.  Note that there are 2 points that can be tracked.  The user always
        # specifies the spacecraft center of mass location with the above code.  If the simulation output should be
        # about another body fixed point B, this can be done as well.  This is useful in particular with more challenging
        # dynamics where the center of mass moves relative to the body.  The following vector would specify the location of
        # the spacecraft hub center of mass (Bc) relative to this body fixed point, as in
        #
        #    scObject.hub.r_BcB_B = [[0.0], [0.0], [1.0]]
        #

        # If this vector is not specified, as in this tutorial scenario, then it defaults to zero.  If only a rigid hub
        # is modeled, the Bc (hub center of mass) is the same as C (spacecraft center of mass).  If the spacecraft contains
        # state effectors such as hinged panels, fuel slosh, imbalanced reaction wheels, etc., then the points
        # Bc and C would not be the same.  Thus, in this simple simulation the body fixed point B and
        # spacecraft center of mass are identical.

        
        # create a logging task object of the spacecraft output message at the desired down sampling ratio
        # dataRec = self.scObjects[0].scStateOutMsg.recorder(samplingTime)
        # self.scSim.AddModelToTask(self.simTaskName, dataRec)

        # # Also record the second spacecraft state
        # dataRec2 = scObject2.scStateOutMsg.recorder(samplingTime)
        # self.scSim.AddModelToTask(self.simTaskName, dataRec2)

        # self.dataRecorders = [dataRec, dataRec2]

        # Vizard Visualization Option
        # ---------------------------
        # If you wish to transmit the simulation data to the United based Vizard Visualization application,
        # then uncomment the following
        # line from the python scenario script.  This will cause the BSK simulation data to
        # be stored in a binary file inside the _VizFiles sub-folder with the scenario folder.  This file can be read in by
        # Vizard and played back after running the BSK simulation.
        # To enable this, uncomment this line:

        viz = vizSupport.enableUnityVisualization(self.scSim, self.simTaskName, self.scObjects,
                                                saveFile=__file__
                                                # liveStream=True
                                                )

        # The vizInterface module must be built into BSK.  This is done if the correct CMake options are selected.
        # The default CMake will include this vizInterface module in the BSK build.  See the BSK HTML documentation on
        # more information of CMake options.

        # By using the gravFactory support class to create and add planetary bodies the vizInterface module will
        # automatically be able to find the correct celestial body ephemeris names.  If these names are changed, then the
        # vizSupport.py support library has to be customized.
        # Currently Vizard supports playback of stored simulation data files as well as live streaming.
        # Further, some display elements such as thruster or reaction wheel panels are only visible if
        # such devices are being simulated in BSK.

        # While Vizard has many visualization features that can be customized from within the application, many Vizard
        # settings can also be scripted from the Basilisk python script.  A complete discussion on these options and
        # features can be found the the Vizard documentation pages.

        # Before the simulation is ready to run, it must be initialized.  The following code uses a
        # convenient macro routine
        # which initializes each BSK module (run self init, cross init and reset) and clears the BSK logging stack.

        #   initialize Simulation:  This function runs the self_init()
        #   and reset() routines on each module.
        self.scSim.InitializeSimulation()

        #   configure a simulation stop time and execute the simulation run
        self.scSim.ConfigureStopTime(simulationDuration)
        
        

    def run(self):
        # Execute the simulation
        logging.debug("Basilisk simulation running...")
        
        

        """
        At the end of the python script you can specify the following example parameters.

        Args:
            show_plots (bool): Determines if the script should display plots
            orbitCase (str):

                ======  ============================
                String  Definition
                ======  ============================
                'LEO'   Low Earth Orbit
                'GEO'   Geosynchronous Orbit
                'GTO'   Geostationary Transfer Orbit
                ======  ============================

            useSphericalHarmonics (Bool): False to use first order gravity approximation: :math:`\\frac{GMm}{r^2}`

            planetCase (str): {'Earth', 'Mars'}
        """

       
        self.scSim.ExecuteSimulation()
        # Note that this module simulates both the translational and rotational motion of the spacecraft.
        # In this scenario only the translational (i.e. orbital) motion is tracked.  This means the rotational motion
        # remains at a default inertial frame orientation in this scenario.  There is no appreciable speed hit to
        # simulate both the orbital and rotational motion for a single rigid body.  In the later scenarios
        # the rotational motion is engaged by specifying rotational initial conditions, as well as rotation
        # related effectors.  In this simple scenario only translational motion is setup and tracked.
        # Further, the default spacecraft parameters, such as the unit mass and the principle inertia values are
        # just fine for this orbit simulation as they don't impact the orbital dynamics in this case.
        # This is true for all gravity force only orbital simulations. Later
        # tutorials, such as scenarioAttitudeFeedback.py,
        # illustrate how to over-ride default values with desired simulation values.

        #   retrieve the logged data
        # the data is stored inside dataLog variable.  The time axis is stored separately from the data vector and
        # can be access through dataLog.times().  The message data is access directly through the message
        # variable names.
        
        # posData = self.dataRecorders[0].r_BN_N
        # velData = self.dataRecorders[0].v_BN_N

        # posData2 = self.dataRecorders[1].r_BN_N
        # velData2 = self.dataRecorders[1].v_BN_N

        """
        
        np.set_printoptions(precision=16)

        # When the simulation completes 2 plots are shown for each case.  One plot always shows
        # the inertial position vector components, while the second plot either shows a planar
        # orbit view relative to the peri-focal frame (no spherical harmonics), or the
        # semi-major axis time history plot (with spherical harmonics turned on).
        figureList, finalDiff = self.plotOrbits(self.dataRecorders[0].times(), posData, velData, posData2, velData2, oe, mu, P,
                                b_set.orbitCase, b_set.useSphericalHarmonics, self.planet)

        if b_set.show_plots:
            plt.show()

        # close the plots being saved off to avoid over-writing old and new figures
        plt.close("all")

        #return finalDiff, figureList
        return
        """
        logging.debug("Basilisk simulation complete")


    def output_data(self) -> None:
        """
        Write the simulation data to a file stored in data/sim_out/
        """
        pass
        
        #### Uncomment once attribute sim_data has been defined ####

        # # Check that simulation data has been stored
        # if self.sim_data is None:
        #     raise ValueError("Simulation data not yet generated. Call skf.run() before skf.output_data().")
        
        # # Log data to file
        # self.sim_data.write_data_to_file(self.cfg.timestamp_str, "bsk")
    

    def plot(self):
        """
        TODO: 
            * Create helper function for getting good plot colors. 
                All position components of the same object should have the different shades of the same color.
            * Modify the function to respond to show_plots
            * Add possibility of saving the plot(s)
            * Add more plots(?)
        """
        # Make configs easily accessible
        d_set = self.cfg        # default config
        b_set = self.cfg.b_set  # basilisk config
        
        # Create time array with all sample times [0, (simulationDuration_sec - timeStep_sec)]
        simulationDuration_sec = d_set.simulationDuration * 60 * 60
        timeStep_sec = b_set.deltaT

        numSamples = int(simulationDuration_sec // timeStep_sec + 1)
        t = np.linspace(0, simulationDuration_sec, numSamples)

        # Get simulation data
        allPosData = []
        allVelData = []
        numTrajectories = len(self.dataRecorders)
        for i, recorder in enumerate(self.dataRecorders):
            posData = recorder.r_BN_N
            velData = recorder.v_BN_N

            allPosData.append(posData)
            allVelData.append(velData)

        # Legend
        lgnd = ['x pos [m]', 'y pos [m]', 'z pos [m]']

        # Plot
        plt.close("all")
        plt.figure(1, figsize=(10,6))
        fig = plt.gcf()

        for i in range(numTrajectories):
            for j in range(3):
                plt.plot(t, allPosData[i][:, j], label=lgnd[j])

        plt.legend()
        plt.grid(True)
        plt.show()
        


    def plotOrbits(self, timeAxis, posData, velData, posData2, velData2, oe, mu, P, orbitCase, useSphericalHarmonics, planet):
        # draw the inertial position vector components
        plt.close("all")  # clears out plots from earlier test runs
        plt.figure(1)
        fig = plt.gcf()
        ax = fig.gca()
        ax.ticklabel_format(useOffset=False, style='plain')
        finalDiff = 0.0

        for idx in range(3):
            plt.plot(timeAxis * macros.NANO2SEC / P, posData[:, idx] / 1000.,
                    color=unitTestSupport.getLineColor(idx, 3),
                    label='$r_{BN,' + str(idx) + '}$')
        
        if posData2 is not None:
            for idx in range(3):
                plt.plot(timeAxis * macros.NANO2SEC / P, posData2[:, idx] / 1000.,
                        linestyle='--', alpha=0.7,
                        color=unitTestSupport.getLineColor(idx, 3),
                        label='$r^{(2)}_{BN,' + str(idx) + '}$')
        plt.legend(loc='lower right')
            

        plt.legend(loc='lower right')
        plt.xlabel('Time [orbits]')
        plt.ylabel('Inertial Position [km]')
        figureList = {}
        pltName = fileName + "1" + orbitCase + str(int(useSphericalHarmonics)) + 'Earth'
        figureList[pltName] = plt.figure(1)

        if useSphericalHarmonics is False:
            # draw orbit in perifocal frame
            b = oe.a * np.sqrt(1 - oe.e * oe.e)
            p = oe.a * (1 - oe.e * oe.e)
            plt.figure(2, figsize=tuple(np.array((1.0, b / oe.a)) * 4.75), dpi=100)
            plt.axis(np.array([-oe.rApoap, oe.rPeriap, -b, b]) / 1000 * 1.25)
            # draw the planet
            fig = plt.gcf()
            ax = fig.gca()

            planetColor = '#008800'
            planetRadius = planet.radEquator / 1000
            ax.add_artist(plt.Circle((0, 0), planetRadius, color=planetColor))
            # draw the actual orbit
            rData = []
            fData = []
            for idx in range(0, len(posData)):
                oeData = orbitalMotion.rv2elem(mu, posData[idx], velData[idx])
                rData.append(oeData.rmag)
                fData.append(oeData.f + oeData.omega - oe.omega)
            plt.plot(rData * np.cos(fData) / 1000, rData * np.sin(fData) / 1000, color='#aa0000', linewidth=3.0
                    )
            # draw the second spacecraft orbit if available
            if posData2 is not None:
                rData2 = []
                fData2 = []
                for idx in range(0, len(posData2)):
                    oeData2 = orbitalMotion.rv2elem(mu, posData2[idx], velData2[idx])
                    rData2.append(oeData2.rmag)
                    fData2.append(oeData2.f + oeData2.omega - oe.omega)
                plt.plot(np.array(rData2) * np.cos(fData2) / 1000, np.array(rData2) * np.sin(fData2) / 1000,
                        linestyle='--', linewidth=2.0, color='#5555ff')
            # draw the full osculating orbit from the initial conditions
            fData = np.linspace(0, 2 * np.pi, 100)
            rData = []
            for idx in range(0, len(fData)):
                rData.append(p / (1 + oe.e * np.cos(fData[idx])))
            plt.plot(rData * np.cos(fData) / 1000, rData * np.sin(fData) / 1000, '--', color='#555555'
                    )
            plt.xlabel('$i_e$ Cord. [km]')
            plt.ylabel('$i_p$ Cord. [km]')
            plt.grid()

            plt.figure(3)
            fig = plt.gcf()
            ax = fig.gca()
            ax.ticklabel_format(useOffset=False, style='plain')
            Deltar = np.empty((0, 3))
            E0 = orbitalMotion.f2E(oe.f, oe.e)
            M0 = orbitalMotion.E2M(E0, oe.e)
            n = np.sqrt(mu/(oe.a*oe.a*oe.a))
            oe2 = copy(oe)
            for idx in range(0, len(posData)):
                M = M0 + n * timeAxis[idx] * macros.NANO2SEC
                Et = orbitalMotion.M2E(M, oe.e)
                oe2.f = orbitalMotion.E2f(Et, oe.e)
                rv, vv = orbitalMotion.elem2rv(mu, oe2)
                Deltar = np.append(Deltar, [posData[idx] - rv], axis=0)
            for idx in range(3):
                plt.plot(timeAxis * macros.NANO2SEC / P, Deltar[:, idx] ,
                        color=unitTestSupport.getLineColor(idx, 3),
                        label=r'$\Delta r_{BN,' + str(idx) + '}$')
            plt.legend(loc='lower right')
            plt.xlabel('Time [orbits]')
            plt.ylabel('Trajectory Differences [m]')
            pltName = fileName + "3" + orbitCase + str(int(useSphericalHarmonics)) + 'Earth'
            figureList[pltName] = plt.figure(3)

            finalDiff = np.linalg.norm(Deltar[-1])

        else:

            plt.figure(2)
            fig = plt.gcf()
            ax = fig.gca()
            ax.ticklabel_format(useOffset=False, style='plain')
            smaData = []
            for idx in range(0, len(posData)):
                oeData = orbitalMotion.rv2elem(mu, posData[idx], velData[idx])
                smaData.append(oeData.a / 1000.)
            plt.plot(timeAxis * macros.NANO2SEC / P, smaData, color='#aa0000',
                    )
            plt.xlabel('Time [orbits]')
            plt.ylabel('SMA [km]')

        pltName = fileName + "2" + orbitCase + str(int(useSphericalHarmonics)) + 'Earth'
        figureList[pltName] = plt.figure(2)
        return figureList, finalDiff
