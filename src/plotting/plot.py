import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import PngImagePlugin
from pathlib import Path
import cartopy.crs as ccrs
import pymap3d as pm
from datetime import datetime, timedelta, timezone

from object_definitions.Config_def import Config
from object_definitions.SimData_def import SimData
from plotting.DataLoader_def import DataLoader
from plotting.DataProcessor_def import DataProcessor


PLT_SAVE_FOLDER_PATH = Path('data/sim_plt')
PLT_HEIGHT = 6.0
PLT_WIDTH = 16.0


def plot(cfg: Config) -> None:
    """
    Description coming here...
    
    Args:
        cfg (Config): Simulation config
    """
    # plot() does nothing if show_plots == false and save_plots == false.
    # Exit if that's the case
    if (not cfg.show_plots) and (not cfg.save_plots):
        logging.debug("Config settings specify: show_plots == false and save_plots == false, which makes the plotting function obsolete. -> Exiting plot()...")
        return

    # Stop plots to clutter the terminal with debug info
    quiet_plots()
    
    ##################################
    # Fetch and Load simulation data #
    ##################################

    # Initialize data loader and processor objects
    data_loader = DataLoader()

    if not cfg.bypass_sim_to_plot:
        # plot results from this simulation 
        data_timestamp = cfg.timestamp_str  
    else:
        # plot results from a previous simulation
        data_timestamp = cfg.data_timestamp_to_plot

    # Get all datafiles with the corresponding timestamp
    datafiles_to_plot = data_loader.get_datafiles_by_timestamp(data_timestamp)
    
    # Load simulation data
    skf_sim_data, bsk_sim_data = data_loader.load_and_separate_data(datafiles_to_plot)
    
    ############
    # Plotting #
    ############

    # plot_groundtrack_comparison(cfg, skf_sim_data, bsk_sim_data)

    plot_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_state_diff(cfg, skf_sim_data, bsk_sim_data)
    
    # plot_rel_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_rel_state_diff(cfg, skf_sim_data, bsk_sim_data)

    # plot_altitude_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_state_abs_diff(cfg, skf_sim_data, bsk_sim_data)

    
    

#################################
# Plotting function definitions #
#################################

def plot_groundtrack_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot the ground-track (projected trajectory on Earth) for each satellite,
    comparing Skyfield (red) and Basilisk (green).

    Positions are given in ECI and converted to geodetic (lat, lon) using pymap3d.eci2geodetic.
    One map figure is generated per satellite.
    """
    main_plt_identifier = "GroundTrack"  # Used in saved plot name

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Epoch for t = 0 (assumed to live in cfg; adjust if you store it elsewhere)
    epoch, duration, deltaT = get_simulation_time(cfg) 

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Shape sanity checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf_sec = np.ravel(skf.time)
        t_bsk_sec = np.ravel(bsk.time)

        # Convert to datetime arrays for pymap3d
        t_skf_dt = [epoch + timedelta(seconds=float(t)) for t in t_skf_sec]
        t_bsk_dt = [epoch + timedelta(seconds=float(t)) for t in t_bsk_sec]

        # Unpack ECI position components
        x_skf, y_skf, z_skf = skf.pos[0, :], skf.pos[1, :], skf.pos[2, :]
        x_bsk, y_bsk, z_bsk = bsk.pos[0, :], bsk.pos[1, :], bsk.pos[2, :]

        # ECI -> geodetic (lat [deg], lon [deg], alt [m])
        # pymap3d will broadcast over arrays
        lat_skf, lon_skf, _ = pm.eci2geodetic(x_skf, y_skf, z_skf, t_skf_dt)
        lat_bsk, lon_bsk, _ = pm.eci2geodetic(x_bsk, y_bsk, z_bsk, t_bsk_dt)

        # Create map figure for this satellite
        fig = plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Background: simple stock image + coastlines
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

        # Plot ground tracks
        sat_name = (
            skf.satellite_name
            if getattr(skf, "satellite_name", None)
            else f"Satellite {i+1}"
        )

        # Skyfield: red
        ax.plot(
            lon_skf,
            lat_skf,
            color="red",
            linewidth=1.5,
            linestyle="-",
            label=f"SKF {sat_name}",
            transform=ccrs.Geodetic(),
        )

        # Basilisk: green
        ax.plot(
            lon_bsk,
            lat_bsk,
            color="green",
            linewidth=1.5,
            linestyle=":",
            label=f"BSK {sat_name}",
            transform=ccrs.Geodetic(),
        )

        ax.set_title(f"Ground track comparison — {sat_name}")
        ax.legend(loc="lower left")

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_pos_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot position vectors for each satellite from two SimData datasets.

    For satellite i (0..n-1), creates a new figure and plots:
      - SKF position components (x,y,z) as different blue shades
      - BSK position components (x,y,z) as different green shades
    The time axis comes from each dataset's own `time` array.

    Args:
        skf_sim_data (SimData):
        bsk_sim_data (SimData):
    Returns:
        None
    """
    main_plt_identifier = "PosComp" # Used in the saved plot name
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
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]

        # Create a new figure for this satellite; no explicit numbering to avoid conflicts
        plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
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
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("ECI Position (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_rel_pos_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot the position vectors relative to the formation chief satellite expressed in the RTN frame.

    For satellite i (0..n-1), creates a new figure and plots:
      - SKF position components (x,y,z) as different blue shades
      - BSK position components (x,y,z) as different green shades
    The time axis comes from each dataset's own `time` array.

    Args:
        skf_sim_data (SimData):
        bsk_sim_data (SimData):
    Returns:
        None
    """
    main_plt_identifier = "RelPosComp"

    # Initialize data processor
    data_processor = DataProcessor()

    # Calculate the relative position vectors from every follower satellites 
    # to the chief satellite expressed in RTN frame, and set results in rel_data attribute
    data_processor.calculate_relative_formation_movement_rtc(skf_sim_data)
    data_processor.calculate_relative_formation_movement_rtc(bsk_sim_data)

    skf_list = skf_sim_data.rel_data
    bsk_list = bsk_sim_data.rel_data

    assert skf_list is not None and bsk_list is not None

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

    # Get the chief satellite name
    skf_chief_sat_name = skf_list[0].satellite_name
    bsk_chief_sat_name = bsk_list[0].satellite_name
    if skf_chief_sat_name != bsk_chief_sat_name:
        raise ValueError(f"Mismatch between the first satellite (chief) name in skf_list ({skf_chief_sat_name} "
                         f"and bsk_list ({bsk_chief_sat_name})")
    chief_sat_name = skf_chief_sat_name

    for i in range(1, n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks (won't fail if shapes are as specified)
        if skf.rel_pos.shape[0] != 3 or bsk.rel_pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]

        # Create a new figure for this satellite; no explicit numbering to avoid conflicts
        plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.gca()

        # Plot SKF (blue shades)
        for comp in range(3):
            ax.plot(
                t_skf,
                skf.rel_pos[comp, :],
                label=f"SKF {comp_labels[comp]}",
                linewidth=1.8,
                color=skf_colors[comp],
            )

        # Plot BSK (green shades)
        for comp in range(3):
            ax.plot(
                t_bsk,
                bsk.rel_pos[comp, :],
                label=f"BSK {comp_labels[comp]}",
                linewidth=1.8,
                linestyle="--",
                color=bsk_colors[comp],
            )

        sat_name = skf.satellite_name if getattr(skf, "satellite_name", None) else f"Satellite {i+1}"
        ax.set_title(f"Relative position between {sat_name} (follower) and {chief_sat_name} (chief)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("RTN Δposition (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)
    

def plot_simulator_state_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    For each satellite i, create a figure with two stacked subplots:
      Top:  (bsk.pos - skf.pos) components over time
      Bottom: (bsk.vel - skf.vel) components over time
    BSK data are interpolated onto the SKF time grid if their time vectors differ.
    Uses colors: x=red, y=green, z=blue. Does not call plt.show().
    """
    main_plt_identifier = "SimDiff" # Part of the saved figure's name

    # Colors for components: x (red), y (green), z (blue)
    COMP_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # r, g, b
    COMP_LABELS = ["x", "y", "z"]

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data
    data_processor = DataProcessor()

    if len(skf_list) != len(bsk_list):
        raise ValueError(f"Satellite count mismatch: SKF={len(skf_list)}, BSK={len(bsk_list)}")

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"[sat {i}] pos must be shape (3, n)")
        if skf.vel.shape[0] != 3 or bsk.vel.shape[0] != 3:
            raise ValueError(f"[sat {i}] vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]
        if t_skf.size != skf.pos.shape[1] or t_skf.size != skf.vel.shape[1]:
            raise ValueError(f"[sat {i}] SKF time length must match pos/vel columns")
        if t_bsk.size != bsk.pos.shape[1] or t_bsk.size != bsk.vel.shape[1]:
            raise ValueError(f"[sat {i}] BSK time length must match pos/vel columns")

        t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf.pos)
        _,     skf_vel = data_processor.ensure_increasing(t_skf, skf.vel)  # skf_pos/vel share t_skf order

        t_bsk, bsk_pos = data_processor.ensure_increasing(t_bsk, bsk.pos)
        _,     bsk_vel = data_processor.ensure_increasing(t_bsk, bsk.vel)

        # Interpolate BSK onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_pos, t_skf)
            bsk_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_vel, t_skf)
        else:
            bsk_pos_on_skf = bsk_pos
            bsk_vel_on_skf = bsk_vel

        # Differences
        dpos = bsk_pos_on_skf - skf_pos
        dvel = bsk_vel_on_skf - skf_vel

        # Create figure with two stacked subplots; no explicit numbering to avoid conflicts
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax_pos, ax_vel = axes

        # Top: position diffs
        for comp in range(3):
            ax_pos.plot(
                t_skf, dpos[comp],
                label=f"Δpos {COMP_LABELS[comp]}",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_pos.set_ylabel("ECI Δposition (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=3)

        # Bottom: velocity diffs
        for comp in range(3):
            ax_vel.plot(
                t_skf, dvel[comp],
                label=f"Δvel {COMP_LABELS[comp]}",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("ECI Δvelocity (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=3)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i+1}"
        fig.suptitle(f"Simulator difference — {sat_name}")
        # fig.tight_layout(rect=[0., 0., 1., 0.96])
        
        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_simulator_rel_state_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    # TODO: Fix chat language
    """
    Plot simulator differences between relative RTN vectors (BSK - SKF).

    Assumes `DataProcessor.calculate_relative_formation_movement_rtc(...)` has already
    been called on both `skf_sim_data` and `bsk_sim_data`, so that `rel_data`
    is populated for each.

    For each follower satellite i (i = 1..n_sats-1), creates a figure with two stacked subplots:
      Top:    (bsk.rel_pos - skf.rel_pos) in RTN, components over time
      Bottom: (bsk.rel_vel - skf.rel_vel) in RTN, components over time

    Uses colors: x=red, y=green, z=blue.
    """
    main_plt_identifier = "SimRelDiff"  # Part of the saved figure's name

    # Colors for components: x (red), y (green), z (blue)
    COMP_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # r, g, b
    COMP_LABELS = ["x", "y", "z"]

    data_processor = DataProcessor()

    skf_list = skf_sim_data.rel_data
    bsk_list = bsk_sim_data.rel_data

    # Ensure relative data has been computed
    if skf_list is None or bsk_list is None:
        raise ValueError(
            "Relative RTN data not found in skf_sim_data/bsk_sim_data. "
            "Make sure plot_rel_pos_comparison (or the underlying "
            "calculate_relative_formation_movement_rtc) has been run first."
        )

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch in rel_data: SKF={len(skf_list)}, BSK={len(bsk_list)}"
        )

    n_sats = len(skf_list)
    if n_sats <= 1:
        # Need at least a chief + one follower
        return

    # Chief is assumed to be index 0 in rel_data (as in plot_rel_pos_comparison)
    skf_chief_sat_name = skf_list[0].satellite_name
    bsk_chief_sat_name = bsk_list[0].satellite_name
    if skf_chief_sat_name != bsk_chief_sat_name:
        raise ValueError(
            "Mismatch between chief satellite names in rel_data: "
            f"SKF chief={skf_chief_sat_name}, BSK chief={bsk_chief_sat_name}"
        )
    chief_sat_name = skf_chief_sat_name

    # Loop over followers only (1..n_sats-1)
    for i in range(1, n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks for relative pos/vel
        if skf.rel_pos.shape[0] != 3 or bsk.rel_pos.shape[0] != 3:
            raise ValueError(f"[rel sat {i}] rel_pos must be shape (3, n)")
        if not hasattr(skf, "rel_vel") or not hasattr(bsk, "rel_vel"):
            raise ValueError(
                f"[rel sat {i}] rel_vel attribute missing on relative data objects. "
                "Ensure calculate_relative_formation_movement_rtc computes rel_vel."
            )
        if skf.rel_vel.shape[0] != 3 or bsk.rel_vel.shape[0] != 3:
            raise ValueError(f"[rel sat {i}] rel_vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60 * 60)  # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # Time [hours]

        if t_skf.size != skf.rel_pos.shape[1] or t_skf.size != skf.rel_vel.shape[1]:
            raise ValueError(
                f"[rel sat {i}] SKF time length must match rel_pos/rel_vel columns"
            )
        if t_bsk.size != bsk.rel_pos.shape[1] or t_bsk.size != bsk.rel_vel.shape[1]:
            raise ValueError(
                f"[rel sat {i}] BSK time length must match rel_pos/rel_vel columns"
            )

        # Ensure increasing time and aligned ordering
        t_skf, skf_rel_pos = data_processor.ensure_increasing(t_skf, skf.rel_pos)
        _,     skf_rel_vel = data_processor.ensure_increasing(t_skf, skf.rel_vel)

        t_bsk, bsk_rel_pos = data_processor.ensure_increasing(t_bsk, bsk.rel_pos)
        _,     bsk_rel_vel = data_processor.ensure_increasing(t_bsk, bsk.rel_vel)

        # Interpolate BSK relative data onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_rel_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_rel_pos, t_skf)
            bsk_rel_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_rel_vel, t_skf)
        else:
            bsk_rel_pos_on_skf = bsk_rel_pos
            bsk_rel_vel_on_skf = bsk_rel_vel

        # Differences (BSK - SKF) in RTN
        d_rel_pos = bsk_rel_pos_on_skf - skf_rel_pos
        d_rel_vel = bsk_rel_vel_on_skf - skf_rel_vel

        # Create figure with two stacked subplots; no explicit numbering
        fig, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT)
        )
        ax_pos, ax_vel = axes

        # Top: relative position diffs (RTN)
        for comp in range(3):
            ax_pos.plot(
                t_skf,
                d_rel_pos[comp],
                label=f"Δrel_pos {COMP_LABELS[comp]} (RTN)",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_pos.set_ylabel("RTN Δrel position (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=3)

        # Bottom: relative velocity diffs (RTN)
        for comp in range(3):
            ax_vel.plot(
                t_skf,
                d_rel_vel[comp],
                label=f"Δrel_vel {COMP_LABELS[comp]} (RTN)",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("RTN Δrel velocity (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=3)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i}"
        fig.suptitle(
            f"Simulator RTN relative difference — {sat_name} (follower) vs {chief_sat_name} (chief)"
        )

        # Save via your existing helper
        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}_vs_{chief_sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_altitude_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot altitude vs time for each satellite, for both Skyfield and Basilisk, in a single figure.

    Altitude is defined as ||pos|| - R_earth_mean, where pos is the ECI position vector.
    """
    main_plt_identifier = "AltComp"  # Used in the saved plot name
    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Mean Earth radius [m] (approx. WGS-84 mean radius)
    EARTH_MEAN_RADIUS_M = 6371e3

    # Choose some colors for different satellites (will cycle if more sats than colors)
    sat_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    # Create a single figure for all satellites
    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    ax = plt.gca()

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D and convert to hours
        t_skf = np.ravel(skf.time) / (60 * 60)  # [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # [hours]

        # Compute altitude = ||pos|| - R_earth_mean
        skf_r = np.linalg.norm(skf.pos, axis=0)
        bsk_r = np.linalg.norm(bsk.pos, axis=0)
        skf_alt = skf_r - EARTH_MEAN_RADIUS_M
        bsk_alt = bsk_r - EARTH_MEAN_RADIUS_M

        color = sat_colors[i % len(sat_colors)]
        sat_name = (
            skf.satellite_name
            if getattr(skf, "satellite_name", None)
            else f"Satellite {i+1}"
        )

        # Skyfield: solid line
        ax.plot(
            t_skf,
            skf_alt,
            label=f"SKF {sat_name}",
            linewidth=1.8,
            linestyle="-",
            color=color,
        )

        # Basilisk: dashed line, same color
        ax.plot(
            t_bsk,
            bsk_alt,
            label=f"BSK {sat_name}",
            linewidth=1.8,
            linestyle="--",
            color=color,
        )

    ax.set_title("Altitude comparison for all satellites")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig = plt.gcf()
    plt_identifier = f"{main_plt_identifier}_AllSats"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_simulator_state_abs_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    For each satellite i, create a figure with two stacked subplots:
      Top:    absolute difference in position magnitude  | |r_BSK| - |r_SKF| |
      Bottom: absolute difference in velocity magnitude | |v_BSK| - |v_SKF| |
    BSK data are interpolated onto the SKF time grid if their time vectors differ.
    """
    main_plt_identifier = "SimAbsDiff"  # Part of the saved figure's name

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data
    data_processor = DataProcessor()

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF={len(skf_list)}, BSK={len(bsk_list)}"
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"[sat {i}] pos must be shape (3, n)")
        if skf.vel.shape[0] != 3 or bsk.vel.shape[0] != 3:
            raise ValueError(f"[sat {i}] vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60 * 60)  # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # Time [hours]
        if t_skf.size != skf.pos.shape[1] or t_skf.size != skf.vel.shape[1]:
            raise ValueError(
                f"[sat {i}] SKF time length must match pos/vel columns"
            )
        if t_bsk.size != bsk.pos.shape[1] or t_bsk.size != bsk.vel.shape[1]:
            raise ValueError(
                f"[sat {i}] BSK time length must match pos/vel columns"
            )

        t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf.pos)
        _,     skf_vel = data_processor.ensure_increasing(t_skf, skf.vel)

        t_bsk, bsk_pos = data_processor.ensure_increasing(t_bsk, bsk.pos)
        _,     bsk_vel = data_processor.ensure_increasing(t_bsk, bsk.vel)

        # Interpolate BSK onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_pos, t_skf)
            bsk_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_vel, t_skf)
        else:
            bsk_pos_on_skf = bsk_pos
            bsk_vel_on_skf = bsk_vel

        # Magnitudes
        skf_pos_norm = np.linalg.norm(skf_pos, axis=0)
        bsk_pos_norm = np.linalg.norm(bsk_pos_on_skf, axis=0)
        skf_vel_norm = np.linalg.norm(skf_vel, axis=0)
        bsk_vel_norm = np.linalg.norm(bsk_vel_on_skf, axis=0)

        # Absolute scalar differences
        dpos = bsk_pos_norm - skf_pos_norm
        dvel = bsk_vel_norm - skf_vel_norm

        # Create figure with two stacked subplots; no explicit numbering to avoid conflicts
        fig, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT)
        )
        ax_pos, ax_vel = axes

        # Top: absolute position magnitude diff
        ax_pos.plot(
            t_skf,
            dpos,
            label="| |r_BSK| - |r_SKF| |",
            linewidth=1.8,
            color="#d62728",  # red
        )
        ax_pos.set_ylabel("Abs Δ|r| (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=1)

        # Bottom: absolute velocity magnitude diff
        ax_vel.plot(
            t_skf,
            dvel,
            label="| |v_BSK| - |v_SKF| |",
            linewidth=1.8,
            color="#1f77b4",  # blue
        )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("Abs Δ|v| (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=1)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i+1}"
        fig.suptitle(f"Simulator absolute difference — {sat_name}")

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)



########################################
# Plotting helper function definitions #
########################################

def conditional_save_plot(cfg: Config, fig: Figure, plt_identifier: str) -> None:
    """
    Save a matplotlib Figure to PLT_SAVE_FOLDER_PATH with a standardized filename iff cfg.save_plots == true.

    Filename: f"{data_timestamp}_{plt_identifier}.png"

    Args:
        fig: Matplotlib Figure object to save.
        data_timestamp: Timestamp string associated with the data (e.g. "20251106_003128").
        plt_identifier: Short identifier for the plot type/content (e.g. "pos_comp_sat1").
    """
    # Only save plots if cfg.save_plots == true
    if not cfg.save_plots:
        return
    
    # Get correct timestamp
    if not cfg.bypass_sim_to_plot:
        # plot results from this simulation 
        data_timestamp = cfg.timestamp_str  
    else:
        # plot results from a previous simulation
        data_timestamp = cfg.data_timestamp_to_plot

    # Ensure target directory exists
    PLT_SAVE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    filename = f"{data_timestamp}_{plt_identifier}.png"
    save_path = PLT_SAVE_FOLDER_PATH / filename

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    logging.debug(f"Saved figure: {filename}")


def conditional_show_plot(cfg: Config) -> None:
    # Show plot only if cfg.show_plots == true
    if cfg.show_plots:
        plt.show()
    else:
        return
 

def quiet_plots() -> None:
    # Only show warnings and errors globally
    logging.basicConfig(level=logging.WARNING)

    # Matplotlib: silence backend + font-manager chatter
    mpl.set_loglevel("warning")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Pillow (PIL): silence PNG chunk debug like "STREAM b'IHDR'"
    #PngImagePlugin.debug = False
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def get_simulation_time(cfg: Config) -> tuple[datetime, int, int]:
    """
    Repurposed from 'SkyfieldSimulator.get_skf_simulation_time
    Parses and converts simulation time parameters from default config into skyfield-compatible types.

    RETURNS:
        startTime       (datetime) utc time
        duration        (float) seconds
        deltaT          (int) seconds
    """
    cfg_startTime = cfg.startTime
    cfg_duration = cfg.simulationDuration
    cfg_deltaT = cfg.s_set.deltaT

    # Parse simulation starttime and convert into a UTC datetime object
    try:
        startTime = datetime.strptime(cfg_startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
    except:
        raise ValueError("Failed to convert config parameter 'startTime' to a datetime object.")
    
    # Convert cfg_duration float(hours) -> int(seconds)
    duration = int(3600 * cfg_duration) 
    if duration < (3600 * cfg_duration):
        raise ValueError("Type conversion float -> int for 'skf_duration' caused a reduction in its value!")

    # deltaT
    deltaT = int(cfg_deltaT)
    if deltaT < cfg_deltaT:
        raise ValueError("Type conversion float -> int for 'skf_deltaT' caused a reduction in its value!")
    
    return startTime, duration, deltaT