import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import PngImagePlugin
from pathlib import Path
from typing import Optional

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
        alt_datafilename_to_plot (Optional[str]): Only plots the current simulation if this is set to 'None'. 
            str containing the timestamp (the part of the filename before '_skf' or '_bsk').
            of the datafiles 
    """

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

    # plot_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_diff(cfg, skf_sim_data, bsk_sim_data)
    
    plot_rel_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    plot_simulator_rel_pos_diff(cfg, skf_sim_data, bsk_sim_data)

    
    


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
        # plt.show()


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
        plt.show()
    

def plot_simulator_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
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
        # plt.show()


def plot_simulator_rel_pos_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
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
        # plt.show()  # Optional, kept commented as in your other functions


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

