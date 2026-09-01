"""HM3D occupancy grids as a planning world for SafeDiffuser + DSTT."""
import pathlib


def plans_dir_for(grid_path):
    """Where a grid's planning outputs belong: a `plans/` sibling of `grids/`.

    Derived from the grid rather than fixed, so pointing the planner at a
    different run (calib vs oracle) sends its results to that run's own folder
    instead of silently overwriting the previous one. Results then sit beside
    the map they were produced from, in the MASt3R-SLAM tree that owns both.
    """
    grid_path = pathlib.Path(grid_path).resolve()
    grids_dir = grid_path.parent
    if grids_dir.name == "grids":
        return grids_dir.parent / "plans"
    # A grid kept somewhere other than a run's grids/ folder: keep its outputs
    # next to it rather than guessing at a tree that may not exist.
    return grids_dir / "plans"
