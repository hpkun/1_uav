"""Li et al. (2023) MADSAC multi-UAV paper reproduction."""

from .environment import PaperUAVCombatEnv, PaperAirCombatGeometry, compute_paper_geometry
from .models import Aircraft, AircraftSpec, AircraftState, ControlCommand, TargetCommand

__all__ = ["Aircraft", "AircraftSpec", "AircraftState", "ControlCommand", "TargetCommand", "PaperAirCombatGeometry", "compute_paper_geometry", "PaperUAVCombatEnv"]
