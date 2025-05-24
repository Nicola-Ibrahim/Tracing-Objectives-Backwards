from dataclasses import dataclass

from .vehicle import Vehicle


@dataclass(frozen=True)
class ProblemSpec:
    """Contains complete specification for an optimization problem instance.

    Attributes:
        target_distance_km: Mission distance in kilometers
        vehicle: Configured Vehicle instance for simulation
    """

    target_distance_km: float
    vehicle: Vehicle
