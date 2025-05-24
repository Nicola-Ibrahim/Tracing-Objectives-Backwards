from dataclasses import dataclass


@dataclass(frozen=True)
class Vehicle:
    """Represents the complete physical and operational configuration of an electric vehicle.

    Encapsulates all vehicle-specific parameters required for motion simulation and energy calculations.

    Attributes:
        max_battery_kwh: Total battery capacity in kilowatt-hours
        max_speed_mps: Maximum allowed speed in meters per second
        mass_kg: Total vehicle mass including payload in kilograms
        air_density_kg_per_m3: Ambient air density (default 1.225 kg/m³ at sea level)
        drag_coefficient: Aerodynamic drag coefficient (unitless)
        frontal_area_m2: Vehicle frontal area in square meters
        rolling_resistance_coeff: Tire rolling resistance coefficient (unitless)
        motor_efficiency: Drivetrain efficiency during acceleration (0.0-1.0)
        regenerative_efficiency: Energy recovery efficiency during braking (0.0-1.0)
        auxiliary_power_watt: Constant auxiliary systems power draw in watts
        min_acceleration: Minimum comfortable acceleration in m/s²
        max_acceleration: Maximum allowed acceleration in m/s²
        initial_state_of_charge: Starting battery charge percentage (0.0-1.0)
    """

    max_battery_kwh: float
    max_speed_mps: float
    mass_kg: float
    air_density_kg_per_m3: float = 1.225
    drag_coefficient: float = 0.24
    frontal_area_m2: float = 2.4
    rolling_resistance_coeff: float = 0.008
    motor_efficiency: float = 0.85
    regenerative_efficiency: float = 0.70
    auxiliary_power_watt: float = 300
    min_acceleration: float = 0.2
    max_acceleration: float = 2.0
    initial_state_of_charge: float = 0.9

    def __post_init__(self):
        """Validate vehicle parameters for physical plausibility."""
        positive_attrs = [
            self.max_battery_kwh,
            self.max_speed_mps,
            self.mass_kg,
            self.air_density_kg_per_m3,
            self.frontal_area_m2,
            self.rolling_resistance_coeff,
            self.auxiliary_power_watt,
        ]

        efficiency_attrs = [self.motor_efficiency, self.regenerative_efficiency]

        if any(v <= 0 for v in positive_attrs):
            raise ValueError("All physical parameters must be positive")

        if any(not (0 <= eff <= 1) for eff in efficiency_attrs):
            raise ValueError("Efficiency values must be between 0 and 1")

        if not 0 <= self.initial_state_of_charge <= 1:
            raise ValueError("Initial SOC must be between 0 and 1")
