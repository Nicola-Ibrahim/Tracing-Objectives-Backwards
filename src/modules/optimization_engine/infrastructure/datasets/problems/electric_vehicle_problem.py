from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from ....domain.datasets.interfaces.base_problem import BaseProblem


class Vehicle(BaseModel):
    """Pydantic model representing electric vehicle configuration with validation."""

    model_config = ConfigDict(validate_assignment=True, frozen=True)

    # Core vehicle parameters
    max_battery_kwh: float = Field(
        default=10.0, gt=0, description="Total battery capacity in kWh"
    )
    max_speed_mps: float = Field(
        default=20.0, gt=0, description="Maximum allowed speed in m/s"
    )
    mass_kg: float = Field(
        default=1000.0, gt=0, description="Total mass including payload in kg"
    )

    # Aerodynamic properties
    air_density_kg_per_m3: float = Field(
        default=1.225, gt=0, description="Ambient air density"
    )
    drag_coefficient: float = Field(
        default=0.24, ge=0, le=1, description="Aerodynamic drag coefficient"
    )
    frontal_area_m2: float = Field(
        default=2.4, gt=0, description="Frontal cross-sectional area in m²"
    )

    # Mechanical properties
    rolling_resistance_coeff: float = Field(
        default=0.008, gt=0, description="Tire rolling resistance coefficient"
    )
    motor_efficiency: float = Field(
        default=0.85,
        gt=0,
        lt=1,
        description="Drivetrain efficiency during acceleration",
    )
    regenerative_efficiency: float = Field(
        default=0.70,
        gt=0,
        lt=1,
        description="Energy recovery efficiency during braking",
    )

    # Operational parameters
    auxiliary_power_watt: float = Field(
        default=300.0, ge=0, description="Constant auxiliary power draw in watts"
    )
    min_acceleration: float = Field(
        default=0.2, description="Minimum comfortable acceleration in m/s²"
    )
    max_acceleration: float = Field(
        default=2.0, gt=0, description="Maximum allowed acceleration in m/s²"
    )

    # Energy management
    initial_state_of_charge: float = Field(
        default=0.9, ge=0, le=1, description="Starting battery charge (0.0-1.0)"
    )
    safety_soc_margin: float = Field(
        default=0.2, ge=0, le=1, description="Safety margin for battery discharge"
    )

    @model_validator(mode="after")
    def validate_vehicle(self) -> "Vehicle":
        """Validate interdependent parameters and physical constraints"""
        # Validate acceleration bounds
        if self.min_acceleration >= self.max_acceleration:
            raise ValueError(
                "Minimum acceleration must be less than maximum acceleration"
            )

        # Validate SOC safety margin
        if self.initial_state_of_charge < self.safety_soc_margin:
            raise ValueError("Initial SOC must be greater than safety margin")

        return self

    @computed_field
    @property
    def available_energy_kwh(self) -> float:
        """Available energy considering safety margin"""
        return (
            self.initial_state_of_charge - self.safety_soc_margin
        ) * self.max_battery_kwh

    @computed_field
    @property
    def kinetic_energy_coeff(self) -> float:
        """Precomputed coefficient for kinetic energy calculations"""
        return (
            0.5
            * self.air_density_kg_per_m3
            * self.drag_coefficient
            * self.frontal_area_m2
        )


class ElectricalVehicleProblemConfig(BaseModel):
    """
    Problem specification for optimization algorithms.
    Contains the target distance for the mission and the vehicle configuration.
    """

    target_distance_km: float = Field(
        ..., ge=0.0, description="Mission distance in kilometers"
    )

    n_var: int = Field(
        ...,
        ge=1,
        description="Number of decision variables in the optimization problem",
    )
    n_obj: int = Field(..., ge=1, description="Number of objectives to optimize")
    n_constr: int = Field(
        0, ge=0, description="Number of constraints in the optimization problem"
    )
    xl: float = Field(..., description="Lower bounds for decision variables")
    xu: float = Field(..., description="Upper bounds for decision variables")


class EVControlProblem(BaseProblem):
    """
    Multi-objective optimization problem for electric vehicle motion control.
    Inherits from EVProblem to implement specific evaluation logic while
    maintaining interface compatibility with pymoo's optimization framework.

    Implements the _evaluate() method required by pymoo's Problem class.
    """

    def __init__(self, config: ElectricalVehicleProblemConfig, vehicle: Vehicle):
        """
        Initialize motion control problem with complete specification.

        Args:
            config: Contains vehicle configuration and mission parameters
        """

        self.vehicle = vehicle

        # Problem-specific setup
        self.target_distance_m = config.target_distance_km * 1000

        super().__init__(
            n_var=config.n_var,
            n_obj=config.n_obj,
            n_constr=config.n_constr,
            xl=np.array([config.xl] * config.n_var),
            xu=np.array([config.xu] * config.n_var),
        )

    def _clamp_velocity(self, velocity: float) -> float:
        """Enforce vehicle speed limits using configured maximum."""
        return np.clip(velocity, 0, self.vehicle.max_speed_mps)

    def _calculate_resistive_forces(self, velocity: float) -> tuple[float, float]:
        """
        Calculate opposing forces affecting vehicle motion.

        Returns:
            Tuple of (aerodynamic_drag_force [N], rolling_resistance_force [N])
        """
        aerodynamic_drag = (
            0.5
            * self.vehicle.air_density_kg_per_m3
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * velocity**2
        )
        rolling_resistance = (
            self.vehicle.rolling_resistance_coeff
            * self.vehicle.mass_kg
            * 9.81  # Gravitational constant
        )
        return aerodynamic_drag, rolling_resistance

    def _simulate_vehicle_motion(
        self, acceleration_rate: float, deceleration_rate: float
    ) -> tuple[float, float, float]:
        """
        Simulate vehicle trip through acceleration, cruising, and deceleration phases

        Args:
            acceleration_rate: Control input for acceleration (m/s²)
            deceleration_rate: Control input for braking (m/s²)

        Returns:
            tuple: (total_time_minutes, peak_velocity_mps, actual_distance_m)
        """
        SIMULATION_TIMESTEP = 0.1  # Seconds between state updates

        time_elapsed = 0.0
        distance_traveled = 0.0
        current_velocity = 0.0
        peak_velocity = 0.0

        # Phase 1: Acceleration to cruising speed
        while (
            current_velocity < self.vehicle.max_speed_mps
            and distance_traveled < self.target_distance_m
        ):
            current_velocity = self._clamp_velocity(
                current_velocity + acceleration_rate * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP
            time_elapsed += SIMULATION_TIMESTEP
            peak_velocity = max(peak_velocity, current_velocity)

        # Phase 2: Constant speed cruising (if distance remains)
        remaining_distance = self.target_distance_m - distance_traveled
        if remaining_distance > 0:
            cruise_duration = remaining_distance / self.vehicle.max_speed_mps
            time_elapsed += cruise_duration
            distance_traveled = self.target_distance_m  # Fully covered

        # Phase 3: Controlled deceleration to stop
        while distance_traveled < self.target_distance_m and current_velocity > 0:
            current_velocity = self._clamp_velocity(
                current_velocity - deceleration_rate * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP
            time_elapsed += SIMULATION_TIMESTEP

        return time_elapsed / 60, peak_velocity, distance_traveled

    def _calculate_energy_consumption(
        self, acceleration_rate: float, deceleration_rate: float
    ) -> float:
        """
        Calculate total energy expenditure during the trip

        Args:
            acceleration_rate: Control input for acceleration (m/s²)
            deceleration_rate: Control input for braking (m/s²)

        Returns:
            float: Total energy consumed in kWh
        """
        SIMULATION_TIMESTEP = 0.1  # Seconds between energy calculations
        energy_watt_hours = 0.0
        current_velocity = 0.0
        distance_traveled = 0.0

        while distance_traveled < self.target_distance_m:
            # Determine current operational phase
            if current_velocity < self.vehicle.max_speed_mps:
                operation_phase = "accelerating"
                current_acceleration = acceleration_rate
            else:
                braking_distance = current_velocity**2 / (2 * deceleration_rate)
                remaining_distance = self.target_distance_m - distance_traveled
                operation_phase = (
                    "braking" if remaining_distance <= braking_distance else "cruising"
                )
                current_acceleration = (
                    -deceleration_rate if operation_phase == "braking" else 0
                )

            # Calculate power requirements for current state
            drag_force, rolling_force = self._calculate_resistive_forces(
                current_velocity
            )

            if operation_phase == "accelerating":
                mechanical_power = (
                    self.vehicle.mass_kg * current_acceleration
                    + drag_force
                    + rolling_force
                ) * current_velocity
                effective_power = mechanical_power / self.vehicle.motor_efficiency
            elif operation_phase == "braking":
                mechanical_power = (
                    self.vehicle.mass_kg * current_acceleration
                    + drag_force
                    + rolling_force
                ) * current_velocity
                effective_power = (
                    mechanical_power * self.vehicle.regenerative_efficiency
                )
            else:  # Cruising
                mechanical_power = (drag_force + rolling_force) * current_velocity
                effective_power = mechanical_power / self.vehicle.motor_efficiency

            # Sum total energy consumption with auxiliary systems
            total_power = effective_power + self.vehicle.auxiliary_power_watt
            energy_watt_hours += (
                total_power * SIMULATION_TIMESTEP / 3600
            )  # Convert to Wh

            # Update kinematic state
            current_velocity = self._clamp_velocity(
                current_velocity + current_acceleration * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP

        return energy_watt_hours / 1000  # Convert to kWh

    def _evaluate(
        self, population: np.ndarray, out: dict[str, Any], *args, **kwargs
    ) -> None:
        """
        Evaluate a population of candidate solutions

        Args:
            population: Array of candidate solutions (acceleration/deceleration pairs)
            out: Dictionary for storing objectives and constraints
        """
        objective_values = []
        constraint_violations = []

        for accel_rate, decel_rate in population:
            # Simulate vehicle motion with current control parameters
            trip_time, max_speed, final_distance = self._simulate_vehicle_motion(
                accel_rate, decel_rate
            )

            # Calculate energy expenditure
            energy_used = self._calculate_energy_consumption(accel_rate, decel_rate)

            # Calculate constraint violations (normalized)
            speed_violation = max(
                0, (max_speed - self.vehicle.max_speed_mps) / self.vehicle.max_speed_mps
            )
            energy_violation = max(
                0,
                (energy_used - self.vehicle.available_energy_kwh)
                / self.vehicle.available_energy_kwh,
            )

            # Control input constraint checks
            accel_violation = max(
                (self.vehicle.min_acceleration - accel_rate)
                / self.vehicle.min_acceleration
                if accel_rate < self.vehicle.min_acceleration
                else 0,
                (accel_rate - self.vehicle.max_acceleration)
                / self.vehicle.max_acceleration
                if accel_rate > self.vehicle.max_acceleration
                else 0,
            )
            decel_violation = max(
                (self.vehicle.min_acceleration - decel_rate)
                / self.vehicle.min_acceleration
                if decel_rate < self.vehicle.min_acceleration
                else 0,
                (decel_rate - self.vehicle.max_acceleration)
                / self.vehicle.max_acceleration
                if decel_rate > self.vehicle.max_acceleration
                else 0,
            )
            control_violation = max(accel_violation, decel_violation)

            # Store evaluation results
            objective_values.append([trip_time, energy_used])
            constraint_violations.append(
                [speed_violation, energy_violation, control_violation]
            )

        # Format outputs for optimization framework
        out["F"] = np.array(objective_values)
        out["G"] = np.array(constraint_violations)
