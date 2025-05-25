from typing import Any

import numpy as np

from .base import EVProblem
from .specs import ProblemSpec
from .vehicle import Vehicle


class EVControlProblem(EVProblem):
    """
    Multi-objective optimization problem for electric vehicle motion control.
    Inherits from EVProblem to implement specific evaluation logic while
    maintaining interface compatibility with pymoo's optimization framework.

    Implements the _evaluate() method required by pymoo's Problem class.
    """

    def __init__(self, spec: ProblemSpec, vehicle: Vehicle):
        """
        Initialize motion control problem with complete specification.

        Args:
            spec: Contains vehicle configuration and mission parameters
        """
        super().__init__(
            spec=spec,
            vehicle=vehicle,
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            n_constr=spec.n_constr,
            xl=np.array([spec.xl] * spec.n_var),
            xu=np.array([spec.xu] * spec.n_var),
        )

        # Problem-specific setup
        self.target_distance_m = spec.target_distance_km * 1000

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
