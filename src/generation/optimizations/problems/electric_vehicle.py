from typing import Any

import numpy as np

from .base import BaseEVProblem
from .specs import ProblemSpec


class EVControlProblem(BaseEVProblem):
    """
    Multi-objective optimization problem for electric vehicle motion control.
    Inherits from EVProblem to implement specific evaluation logic while
    maintaining interface compatibility with pymoo's optimization framework.

    Implements the _evaluate() method required by pymoo's Problem class.
    """

    def __init__(self, spec: ProblemSpec):
        """
        Initialize motion control problem with complete specification.

        Args:
            spec: Contains vehicle configuration and mission parameters
        """
        super().__init__(spec)

        # Problem-specific setup
        self.target_distance_m = spec.target_distance_km * 1000
        self.available_energy_kwh = (
            self.vehicle.initial_state_of_charge - 0.2
        ) * self.vehicle.max_battery_kwh

    def _get_num_variables(self) -> int:
        """2 decision variables: [acceleration, deceleration]"""
        return 2

    def _get_num_objectives(self) -> int:
        """2 objectives: minimize time and energy consumption"""
        return 2

    def _get_num_constraints(self) -> int:
        """3 constraints: speed, energy, and control limits"""
        return 3

    def _get_lower_bounds(self) -> np.ndarray:
        """Minimum acceleration/deceleration values"""
        return np.array([self.vehicle.min_acceleration] * 2)

    def _get_upper_bounds(self) -> np.ndarray:
        """Maximum acceleration/deceleration values"""
        return np.array([self.vehicle.max_acceleration] * 2)

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
        self, accel: float, decel: float
    ) -> tuple[float, float, float]:
        """
        Simulate complete vehicle trip through three phases:
        1. Acceleration to cruising speed
        2. Constant speed cruising (if distance remains)
        3. Controlled deceleration to stop

        Returns:
            tuple: (total_time_minutes, peak_velocity_mps, actual_distance_m)
        """
        SIMULATION_TIMESTEP = 0.1  # Seconds between state updates

        time_elapsed = 0.0
        distance_traveled = 0.0
        current_velocity = 0.0
        peak_velocity = 0.0

        # Phase 1: Acceleration to configured maximum speed
        while (
            current_velocity < self.vehicle.max_speed_mps
            and distance_traveled < self.target_distance_m
        ):
            current_velocity = self._clamp_velocity(
                current_velocity + accel * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP
            time_elapsed += SIMULATION_TIMESTEP
            peak_velocity = max(peak_velocity, current_velocity)

        # Phase 2: Cruising at maximum speed if distance remains
        remaining_distance = self.target_distance_m - distance_traveled
        if remaining_distance > 0:
            cruise_duration = remaining_distance / self.vehicle.max_speed_mps
            time_elapsed += cruise_duration
            distance_traveled = self.target_distance_m  # Mark complete

        # Phase 3: Controlled deceleration to stop
        while distance_traveled < self.target_distance_m and current_velocity > 0:
            current_velocity = self._clamp_velocity(
                current_velocity - decel * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP
            time_elapsed += SIMULATION_TIMESTEP

        return time_elapsed / 60, peak_velocity, distance_traveled

    def _calculate_energy_consumption(self, accel: float, decel: float) -> float:
        """Calculate total energy expenditure during trip in kWh."""
        SIMULATION_TIMESTEP = 0.1
        energy_wh = 0.0
        current_velocity = 0.0
        distance_traveled = 0.0

        while distance_traveled < self.target_distance_m:
            # Determine current operational phase
            if current_velocity < self.vehicle.max_speed_mps:
                phase = "accelerating"
                current_accel = accel
            else:
                braking_distance = current_velocity**2 / (2 * decel)
                remaining_distance = self.target_distance_m - distance_traveled
                phase = (
                    "braking" if remaining_distance <= braking_distance else "cruising"
                )
                current_accel = -decel if phase == "braking" else 0

            # Calculate power requirements
            drag, rolling = self._calculate_resistive_forces(current_velocity)

            mechanical_power = (
                self.vehicle.mass_kg * current_accel + drag + rolling
            ) * current_velocity

            if phase == "accelerating":
                effective_power = mechanical_power / self.vehicle.motor_efficiency
            elif phase == "braking":
                effective_power = (
                    mechanical_power * self.vehicle.regenerative_efficiency
                )
            else:  # cruising
                effective_power = mechanical_power / self.vehicle.motor_efficiency

            # Account for auxiliary systems and accumulate energy
            total_power = effective_power + self.vehicle.auxiliary_power_watt
            energy_wh += total_power * SIMULATION_TIMESTEP / 3600  # Convert to Wh

            # Update kinematic state
            current_velocity = self._clamp_velocity(
                current_velocity + current_accel * SIMULATION_TIMESTEP
            )
            distance_traveled += current_velocity * SIMULATION_TIMESTEP

        return energy_wh / 1000  # Convert to kWh

    def _evaluate(
        self, population: np.ndarray, out: dict[str, Any], *args, **kwargs
    ) -> None:
        """Core evaluation method required by pymoo framework."""
        objectives = []
        constraints = []

        for solution in population:
            accel, decel = solution

            # Simulate vehicle motion with current control parameters
            time, max_speed, distance = self._simulate_vehicle_motion(accel, decel)

            # Calculate energy consumption
            energy = self._calculate_energy_consumption(accel, decel)

            # Calculate constraint violations (normalized)
            speed_violation = max(
                0, (max_speed - self.vehicle.max_speed_mps) / self.vehicle.max_speed_mps
            )
            energy_violation = max(
                0, (energy - self.available_energy_kwh) / self.available_energy_kwh
            )

            # Control input constraints
            accel_violation = max(
                (self.vehicle.min_acceleration - accel) / self.vehicle.min_acceleration
                if accel < self.vehicle.min_acceleration
                else 0,
                (accel - self.vehicle.max_acceleration) / self.vehicle.max_acceleration
                if accel > self.vehicle.max_acceleration
                else 0,
            )
            decel_violation = max(
                (self.vehicle.min_acceleration - decel) / self.vehicle.min_acceleration
                if decel < self.vehicle.min_acceleration
                else 0,
                (decel - self.vehicle.max_acceleration) / self.vehicle.max_acceleration
                if decel > self.vehicle.max_acceleration
                else 0,
            )
            control_violation = max(accel_violation, decel_violation)

            objectives.append([time, energy])
            constraints.append([speed_violation, energy_violation, control_violation])

        # Format outputs for optimization framework
        out["F"] = np.array(objectives)
        out["G"] = np.array(constraints)
