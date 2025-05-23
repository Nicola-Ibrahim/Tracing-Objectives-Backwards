import numpy as np

from .base import EVProblem


class EVControlProblem(EVProblem):
    """
    Multi-objective optimization problem for electric vehicle motion control.
    Optimizes acceleration/deceleration strategies to balance travel time and energy consumption.

    Objectives:
    1. Minimize total travel time (minutes)
    2. Minimize energy consumption (kWh)

    Constraints:
    1. Vehicle speed must not exceed allowed maximum
    2. Total energy used must not exceed battery capacity
    3. Control inputs must stay within acceleration/deceleration limits
    """

    def __init__(self, target_distance_km=10.0, initial_state_of_charge=0.9) -> None:
        """Initialize vehicle parameters and problem configuration"""

        # ========================
        # Vehicle Physical Parameters
        # ========================
        self.vehicle_mass_kg = 1000  # Total mass of the vehicle
        self.air_density_kg_per_m3 = 1.225  # Air density at standard conditions
        self.drag_coefficient = 0.24  # Aerodynamic shape efficiency
        self.frontal_area_m2 = 2.4  # Cross-sectional area facing airflow
        self.rolling_resistance_coeff = 0.008  # Tire-road friction coefficient
        self.motor_efficiency = 0.85  # Power conversion efficiency (motor)
        self.regenerative_efficiency = 0.70  # Energy recovery efficiency (braking)
        self.auxiliary_power_watt = 300  # Constant power for non-drive systems

        # ========================
        # Operational Constraints
        # ========================
        self.max_speed_mps = 20  # 72 km/h (20 m/s) speed limit
        self.battery_capacity_kwh = 10.0  # Total battery capacity
        self.initial_state_of_charge = (
            initial_state_of_charge  # Starting charge level (90%)
        )
        self.minimum_state_of_charge = 0.2  # Safety threshold for battery drain

        # Energy availability calculation
        self.available_energy_kwh = (
            self.initial_state_of_charge - self.minimum_state_of_charge
        ) * self.battery_capacity_kwh

        # ========================
        # Mission Parameters
        # ========================
        self.target_distance_km = target_distance_km  # Trip distance requirement
        self.target_distance_m = self.target_distance_km * 1000  # Convert to meters

        # ========================
        # Control Input Limits
        # ========================
        self.min_acceleration = 0.2  # m/s² (comfort-focused minimum)
        self.max_acceleration = 2.0  # m/s² (performance-focused maximum)

        # Initialize optimization problem structure
        super().__init__(
            n_var=2,  # Decision variables: [acceleration_rate, deceleration_rate]
            n_obj=2,  # Objectives: [time_minutes, energy_kwh]
            n_constr=3,  # Constraints: [speed_limit, energy_limit, control_bounds]
            xl=np.array([self.min_acceleration, self.min_acceleration]),
            xu=np.array([self.max_acceleration, self.max_acceleration]),
        )

    def _clamp_velocity(self, velocity: float) -> float:
        """Ensure velocity stays within physical and regulatory limits"""
        return np.clip(velocity, 0, self.max_speed_mps)

    def _calculate_resistive_forces(self, velocity: float) -> tuple[float, float]:
        """
        Compute opposing forces affecting vehicle motion

        Args:
            velocity: Current vehicle speed in m/s

        Returns:
            tuple: (aerodynamic_drag_force [N], rolling_resistance_force [N])
        """
        aerodynamic_drag = (
            0.5
            * self.air_density_kg_per_m3
            * self.drag_coefficient
            * self.frontal_area_m2
            * velocity**2
        )
        rolling_resistance = self.rolling_resistance_coeff * self.vehicle_mass_kg * 9.81

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
            current_velocity < self.max_speed_mps
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
            cruise_duration = remaining_distance / self.max_speed_mps
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
            if current_velocity < self.max_speed_mps:
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
                    self.vehicle_mass_kg * current_acceleration
                    + drag_force
                    + rolling_force
                ) * current_velocity
                effective_power = mechanical_power / self.motor_efficiency
            elif operation_phase == "braking":
                mechanical_power = (
                    self.vehicle_mass_kg * current_acceleration
                    + drag_force
                    + rolling_force
                ) * current_velocity
                effective_power = mechanical_power * self.regenerative_efficiency
            else:  # Cruising
                mechanical_power = (drag_force + rolling_force) * current_velocity
                effective_power = mechanical_power / self.motor_efficiency

            # Sum total energy consumption with auxiliary systems
            total_power = effective_power + self.auxiliary_power_watt
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
                0, (max_speed - self.max_speed_mps) / self.max_speed_mps
            )
            energy_violation = max(
                0, (energy_used - self.available_energy_kwh) / self.available_energy_kwh
            )

            # Control input constraint checks
            accel_violation = max(
                (self.min_acceleration - accel_rate) / self.min_acceleration
                if accel_rate < self.min_acceleration
                else 0,
                (accel_rate - self.max_acceleration) / self.max_acceleration
                if accel_rate > self.max_acceleration
                else 0,
            )
            decel_violation = max(
                (self.min_acceleration - decel_rate) / self.min_acceleration
                if decel_rate < self.min_acceleration
                else 0,
                (decel_rate - self.max_acceleration) / self.max_acceleration
                if decel_rate > self.max_acceleration
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
