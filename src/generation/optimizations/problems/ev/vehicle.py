from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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
