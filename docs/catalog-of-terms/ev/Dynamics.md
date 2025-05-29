# Electric Vehicle Control Optimization

## System Definition

We model an electric vehicle (EV) traveling a fixed distance $D$, where we control its acceleration profile to optimize:

1. **Travel Time ($T$)** - Minimize total trip duration
2. **Energy Consumption ($E$)** - Minimize total energy used

## State-Space Representation

| Variable | Symbol | Units   | Description                          |
|----------|--------|---------|--------------------------------------|
| $ x(t) $ | Position | m      | Distance traveled                    |
| $ v(t) $ | Velocity | m/s    | Instantaneous speed                  |
| $ E(t) $ | Energy   | kWh    | Cumulative energy consumption       |

### Control Parameterization

| Phase | Control Input | Symbol | Bounds (m/s²) | Description              |
|-------|---------------|--------|---------------|--------------------------|
| 1     | Acceleration  | $ u_1 $ | [0.3, 4.0]    | Constant acceleration    |
| 2     | Deceleration  | $ u_2 $ | [0.3, 4.0]    | Constant deceleration    |

## System Dynamics

The EV's motion is governed by:

\begin{cases}
\dot{x}(t) = v(t) & \text{(Position)} \\
\dot{v}(t) = \frac{u(t)}{m} - \frac{F_{drag}(v) + F_{roll}}{m} & \text{(Velocity)} \\
F_{drag} = \frac{1}{2}\rho C_d A v^2(t) & \text{(Drag force)} \\
F_{roll} = C_r m g & \text{(Rolling resistance)}
\end{cases}

### Initial Conditions

\begin{aligned}
x(0) = 0,\ v(0) = 0
\end{aligned}

### Terminal Condition

\begin{aligned}
x(T) = D \quad \text{(Must reach target distance)}
\end{aligned}

## Optimization Problem Formulation

### Objectives

\begin{aligned}
\min_{u(t)} \mathbf{J} = \begin{bmatrix} T \\ E \end{bmatrix}
\end{aligned}

1. **Travel Time ($T$)**:
\begin{aligned}
   T = \inf \left\{ t \, \big| \, x(t) = D \right\}
   \end{aligned}
   - $D$: Total travel distance  
   - $x(t)$: Position at time $t$

2. **Energy Consumption ($E$)**:
\begin{aligned}
   E(u) = \int_0^T P\big(u(t), v(t)\big) \, dt
   \end{aligned}
   \begin{aligned}
   P(u,v) = \frac{u(t) \cdot v(t)}{\eta} + P_{\text{aux}}
   \end{aligned}
   - $\eta$: Motor efficiency (0 < $\eta$ ≤ 1)  
   - $P_{\text{aux}}$: Constant auxiliary power  
   - $u(t)$: Control input (acceleration/deceleration)  
   - $v(t)$: Velocity

### Hard Constraints

1. **Control Inputs**:
   $
   u_1 \in [0.3, 4.0]\, \text{m/s}^2,\quad u_2 \in [0.3, 4.0]\, \text{m/s}^2
   $

2. **Speed Limit**:
   $
   v(t) \leq v_{max} = 30\, \text{m/s}\quad \forall t\in[0,T]
   $

3. **Battery Limit**:
   $
   E_{total} \leq 20\%, E_{batt} = 0.4\, \text{kWh}
   $

4. **Distance Completion**:
   $
   x(T) = D = 1000\, \text{m} \pm 0.1\%
   $

## Physical Parameters

| Parameter | Symbol | Value | Units | Description |
|-----------|--------|-------|-------|-------------|
| Vehicle mass | \( m \) | 1000 | kg | Total curb weight |
| Drag coefficient | \( C_d \) | 0.24 | - | Aerodynamic profile |
| Frontal area | \( A \) | 2.4 | m² | Cross-sectional area |
| Rolling resistance | \( C_r \) | 0.008 | - | Tire-road friction |
| Motor efficiency | \( \eta_{mtr} \) | 0.85 | - | Drivetrain efficiency |
| Regen efficiency | \( \eta_{regen} \) | 0.70 | - | Braking recovery |
| Air density | \( \rho \) | 1.225 | kg/m³ | Sea level conditions |
| Auxiliary power | \( P_{aux} \) | 300 | W | Electronics load |

## Implementation Notes

- **Discretization**: The continuous control $u(t)$ is parameterized as:
  - $u_1$: Constant acceleration in Phase 1
  - $u_2$: Constant deceleration in Phase 2
- **Numerical Integration**: Trapezoidal rule used for energy calculation
- **Constraints Handling**: Normalized constraints ensure balanced optimization
