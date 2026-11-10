# JR Orrery

**JR Orrery** is a Godot 4.7+ addon for simulating and visualizing hierarchical analytical Keplerian ("on-rails") orbits and GPU-accelerated N-body ballistic trajectories across both **2D** and **3D** scenes with dynamic reference-frame switching and real-scale astronomical precision.

Simulation math is cleanly decoupled from visual presentation: a pure `Node` hierarchy computes 3D orbital kinematics and GPU compute-shader physics, while `JrOrbiter2D` and `JrOrbiter3D` follower nodes project or apply those transforms to your sprites, meshes, and cameras.

---

## Features

### Decoupled Simulation & Dual 2D/3D Visual Followers
- **Call Down / Signal Up Architecture**: Simulation nodes (`JrOrbitDriver`, `JrRailOrbiter`, `JrBallisticOrbiter`) inherit from `JrOrreryElement` (`Node`), bubbling hierarchy changes upward and pushing time, physical units, and reference frames downward.
- **Simultaneous 2D & 3D Views**: Drive `JrOrbiter2D` (`Node2D`) and `JrOrbiter3D` (`Node3D`) followers from the same simulation nodes at the same time, with independent `position_scale` multipliers.
- **Flexible Scene Hierarchies (`TransformMode`)**:
  - `LOCAL`: Parent visual moons under visual planets—parent axial tilt and sidereal spin are automatically canceled so child orbits remain fixed in the system's reference plane.
  - `SYSTEM`: Keep all visual followers as flat siblings under a common root, applying full perspective-shifted transforms directly.
  - `AUTO`: Automatically selects `LOCAL` when parented to another `JrOrbiter2D`/`JrOrbiter3D`, or `SYSTEM` otherwise.
- **Editor Transform Guards**: Detects accidental manual drags on simulation-controlled visual nodes in the editor, warns in the Scene dock, and restores the calculated orbital transform.

### Analytical Keplerian Orbits (`JrRailOrbiter`)
- **All Conic Sections**: Supports circular ($e = 0$), elliptical ($0 < e < 1$), parabolic ($e = 1$), and hyperbolic ($e > 1$) trajectories using Taylor warm-started Newton-Raphson Kepler solvers and analytical Barker parabolic solvers.
- **Full 3D Orientation & Spin**: Configurable `inclination`, `longitude_of_the_ascending_node` (RAAN), `argument_of_periapsis`, `mean_anomaly_at_epoch`, `axial_tilt` (obliquity), `axial_tilt_longitude`, `rotational_motion` / `sidereal_period`, and `rotation_at_epoch`.
- **Gravity-Synced Motion**: Enable `sync_motion_to_gravity` to automatically derive `mean_motion` and `orbital_period` from parent and self mass via Kepler's Third Law ($\sqrt{(\mu_{\text{parent}} + \mu_{\text{self}}) / |a|^3}$).
- **Binary Star & Binary Body Coupling**: Link two sibling `JrRailOrbiter` nodes under a zero-mass barycenter via `binary_partner` to reciprocally lock their eccentricities, inclinations, ascending nodes, opposite arguments of periapsis ($+180^\circ$), mass-weighted periapses ($q_1 \mu_1 = q_2 \mu_2$), and shared binary orbital period. Children parented to a massless binary barycenter automatically inherit the pair's combined $\mu$ for circumbinary orbits.

### GPU-Accelerated Hybrid N-Body Ballistics (`JrBallisticOrbiter`)
- **Batched GLSL Compute Shaders (`JrBallisticComputeSolver`)**: Steps hundreds of ballistic bodies in parallel on a shared local `RenderingDevice` (`ballistic_gravity.glsl`), with an identical CPU solver fallback for headless or server environments.
- **Universal-Variable Kick-Drift-Kick & Battin $f(q)$ Tides**: Propagates motion in dominant-anchor-relative coordinates using Stumpff functions $C(z)$ and $S(z)$, integrating differential third-body tidal perturbations via Battin's $f(q)$ formulation without subtracting large heliocentric coordinates.
- **Lowest Common Ancestor (LCA) Precision**: Resolves inter-well and perspective offset vectors by walking the hierarchy only up to the Lowest Common Ancestor, preventing `float32` catastrophic cancellation even for meter-scale Low Earth Orbits at $1\text{ AU}$ ($1.496 \times 10^{11}\text{ m}$) from a star.
- **Tidal Culling & Tier-1 On-Rails Coasting**: Distant third-body tides below `tidal_cull_threshold` are culled without square roots. Unperturbed bodies with zero thrust automatically freeze onto analytical Tier-1 epoch conics (eliminating numerical energy drift over high time-warp) and wake up automatically when crossing an analytical tidal wake radius or firing thrusters.
- **Automatic Hill-Sphere Anchor Switching**: Transitions dominant gravity-well anchors automatically using Hill-sphere hysteresis (`0.98 * r_Hill` entry, `1.05 * r_Hill` exit) and emits `dominant_body_changed(previous_body, current_body)`, or locks to an explicit `pinned_anchor`.
- **Dual Initial-State Authoring**: Author initial conditions as Cartesian `STATE_VECTORS` (`initial_position`, `initial_velocity`) or `KEPLERIAN_ELEMENTS` (`initial_periapsis`, `initial_eccentricity`, etc.) with live bidirectional synchronization.
- **Continuous & Impulsive Propulsion**:
  - Continuous `external_acceleration` scaled by physical `AccelerationUnit` in `INERTIAL`, `BODY_LOCAL`, or instantaneous `ORBITAL` (Prograde `-Z`, Orbit Normal `+Y`, Radial Outward `+X`) frames.
  - Runtime delta-v helpers: `apply_impulse()`, `apply_local_impulse()`, and `apply_orbital_impulse()`.
- **Live GPU Trajectory Forecasting**: Dispatches `ballistic_forecast.glsl` to predict future paths across moving gravity wells, continuous orbital burns, and mid-forecast Sphere-of-Influence handoffs (`get_forecast_points()`, `get_forecast_points_2d()`, `get_forecast_reference_ids()`).

### Variable Perspectives & Reference Frames (`JrOrbitDriver`)
- **Dynamic Origin Shifting (`perspective_node`)**: Lock any planet, moon, submoon, or spacecraft at the visual system origin at runtime without altering the underlying physics.
- **Three Reference-Frame Rotation Modes (`PerspectiveRotationMode`)**:
  - `INERTIAL`: Fixes `perspective_node` at the origin while keeping coordinate axes aligned to the inertial stars.
  - `BODY_SPIN`: Locks both position and axial spin orientation of `perspective_node` (e.g., geostationary / surface-fixed views).
  - `SYNODIC_ORBITAL`: Co-rotates with `perspective_node`'s orbital radius vector around its parent focus so Lagrange points ($L_1$–$L_5$) remain stationary on screen.
- **Unified Reference Direction & Planar Lock**: Configure system-wide `reference_rotation`, `euler_order`, `positive_is_clockwise`, and `lock_to_ecliptic` (which constrains all orbits, ballistic states, thrust vectors, and tilts flat onto the ecliptic plane while still permitting prograde $0^\circ$ and retrograde $180^\circ$ motion).

### 64-Bit Physical Unit System (`JrOrreryUnits`)
Configure simulation scales using CODATA 2018 and IAU 2015 nominal gravitational constants (`GM`), converted in 64-bit floating-point precision on the CPU before GPU upload:
- **Distance Units**: `METERS`, `KILOMETERS`, `MEGAMETERS`, `EARTH_RADII`, `LUNAR_DISTANCES`, `ASTRONOMICAL_UNITS`.
- **Time Units**: `SECONDS`, `MINUTES`, `HOURS`, `DAYS`, `YEARS`.
- **Mass Units**: `DRIVER_GM` (raw $\mu$), `KILOGRAMS`, `METRIC_TONS`, `TERAGRAMS`, `LUNAR_MASSES`, `EARTH_MASSES`, `JUPITER_MASSES`, `SOLAR_MASSES`.
- **Acceleration Units**: `DRIVER_UNITS`, `METERS_PER_SECOND_SQUARED`, `STANDARD_GRAVITIES_G` ($g_0 = 9.80665\text{ m/s}^2$).

### Interactive 2D & 3D Editor Tooling
- **2D & 3D Viewport Conic Gizmos**: Select any `JrRailOrbiter`, Keplerian `JrBallisticOrbiter`, or its `JrOrbiter2D`/`JrOrbiter3D` follower (or direct visual child) to drag interactive handles directly in the viewport with full Undo/Redo support:
  - **2D & 3D**: Periapsis, Apoapsis (automatically swaps periapsis/apoapsis when dragged past the opposite apsis), Semi-Parameter (eccentricity), and Mean Anomaly at Epoch.
  - **3D Exclusive**: Ascending Node (RAAN), Inclination, Axial Tilt (north pole), Axial Tilt Longitude, and Starting Day Angle (`rotation_at_epoch`).
- **2D & 3D Ballistic Vector Gizmos**: In `STATE_VECTORS` mode, drag initial position, scaled velocity arrows, and 3D vertical elevation stems directly in the editor with live coordinate/speed readouts.
- **Animated Editor Preview Ghosts**: Preview orbital revolutions, unbound flybys, body spin, and forward-integrated N-body ballistic paths live in the 2D and 3D editor viewports (`editor_play_speed`) without dirtying saved node transforms.

---

## Architecture Overview

```text
JrOrreryElement (Abstract Node)
├── JrOrbitDriver          # Root simulation controller, units, time, perspective & GPU solver host
├── JrRailOrbiter          # Analytical Keplerian conic & binary-coupled orbiting body
└── JrBallisticOrbiter     # Hybrid GPU N-body, tidal & thrust-driven ballistic body

Visual Followers (subscribe to JrOrreryElement.transform_updated)
├── JrOrbiter2D (Node2D)   # Top-down X-Z projection, 2D editor orbit polyline & ghost
└── JrOrbiter3D (Node3D)   # Full 3D transform & basis, ImmediateMesh 3D orbit line & wireframe ghost
```

| Class | Base Type | Purpose |
| :--- | :--- | :--- |
| `JrOrbitDriver` | `JrOrreryElement` (`Node`) | Owns simulation `time`, `time_scale`, physical unit scales, `JrReferenceDirection`, `perspective_node`, and `JrBallisticComputeSolver`. |
| `JrRailOrbiter` | `JrOrreryElement` (`Node`) | Analytical 3D conic section solver (elliptical, circular, parabolic, hyperbolic), axial spin, gravity-synced period, and binary coupling. |
| `JrBallisticOrbiter` | `JrOrreryElement` (`Node`) | Physics-integrated spacecraft/asteroid supporting state-vector or Keplerian initialization, continuous/impulsive thrust, Hill-sphere handoffs, and GPU forecasts. |
| `JrOrbiter2D` | `Node2D` | Projects a `JrOrreryElement`'s 3D motion onto 2D (`Vector2(x, z) * position_scale`) with optional yaw sync. |
| `JrOrbiter3D` | `Node3D` | Applies a `JrOrreryElement`'s 3D translation (`* position_scale`) and orthonormal rotational `Basis`. |
| `JrOrreryUnits` | `RefCounted` | Static physical unit enums and 64-bit SI / IAU `GM` conversion utilities. |
| `JrReferenceDirection` | `RefCounted` | Unified 3D ecliptic plane, vernal point, handedness, and planar-lock helper. |
| `JrBallisticComputeSolver` | `RefCounted` | Manages shared `RenderingDevice` pipelines (`ballistic_gravity.glsl` & `ballistic_forecast.glsl`) and CPU fallback. |

---

## Installation

1. Copy the `addons/jr_orrery` folder into your Godot 4.7+ project's `addons/` directory.
2. Open **Project > Project Settings > Plugins** and enable **JR Orrery**.

---

## Quick Start

### 1. Build a Simulation Hierarchy
Add a `JrOrbitDriver` node to your scene, then add `JrRailOrbiter` and `JrBallisticOrbiter` children beneath it to represent stars, planets, moons, and spacecraft:

```text
MainScene
├── JrOrbitDriver (mass = 1.0, mass_unit = SOLAR_MASSES, distance_unit = ASTRONOMICAL_UNITS, time_unit = DAYS)
│   ├── Earth (JrRailOrbiter: periapsis = 1.0, sync_motion_to_gravity = true, mass = 1.0, mass_unit = EARTH_MASSES)
│   │   ├── Moon (JrRailOrbiter: periapsis = 0.00257, sync_motion_to_gravity = true)
│   │   └── Probe (JrBallisticOrbiter: initial_state_mode = KEPLERIAN_ELEMENTS)
└── View3D (Node3D)
    ├── SunFollower (JrOrbiter3D: orbiter = JrOrbitDriver, position_scale = 20.0)
    ├── EarthFollower (JrOrbiter3D: orbiter = Earth, position_scale = 20.0)
    ├── MoonFollower (JrOrbiter3D: orbiter = Moon, position_scale = 20.0)
    └── ProbeFollower (JrOrbiter3D: orbiter = Probe, position_scale = 20.0)
```

> [!TIP]
> Attaching a `JrOrbiter2D` or `JrOrbiter3D` follower to the root `JrOrbitDriver` allows your central star visual to shift and rotate automatically whenever `JrOrbitDriver.perspective_node` is set to an orbiting planet or spacecraft!

### 2. Apply Spacecraft Maneuvers & Switch Perspectives in GDScript

```gdscript
extends Node

@export var driver: JrOrbitDriver;
@export var earth: JrRailOrbiter;
@export var probe: JrBallisticOrbiter;

func perform_prograde_burn() -> void:
	# Apply an instantaneous 0.5 driver-unit/time-unit prograde delta-v (-Z in orbital RTN frame):
	probe.apply_orbital_impulse(Vector3.FORWARD * 0.5);

func enable_continuous_ion_thrust() -> void:
	# Apply continuous 0.01 m/s^2 prograde thrust:
	probe.thrust_frame = JrBallisticOrbiter.ThrustFrame.ORBITAL;
	probe.acceleration_unit = JrOrreryUnits.AccelerationUnit.METERS_PER_SECOND_SQUARED;
	probe.external_acceleration = Vector3.FORWARD * 0.01;

func focus_on_earth_synodic() -> void:
	# Lock Earth at the system origin and co-rotate so Sun-Earth Lagrange points stay fixed:
	driver.perspective_node = earth;
	driver.perspective_rotation_mode = JrOrbitDriver.PerspectiveRotationMode.SYNODIC_ORBITAL;
```

### 3. Batch Spawning & Flattening Ballistic Swarms
When spawning large numbers of `JrBallisticOrbiter` bodies at runtime (such as an asteroid belt or debris cloud), wrap creation in `begin_batch_updates()` / `end_batch_updates()` and call `collect_ballistic_nodes()` to flatten them directly under the driver while preserving their initial parent-relative orbital states:

```gdscript
driver.begin_batch_updates();
for i: int in range(256):
	var asteroid: JrBallisticOrbiter = JrBallisticOrbiter.new();
	asteroid.initial_position = Vector3(0.0, 0.0, -20.0 - float(i) * 0.05);
	asteroid.initial_velocity = Vector3(2.8, 0.0, 0.0);
	asteroid.runtime_forecast_enabled = false;
	driver.add_child(asteroid);
driver.collect_ballistic_nodes();
driver.end_batch_updates();
```

---

## Viewport Gizmo Reference

When simulation playback is stopped in the editor, selecting an orbiter or its visual follower exposes interactive handles:

| Handle | Color | Viewport | Behavior & Modifiers |
| :--- | :--- | :--- | :--- |
| **Periapsis** | Orange | 2D & 3D | Adjusts `periapsis` and `argument_of_periapsis`. Hold `Ctrl` to lock `eccentricity`; hold `Alt` to lock `argument_of_periapsis`. |
| **Apoapsis** | Cyan | 2D & 3D | Shown on bound orbits ($e < 1$). Adjusts `apoapsis` and `argument_of_periapsis` (auto-swaps with Periapsis if dragged closer than periapsis). |
| **Semi-Parameter** | Green | 2D & 3D | Adjusts `eccentricity` via the semi-latus rectum without altering `periapsis`. |
| **Mean Anomaly at Epoch** | Magenta | 2D & 3D | Adjusts `mean_anomaly_at_epoch` along the conic path. |
| **Ascending Node (RAAN)** | Yellow | 3D | Rotates `longitude_of_the_ascending_node` around the ecliptic pole (hidden when `lock_to_ecliptic` is enabled). |
| **Inclination** | Teal | 3D | Tilts `inclination` across $(-180^\circ, 180^\circ]$ around the line of nodes. |
| **Axial Tilt & Longitude** | Red / Violet | 3D | Tilts the body's north spin pole (`axial_tilt`) and azimuthal tilt direction (`axial_tilt_longitude`). |
| **Rotation at Epoch** | Lime | 3D | Rotates the prime meridian (`rotation_at_epoch`) around the body's equatorial ring. |
| **Ballistic Position & Velocity** | Orange / Mint / Blue | 2D & 3D | Shown when `JrBallisticOrbiter` is in `STATE_VECTORS` mode; includes vertical elevation stems in 3D. |

---

## Included Demo & Showcase Scenarios

Open and run `res://addons/jr_orrery/demo/demo.tscn` to explore five interactive scenarios driven simultaneously in 2D and 3D from a single shared `JrOrbitDriver`:

1. **Planetary System & Submoons** (`demo_planetary_system.tscn`) — Hierarchical Keplerian system with planets, moons, a submoon, and GPU ballistic probes featuring live dashed trajectory forecasts and historical wake trails.
2. **Real-Scale LEO (1 AU) & Low-Thrust Spiral** (`demo_real_scale_leo.tscn`) — Real-scale Sun–Earth–Moon system in Kilometers & Seconds (`1 AU = 149.6M km`), demonstrating drift-free 400 km Low Earth Orbit precision alongside an `ORBITAL`-frame prograde ion-thrust spiral at $600\times$ time warp.
3. **Planet–Moon Lagrange Points (Stable L4 & L5)** (`demo_lagrange_points.tscn`) — Co-rotating Planet–Moon synodic frame (`PerspectiveRotationMode.SYNODIC_ORBITAL`) showcasing stable $L_4$ ($+60^\circ$ leading) and $L_5$ ($-60^\circ$ trailing) Trojan tadpole orbits.
4. **Binary Stars & Circumbinary Planet** (`demo_binary_stars.tscn`) — Massless barycenter (`mass = 0`) with reciprocally coupled binary stars (`binary_partner`), a circumbinary planet, and a ballistic probe navigating the dual gravity wells.
5. **GPU Asteroid Swarm (256 Ballistic Bodies)** (`demo_asteroid_swarm.tscn`) — Batched GLSL compute-shader N-body stepping of 256 `JrBallisticOrbiter` asteroids perturbed by a gas giant using `collect_ballistic_nodes()`.

### Demo Controls
- **Switch 2D / 3D View**: Click the HUD **Switch View** button or press `Tab` / `V`.
- **Switch Showcase Scenario**: Select from the HUD dropdown or press `1`–`5`.
- **Select Reference Perspective**: **Double-click** any planet, moon, probe, or the central star in 2D or 3D to lock the system origin to that body.
- **Cycle Perspective Rotation Mode**: Press `R` to cycle between `INERTIAL`, `BODY_SPIN`, and `SYNODIC_ORBITAL`.
- **Camera Navigation**: Mouse wheel to zoom; right-drag to orbit (3D) or pan (2D).
- **HUD Toggle**: Press `H` to collapse or expand the instructions panel.

---

## Author, Version & License

- **Author**: Jonathan Riger
- **Version**: `1.0`
- **Engine Compatibility**: Godot `4.7+` (Forward+ / Vulkan / D3D12 / Metal for GPU compute shaders; automatic CPU fallback on Compatibility / Headless)
- **License**: Released under the [MIT License](LICENSE) — free for personal and commercial use, modification, and distribution with attribution (preserving the copyright and permission notice).
