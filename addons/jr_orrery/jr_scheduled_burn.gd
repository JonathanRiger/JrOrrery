@tool
class_name JrScheduledBurn
extends Resource
## Resource defining a scheduled maneuver burn on a [JrBallisticOrbiter].
##
## Supports both multi-stage vehicles (referencing [member JrBallisticOrbiter.vehicle_stages]
## with automatic mid-burn stage splitting and dry-mass jettisoning) and standalone
## rocket-equation burns parameterized by [member isp], [member start_mass], [member end_mass]
## (or chained [member propellant_mass]), [member start_time], and [member duration].

## Coordinate frame in which [member direction] is interpreted (matching `JrBallisticOrbiter.ThrustFrame`).
enum ThrustFrame {
	## Unrotated inertial coordinate frame of the owning [JrOrbitDriver].
	INERTIAL = 0,
	## Body-local manual orientation frame of the [JrBallisticOrbiter].
	BODY_LOCAL = 1,
	## Instantaneous orbital RTN frame relative to the active dominant anchor:
	## `-Z` ([constant Vector3.FORWARD]) = Prograde,
	## `+Y` ([constant Vector3.UP]) = Orbit Normal,
	## `+X` ([constant Vector3.RIGHT]) = Radial Outward.
	ORBITAL = 2,
}

@export_group("Schedule & Direction")
## Whether this scheduled burn is active and included in trajectory integration.
@export var enabled: bool = true:
	set(value):
		if enabled != value:
			enabled = value;
			emit_changed();

## Optional human-readable identifier for this maneuver (e.g., `"Trans-Lunar Injection"`).
@export var burn_name: String = "":
	set(value):
		if burn_name != value:
			burn_name = value;
			emit_changed();

## Simulation timestamp (in the owning [JrOrbitDriver]'s `time_unit`) when this burn begins.
@export var start_time: float = 0.0:
	set(value):
		if not is_equal_approx(start_time, value):
			start_time = value;
			emit_changed();

## Duration of the burn in the owning [JrOrbitDriver]'s `time_unit` (`>= 0.0`).
## A duration of `0.0` executes an instantaneous impulsive rocket-equation burn at [member start_time].
@export_range(0.0, 1_000_000.0, 0.01, "or_greater")
var duration: float = 1.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(duration, clamped):
			duration = clamped;
			emit_changed();

## Reference frame in which [member direction] is evaluated during the burn.
@export var thrust_frame: int = ThrustFrame.ORBITAL:
	set(value):
		var clamped: int = clampi(value, ThrustFrame.INERTIAL, ThrustFrame.ORBITAL);
		if thrust_frame != clamped:
			thrust_frame = clamped;
			emit_changed();

## Burn direction vector in [member thrust_frame] (automatically normalized during integration;
## default `Vector3.FORWARD = (0, 0, -1)` is Prograde in [constant ThrustFrame.ORBITAL]).
@export var direction: Vector3 = Vector3.FORWARD:
	set(value):
		if not direction.is_equal_approx(value):
			direction = value;
			emit_changed();

@export_group("Stage & Propellant")
## Target stage index in [member JrBallisticOrbiter.vehicle_stages] (`-1` = use current active
## stage; `>= 0` = jettison any earlier stages `0 .. stage_index - 1` and fire `stage_index`).
@export var stage_index: int = -1:
	set(value):
		var clamped: int = maxi(-1, value);
		if stage_index != clamped:
			stage_index = clamped;
			emit_changed();

## When `true` and using [member JrBallisticOrbiter.vehicle_stages], consumes 100% of the active
## stage's remaining propellant across [member duration].
@export var burn_full_stage: bool = false:
	set(value):
		if burn_full_stage != value:
			burn_full_stage = value;
			emit_changed();

## Propellant mass consumed by this burn (in [member JrBallisticOrbiter.vehicle_mass_unit]).
## When using [member JrBallisticOrbiter.vehicle_stages], if this exceeds the active stage's
## remaining propellant, the burn automatically jettisons the depleted stage and continues
## burning on the next stage.
@export_range(0.0, 1_000_000_000.0, 0.1, "or_greater")
var propellant_mass: float = 200.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(propellant_mass, clamped):
			propellant_mass = clamped;
			emit_changed();

## When `true`, jettisons the active stage immediately upon completing this burn even if
## unburned propellant remains in that stage.
@export var jettison_after_burn: bool = false:
	set(value):
		if jettison_after_burn != value:
			jettison_after_burn = value;
			emit_changed();

@export_group("Standalone / Override Parameters")
## When `true` (or when [member JrBallisticOrbiter.vehicle_stages] is empty), uses this burn's
## own [member isp], [member start_mass], and [member end_mass] (or [member use_vehicle_mass_chain]).
@export var override_stage_parameters: bool = false:
	set(value):
		if override_stage_parameters != value:
			override_stage_parameters = value;
			emit_changed();

## Specific impulse (`I_sp`) in SI seconds (`s`, `> 0.0`) when [member JrBallisticOrbiter.vehicle_stages]
## is empty or [member override_stage_parameters] is `true`.
@export_range(1.0, 100_000.0, 1.0, "or_greater")
var isp: float = 300.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(isp, clamped):
			isp = clamped;
			emit_changed();

## When `true` in standalone mode (no stages), chains `start_mass` from the vehicle's remaining
## mass after prior burns and subtracts [member propellant_mass] down to `vehicle_dry_mass`.
## When `false`, uses explicit [member start_mass] and [member end_mass].
@export var use_vehicle_mass_chain: bool = false:
	set(value):
		if use_vehicle_mass_chain != value:
			use_vehicle_mass_chain = value;
			emit_changed();

## Explicit vehicle mass at the start of this burn (`m_0 > 0.0`) when not using stages or
## [member use_vehicle_mass_chain].
@export_range(0.001, 1_000_000_000.0, 0.1, "or_greater")
var start_mass: float = 1000.0:
	set(value):
		var clamped: float = maxf(1e-6, value);
		if not is_equal_approx(start_mass, clamped):
			start_mass = clamped;
			emit_changed();

## Explicit vehicle mass at the end of this burn (`0.0 < m_f <= start_mass`) when not using
## stages or [member use_vehicle_mass_chain].
@export_range(0.001, 1_000_000_000.0, 0.1, "or_greater")
var end_mass: float = 800.0:
	set(value):
		var clamped: float = maxf(1e-6, value);
		if not is_equal_approx(end_mass, clamped):
			end_mass = clamped;
			emit_changed();


## Returns the simulation timestamp (`start_time + duration`) at which this burn finishes.
func get_end_time() -> float:
	return start_time + maxf(0.0, duration);


## Returns the normalized 3D thrust direction vector (or `Vector3.ZERO` if degenerate).
func get_normalized_direction() -> Vector3:
	var len_sq: float = direction.length_squared();
	if len_sq <= 1e-16:
		return Vector3.ZERO;
	return direction / sqrt(len_sq);


## Returns the standalone effective exhaust velocity (`v_e = isp * g_0`) in SI `m/s`.
func get_exhaust_velocity_si() -> float:
	return maxf(0.0, isp) * JrOrreryUnits.STANDARD_GRAVITY_SI;


## Returns the standalone effective exhaust velocity (`v_e = isp * g_0`) in driver velocity
## units (`dist_unit / time_unit`).
func get_exhaust_velocity_driver(
		dist_unit: JrOrreryUnits.DistanceUnit,
		time_unit: JrOrreryUnits.TimeUnit
) -> float:
	return JrOrreryUnits.convert_isp_to_driver_exhaust_velocity(isp, dist_unit, time_unit);


## Computes the Tsiolkovsky delta-v (in SI `m/s`) for [param eff_start_mass] (`m_0`),
## [param eff_end_mass] (`m_f`), and [param eff_isp] (defaults to this burn's standalone properties).
func get_delta_v_si(
		eff_start_mass: float = -1.0,
		eff_end_mass: float = -1.0,
		eff_isp: float = -1.0
) -> float:
	var m0: float = eff_start_mass if eff_start_mass > 0.0 else start_mass;
	var mf: float = eff_end_mass if eff_end_mass > 0.0 else end_mass;
	var use_isp: float = eff_isp if eff_isp > 0.0 else isp;
	if mf <= 1e-12 or m0 <= mf or use_isp <= 0.0:
		return 0.0;
	return use_isp * JrOrreryUnits.STANDARD_GRAVITY_SI * log(m0 / mf);


## Computes the Tsiolkovsky delta-v in driver velocity units (`dist_unit / time_unit`) for
## [param mass_bounds] (`Vector2(m_0, m_f)`, or negative to use standalone [member start_mass] / [member end_mass]).
func get_delta_v_driver(
		dist_unit: JrOrreryUnits.DistanceUnit,
		time_unit: JrOrreryUnits.TimeUnit,
		mass_bounds: Vector2 = Vector2(-1.0, -1.0),
		eff_isp: float = -1.0
) -> float:
	var dv_si: float = get_delta_v_si(mass_bounds.x, mass_bounds.y, eff_isp);
	return dv_si * JrOrreryUnits.get_si_to_driver_velocity_factor(dist_unit, time_unit);


## Returns the constant thrust force magnitude in SI Newtons (`N`) across [member duration]
## for [param mass_bounds] (`Vector2(m_0, m_f)` in [param mass_unit]) and [param time_unit].
func get_thrust_force_si(
		mass_unit: JrOrreryUnits.MassUnit,
		time_unit: JrOrreryUnits.TimeUnit,
		mass_bounds: Vector2 = Vector2(-1.0, -1.0),
		eff_isp: float = -1.0
) -> float:
	var dur_s: float = maxf(0.0, duration) * JrOrreryUnits.get_time_scale_seconds(time_unit);
	if dur_s <= 1e-12:
		return 0.0;
	var m0: float = mass_bounds.x if mass_bounds.x > 0.0 else start_mass;
	var mf: float = mass_bounds.y if mass_bounds.y > 0.0 else end_mass;
	var use_isp: float = eff_isp if eff_isp > 0.0 else isp;
	if m0 <= mf or use_isp <= 0.0:
		return 0.0;
	var dm_kg: float = (m0 - mf) * JrOrreryUnits.get_mass_scale_kilograms(mass_unit);
	var ve_si: float = use_isp * JrOrreryUnits.STANDARD_GRAVITY_SI;
	return (dm_kg / dur_s) * ve_si;


## Samples the linear vehicle mass `m(t)` at [param eval_time] for a single-segment burn with
## [param mass_bounds] (`Vector2(m_0, m_f)`).
func sample_mass_at(
		eval_time: float,
		mass_bounds: Vector2 = Vector2(-1.0, -1.0)
) -> float:
	var m0: float = mass_bounds.x if mass_bounds.x > 0.0 else start_mass;
	var mf: float = minf(m0, mass_bounds.y if mass_bounds.y > 0.0 else end_mass);
	if eval_time <= start_time:
		return m0;
	var end_t: float = start_time + maxf(0.0, duration);
	if eval_time >= end_t or duration <= 1e-12:
		return mf;
	var frac: float = clampf((eval_time - start_time) / duration, 0.0, 1.0);
	return m0 - (m0 - mf) * frac;
