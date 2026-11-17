@tool
class_name JrVehicleStage
extends Resource
## Resource defining a single rocket stage within a [JrBallisticOrbiter] vehicle stack.
##
## Each stage specifies its structural [member dry_mass], initial usable [member propellant_mass],
## and engine specific impulse [member isp] (in SI seconds). When attached in
## [member JrBallisticOrbiter.vehicle_stages], lower stages (`index 0`, `1`, ...) push the
## combined wet and dry mass of all upper stages above them until depleted or jettisoned.

## Optional human-readable label for this stage (e.g., `"Stage 1 - Booster"`).
@export var stage_name: String = "":
	set(value):
		if stage_name != value:
			stage_name = value;
			emit_changed();

## Structural, engine, and tank dry mass of this stage alone (`>= 0.0`), expressed in
## [member JrBallisticOrbiter.vehicle_mass_unit].
@export_range(0.0, 1_000_000_000.0, 0.1, "or_greater")
var dry_mass: float = 200.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(dry_mass, clamped):
			dry_mass = clamped;
			emit_changed();

## Initial usable propellant mass carried in this stage (`>= 0.0`), expressed in
## [member JrBallisticOrbiter.vehicle_mass_unit].
@export_range(0.0, 1_000_000_000.0, 0.1, "or_greater")
var propellant_mass: float = 800.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(propellant_mass, clamped):
			propellant_mass = clamped;
			emit_changed();

## Specific impulse (`I_sp`) of this stage's propulsion system in SI seconds (`s`, `> 0.0`).
@export_range(1.0, 100_000.0, 1.0, "or_greater")
var isp: float = 300.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(isp, clamped):
			isp = clamped;
			emit_changed();


## Returns the total initial wet mass (`dry_mass + propellant_mass`) of this stage alone.
func get_wet_mass() -> float:
	return dry_mass + propellant_mass;


## Returns the effective exhaust velocity (`v_e = isp * g_0`) in SI meters per second (`m/s`).
func get_exhaust_velocity_si() -> float:
	return maxf(0.0, isp) * JrOrreryUnits.STANDARD_GRAVITY_SI;


## Returns the effective exhaust velocity (`v_e = isp * g_0`) converted into driver velocity
## units (`dist_unit / time_unit`).
func get_exhaust_velocity_driver(
		dist_unit: JrOrreryUnits.DistanceUnit,
		time_unit: JrOrreryUnits.TimeUnit
) -> float:
	return JrOrreryUnits.convert_isp_to_driver_exhaust_velocity(isp, dist_unit, time_unit);


## Returns the ideal Tsiolkovsky delta-v (in SI `m/s`) produced by burning all
## [member propellant_mass] in this stage while carrying [param upper_stack_mass] above it.
func get_stage_delta_v_si(upper_stack_mass: float = 0.0) -> float:
	var m_end: float = maxf(0.0, dry_mass + maxf(0.0, upper_stack_mass));
	var m_start: float = m_end + maxf(0.0, propellant_mass);
	if m_end <= 1e-12 or m_start <= m_end:
		return 0.0;
	return get_exhaust_velocity_si() * log(m_start / m_end);
