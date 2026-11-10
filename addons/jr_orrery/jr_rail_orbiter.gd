@tool
@icon("res://addons/jr_orrery/icons/orbiter.svg")
class_name JrRailOrbiter
extends JrOrreryElement
## Simulation [Node] representing an analytical Keplerian ("on-rails") orbiting body
## within a [JrOrbitDriver] hierarchy.
##
## Directly defines the 3D Keplerian orbital elements, axial tilt, sidereal
## rotation, physical mass, gravity-synced mean motion, and optional binary-star
## partner coupling. Solves Kepler's and Barker's equations each tick and computes
## both parent-relative and system/perspective-adjusted 3D transforms. Visual follower
## nodes ([JrOrbiter2D] and [JrOrbiter3D]) subscribe to [signal JrOrreryElement.transform_updated]
## to synchronize their scene transforms.

## Maximum number of Newton-Raphson iterations used when solving Kepler's equation.
const MAX_KEPLER_ITERATIONS: int = 32

## Maximum unwrapped change in normalized mean anomaly (in radians) for which
## the cached eccentric anomaly is used to warm-start Newton-Raphson iteration.
const WARM_START_MAX_DELTA_M: float = PI * 0.5

## Default duration (in time units) past the mean anomaly at epoch or periapsis
## (whichever is later) before an unbound editor preview loop resets.
const UNBOUND_PREVIEW_TAIL_DURATION: float = 30.0

## Ratio of [member periapsis] used for the ascending node and inclination gizmo radius.
const ORBITAL_PLANE_GIZMO_RADIUS_RATIO: float = 0.65

## Multiplier applied to the body gizmo radius for the axial tilt north-pole grabber handle.
const AXIAL_TILT_GIZMO_RADIUS_SCALE: float = 1.25

## Multiplier applied to the body gizmo radius for the equatorial starting day angle grabber
## handle.
const STARTING_DAY_GIZMO_RADIUS_SCALE: float = 0.85

## Shape of the conic section orbit.
## - [code]0.0[/code]: Circular orbit.
## - [code](0.0, 1.0)[/code]: Elliptical (bound) orbit.
## - [code]1.0[/code]: Parabolic escape trajectory.
## - [code]> 1.0[/code]: Hyperbolic escape trajectory.
@export_range(0.0, 2.0, 0.0001, "or_greater")
var eccentricity: float = 0.0:
	set(value):
		var clamped: float = maxf(value, 0.0)
		if not is_equal_approx(eccentricity, clamped):
			eccentricity = clamped
			_update_orbital_shape_cache()
			_invalidate_anomaly_cache()
			_refresh_gravity_dependent_motion()
			_notify_orbital_parameters_changed()

## Closest approach distance (periapsis radius) to the central body focus.
## Must be strictly positive.
@export_range(0.001, 1_000_000_000.0, 0.001, "exp", "or_greater")
var periapsis: float = 10.0:
	set(value):
		var clamped: float = maxf(1e-6, value)
		if not is_equal_approx(periapsis, clamped):
			periapsis = clamped
			_update_orbital_shape_cache()
			_invalidate_anomaly_cache()
			_refresh_gravity_dependent_motion()
			_notify_orbital_parameters_changed()

## Farthest distance (apoapsis radius) from the central body focus.
## Returns [constant @GDScript.INF] for unbound ([code]eccentricity >= 1.0[/code]) trajectories.
## Setting a finite value greater than or equal to [member periapsis] updates
## [member eccentricity].
@export var apoapsis: float = 10.0:
	get:
		if is_unbound_orbit():
			return INF
		return _semi_major_axis * (1.0 + eccentricity)
	set(value):
		if is_inf(value) or is_nan(value):
			return
		var clamped_apoapsis: float = maxf(value, periapsis)
		eccentricity = (clamped_apoapsis - periapsis) / (clamped_apoapsis + periapsis)

## Continuous orbital inclination in radians within [code](-PI, PI][/code],
## tilting the orbital plane around the line of ascending node.
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var inclination: float = 0.0:
	set(value):
		var normalized: float = normalize_angle(value)
		if is_locked_to_ecliptic():
			normalized = PI if absf(normalized) > PI * 0.5 else 0.0
		if not is_equal_approx(inclination, normalized):
			inclination = normalized
			_update_orbital_orientation_cache()
			_invalidate_anomaly_cache()
			_sync_with_binary_partner()
			_notify_orbital_parameters_changed()

## Angle in radians in the reference ecliptic plane from the vernal point to the
## ascending node, normalized to the range [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var longitude_of_the_ascending_node: float = 0.0:
	set(value):
		var normalized: float = normalize_angle(value)
		if not is_equal_approx(longitude_of_the_ascending_node, normalized):
			longitude_of_the_ascending_node = normalized
			_update_orbital_orientation_cache()
			_invalidate_anomaly_cache()
			_sync_with_binary_partner()
			_notify_orbital_parameters_changed()

## Angle in radians within the orbital plane from the ascending node to the
## periapsis, normalized to the range [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var argument_of_periapsis: float = 0.0:
	set(value):
		var normalized: float = normalize_angle(value)
		if not is_equal_approx(argument_of_periapsis, normalized):
			argument_of_periapsis = normalized
			_update_orbital_orientation_cache()
			_invalidate_anomaly_cache()
			_sync_with_binary_partner()
			_notify_orbital_parameters_changed()

## Mean orbital angular speed of the body in radians per unit time. Must be non-negative.
@export var mean_motion: float = 1.0:
	set(value):
		var clamped: float = maxf(0.0, value)
		if mean_motion != clamped:
			mean_motion = clamped
			_invalidate_anomaly_cache()
			_notify_orbital_parameters_changed()

## Time required to complete one full orbital revolution ([code]TAU / mean_motion[/code]).
## Returns [constant @GDScript.NAN] for unbound orbits or when [member mean_motion] is zero.
## Setting a positive finite period updates [member mean_motion].
@export var orbital_period: float:
	get:
		if is_unbound_orbit() or mean_motion <= 0.0:
			return NAN
		return TAU / mean_motion
	set(value):
		if value > 0.0 and not is_inf(value) and not is_nan(value):
			mean_motion = TAU / value

@export_group("Rotation & Axial Tilt")
## Obliquity (axial tilt) angle in radians measured from the body's orbital normal,
## normalized to [code](-PI, PI][/code]. Values with magnitude greater than 90 degrees
## result in retrograde axial rotation.
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var axial_tilt: float = 0.0:
	set(value):
		var normalized: float = 0.0 if is_locked_to_ecliptic() else normalize_angle(value)
		if not is_equal_approx(axial_tilt, normalized):
			axial_tilt = normalized
			_notify_orbital_parameters_changed()

## Azimuthal direction of [member axial_tilt] within the orbital plane (in radians),
## normalized to [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var axial_tilt_longitude: float = 0.0:
	set(value):
		var normalized: float = normalize_angle(value)
		if not is_equal_approx(axial_tilt_longitude, normalized):
			axial_tilt_longitude = normalized
			_notify_orbital_parameters_changed()

## Sidereal rotational angular speed around the body's spin axis in radians per
## unit time. Must be non-negative; set to [code]0.0[/code] for a non-rotating body.
@export var rotational_motion: float = 1.0:
	set(value):
		var clamped: float = maxf(0.0, value)
		if rotational_motion != clamped:
			rotational_motion = clamped
			_notify_orbital_parameters_changed()

## Time required for the body to complete one full sidereal rotation around its
## spin axis ([code]TAU / rotational_motion[/code]). Returns [constant @GDScript.NAN]
## when [member rotational_motion] is zero. Setting a positive finite period updates
## [member rotational_motion].
@export var sidereal_period: float:
	get:
		if rotational_motion <= 0.0:
			return NAN
		return TAU / rotational_motion
	set(value):
		if value > 0.0 and not is_inf(value) and not is_nan(value):
			rotational_motion = TAU / value

@export_group("Epoch")
## Mean anomaly at epoch ([code]time = 0.0[/code]) in radians. Wrapped to
## [code](-PI, PI][/code] for bound orbits and unwrapped for unbound trajectories.
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees", "or_less", "or_greater")
var mean_anomaly_at_epoch: float = 0.0:
	set(value):
		var target_val: float = normalize_angle(value) if is_bound_orbit() else value
		if not is_equal_approx(mean_anomaly_at_epoch, target_val):
			mean_anomaly_at_epoch = target_val
			_invalidate_anomaly_cache()
			_sync_with_binary_partner()
			_notify_orbital_parameters_changed()

## Initial sidereal spin angle around the body's spin axis at epoch ([code]time = 0.0[/code])
## in radians, normalized to [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var rotation_at_epoch: float = 0.0:
	set(value):
		var normalized: float = normalize_angle(value)
		if not is_equal_approx(rotation_at_epoch, normalized):
			rotation_at_epoch = normalized
			_notify_orbital_parameters_changed()

@export_group("Gravity & Binary Coupling")
## Physical mass of this body expressed in [member mass_unit]. Automatically scaled
## by the owning [JrOrbitDriver]'s length and time units to compute [member gravity_strength] (mu =
## G * M).
@export_range(0.0, 1_000_000_000.0, 0.001, "or_greater")
var mass: float = 0.0:
	set(value):
		var clamped: float = maxf(0.0, value)
		if mass != clamped:
			mass = clamped
			_mass = clamped
			_recompute_gravity_strength_from_mass()
			_refresh_gravity_dependent_motion()
			orbit_visuals_changed.emit()

## Physical mass unit used to interpret [member mass].
@export var mass_unit := JrOrreryUnits.MassUnit.DRIVER_GM:
	set(value):
		if mass_unit != value:
			mass_unit = value
			_mass_unit = value
			_recompute_gravity_strength_from_mass()
			_refresh_gravity_dependent_motion()
			orbit_visuals_changed.emit()

## If [code]true[/code], automatically derives [member mean_motion] and [member orbital_period]
## from Kepler's third law ([code]sqrt((mu_parent + mu_self) / |a|^3)[/code]) whenever
## orbital shape or mass changes.
@export var sync_motion_to_gravity: bool = false:
	set(value):
		if sync_motion_to_gravity != value:
			sync_motion_to_gravity = value
			if sync_motion_to_gravity:
				_refresh_gravity_dependent_motion()
				_notify_orbital_parameters_changed()

## Optional sibling [JrRailOrbiter] sharing a common zero-mass barycenter parent.
## When assigned, reciprocally locks both bodies' eccentricities, inclinations,
## ascending nodes, epoch anomalies, opposite arguments of periapsis (+180 deg),
## mass-weighted periapses ([code]q_1 * mu_1 = q_2 * mu_2[/code]), and shared binary orbital
## period.
@export var binary_partner: JrRailOrbiter = null:
	set(value):
		var valid_partner: JrRailOrbiter = null if value == self else value
		if binary_partner == valid_partner:
			return
		var old_partner: JrRailOrbiter = binary_partner
		binary_partner = valid_partner
		if not _syncing_binary_partner:
			_syncing_binary_partner = true
			if (
					old_partner != null
					and is_instance_valid(old_partner)
					and old_partner.binary_partner == self
			):
				old_partner.binary_partner = null
			if binary_partner != null and is_instance_valid(binary_partner):
				binary_partner.binary_partner = self
			_syncing_binary_partner = false
			_sync_with_binary_partner()
			_notify_orbital_parameters_changed()

## Half of the major axis of the orbit. Positive for elliptical/circular orbits
## ([code]eccentricity < 1.0[/code]), [constant @GDScript.INF] for parabolic trajectories
## ([code]eccentricity == 1.0[/code]), and negative for hyperbolic trajectories
## ([code]eccentricity > 1.0[/code]).
var semi_major_axis: float:
	get:
		return _semi_major_axis

## Half of the minor axis of the orbit. Returns [constant @GDScript.INF] for
## parabolic trajectories.
var semi_minor_axis: float:
	get:
		return _semi_minor_axis

## The semi-latus rectum (semi-parameter) of the orbit, measuring the chord
## distance perpendicular to the major axis through the focus. Defined for all
## conic sections as [code]periapsis * (1.0 + eccentricity)[/code].
var semi_parameter: float:
	get:
		return _semi_parameter

## Indicator returning [code]true[/code] when [member inclination] exceeds
## 90 degrees in magnitude ([code]absf(inclination) > PI * 0.5[/code]),
## corresponding to retrograde motion relative to the reference plane pole.
var is_orbit_retrograde: bool:
	get:
		return absf(inclination) > PI * 0.5
	set(value):
		if value and not is_orbit_retrograde:
			inclination = PI if is_zero_approx(inclination) else normalize_angle(PI - inclination)
		elif not value and is_orbit_retrograde:
			inclination = (
					0.0
					if is_equal_approx(absf(inclination), PI)
					else normalize_angle(PI - inclination)
			)

## Read-only dogleg longitude of periapsis in radians, normalized to [code](-PI, PI][/code].
var longitude_of_periapsis: float:
	get:
		if is_orbit_retrograde:
			return normalize_angle(longitude_of_the_ascending_node - argument_of_periapsis)
		return normalize_angle(longitude_of_the_ascending_node + argument_of_periapsis)

## Standard gravitational parameter (mu = G * M) in driver units (dist_unit^3 / time_unit^2).
## Maintained for backwards compatibility; setting this updates [member mass] in
## [member mass_unit].
var gravity_strength: float:
	get:
		return _gravity_strength
	set(value):
		var clamped: float = maxf(0.0, value)
		_gravity_strength = clamped
		_mass = JrOrreryUnits.mass_from_mu_in_driver_units(
				clamped,
				mass_unit,
				_driver_distance_unit,
				_driver_time_unit
		)
		mass = _mass
		_refresh_gravity_dependent_motion()
		orbit_visuals_changed.emit()

## Cached semi-major axis, recalculated when [member periapsis] or [member eccentricity] changes.
var _semi_major_axis: float = 10.0

## Cached semi-minor axis, recalculated when [member periapsis] or [member eccentricity] changes.
var _semi_minor_axis: float = 10.0

## Cached semi-latus rectum (semi-parameter), recalculated when [member periapsis] or
## [member eccentricity] changes.
var _semi_parameter: float = 10.0

## Cached 3D basis transform mapping canonical perifocal coordinates ([code]-Z[/code] periapsis,
## [code]+X[/code] +90-degree true anomaly, [code]+Y[/code] orbital normal) into canonical
## 3D ecliptic space for clockwise progression.
var _orbital_transform := Transform3D.IDENTITY

## Cached right-handed [Basis] for counter-clockwise progression when
## [member JrReferenceDirection.positive_is_clockwise] is [code]false[/code].
var _orbital_basis_ccw := Basis.IDENTITY

## Timestamp of the most recently cached eccentric anomaly solution, or
## [constant @GDScript.NAN] if the cache is invalid.
var _cached_time: float = NAN

## Normalized mean anomaly (in radians) from the most recently cached solution,
## or [constant @GDScript.NAN] if the cache is invalid.
var _cached_mean_anomaly: float = NAN

## Eccentric anomaly solution (in radians) from the most recently cached run,
## or [constant @GDScript.NAN] if the cache is invalid.
var _cached_eccentric_anomaly: float = NAN

## Cached reciprocal derivative [code]1.0 / (1.0 - e * cos(E))[/code] at
## [member _cached_eccentric_anomaly], used for first-order warm-start extrapolation.
var _cached_inv_dm_de: float = 1.0

## Last simulation timestamp passed to [method evaluate_orbital_state].
var _last_time: float = 0.0

## Last editor preview timestamp passed to [method update_editor_ghost].
var _last_preview_time: float = 0.0

## Relative position of the animated editor preview ghost with respect to the parent focus.
var _ghost_relative_position := Vector3.ZERO

## Orientation [Basis] of the animated editor preview ghost at the preview timestamp.
var _ghost_body_basis := Basis.IDENTITY

## Whether [member _ghost_relative_position] currently holds a valid preview position.
var _has_ghost_position: bool = false

## Effective gravitational parameter (mu) of the immediate parent element supplied
## downward by [JrOrbitDriver].
var _parent_effective_mu: float = 0.0

## Re-entrancy guard preventing infinite recursion when synchronizing [member binary_partner].
var _syncing_binary_partner: bool = false

## Re-entrancy guard preventing recursion when deriving [member mean_motion] from gravity.
var _updating_synced_motion: bool = false


## Computes the analytical Keplerian mean motion [code]n[/code] (in rad / time_unit)
## for a conic section with [param pe], [param ecc], and effective gravitational parameter
## [param mu_eff].
static func compute_keplerian_mean_motion(pe: float, ecc: float, mu_eff: float) -> float:
	if mu_eff <= 0.0 or pe <= 0.0 or ecc < 0.0:
		return 0.0
	if is_equal_approx(ecc, 1.0):
		return sqrt(mu_eff / (2.0 * pe * pe * pe))
	var abs_a: float = absf(pe / (1.0 - ecc))
	if abs_a <= 0.0 or is_inf(abs_a):
		return 0.0
	return sqrt(mu_eff / (abs_a * abs_a * abs_a))


## Converts 3D Cartesian state vectors [param pos_canonical] and [param vel_canonical]
## (in the canonical reference frame prior to [member JrReferenceDirection.rotation_basis])
## into Keplerian elements under gravitational parameter [param mu] and progression
## flag [param positive_is_clockwise].
static func compute_conic_from_state_vectors(
		pos_canonical: Vector3,
		vel_canonical: Vector3,
		mu: float,
		positive_is_clockwise: bool = true
) -> Dictionary[StringName, float]:
	var r_cw: Vector3 = (
			pos_canonical
			if positive_is_clockwise
			else Vector3(-pos_canonical.x, pos_canonical.y, pos_canonical.z)
	)
	var v_cw: Vector3 = (
			vel_canonical
			if positive_is_clockwise
			else Vector3(-vel_canonical.x, vel_canonical.y, vel_canonical.z)
	)
	var r_len: float = r_cw.length()
	var safe_mu: float = maxf(1e-12, mu)

	if r_len <= 1e-8:
		return _build_default_conic_dict(10.0, safe_mu)

	var h_vec: Vector3 = v_cw.cross(r_cw)
	var h_len: float = h_vec.length()
	var orb_y: Vector3 = (h_vec / h_len) if h_len > 1e-8 else Vector3.UP

	var v_sq: float = v_cw.length_squared()
	var r_dot_v: float = r_cw.dot(v_cw)
	var e_vec: Vector3 = ((v_sq - safe_mu / r_len) * r_cw - r_dot_v * v_cw) / safe_mu
	var ecc: float = maxf(0.0, e_vec.length())

	var p_semi: float = (h_len * h_len) / safe_mu
	var pe_dist: float = maxf(1e-6, p_semi / (1.0 + ecc) if p_semi > 1e-9 else r_len)

	var inc: float = acos(clampf(orb_y.y, -1.0, 1.0))
	var sin_inc_sq: float = orb_y.x * orb_y.x + orb_y.z * orb_y.z
	var lan: float = atan2(-orb_y.z, -orb_y.x) if sin_inc_sq > 1e-10 else 0.0

	var cw_lan := Basis(Vector3.UP, -lan)
	var cw_inc := Basis(Vector3.FORWARD, -inc)
	var nodal_basis: Basis = cw_lan * cw_inc

	var pe_dir: Vector3 = -nodal_basis.z
	if ecc > 1e-5:
		var proj_e: Vector3 = e_vec - orb_y * e_vec.dot(orb_y)
		if proj_e.length_squared() > 1e-12:
			pe_dir = proj_e.normalized()

	var nodal_pe: Vector3 = nodal_basis.transposed() * pe_dir
	var arg_pe: float = normalize_angle(atan2(nodal_pe.x, -nodal_pe.z))

	var cw_arg := Basis(Vector3.UP, -arg_pe)
	var orb_basis: Basis = (nodal_basis * cw_arg).orthonormalized()
	var true_anom: float = atan2(r_cw.dot(orb_basis.x), -r_cw.dot(orb_basis.z))
	var mean_anom: float = _compute_mean_anomaly_static(true_anom, ecc)
	var derived_n: float = compute_keplerian_mean_motion(pe_dist, ecc, safe_mu)

	return {
		&"periapsis": pe_dist,
		&"eccentricity": ecc,
		&"inclination": normalize_angle(inc),
		&"longitude_of_the_ascending_node": normalize_angle(lan),
		&"argument_of_periapsis": arg_pe,
		&"mean_anomaly_at_epoch": mean_anom,
		&"mean_motion": derived_n,
	}


## Normalizes an [param angle] in radians to the half-open interval [code](-PI, PI][/code].
static func normalize_angle(angle: float) -> float:
	return PI - fposmod(PI - angle, TAU)


## Builds a fallback circular conic dictionary at radius [param radius] and gravity [param mu].
static func _build_default_conic_dict(radius: float, mu: float) -> Dictionary[StringName, float]:
	var pe: float = maxf(1e-6, radius)
	return {
		&"periapsis": pe,
		&"eccentricity": 0.0,
		&"inclination": 0.0,
		&"longitude_of_the_ascending_node": 0.0,
		&"argument_of_periapsis": 0.0,
		&"mean_anomaly_at_epoch": 0.0,
		&"mean_motion": compute_keplerian_mean_motion(pe, 0.0, mu),
	}


## Converts [param true_anomaly] and [param ecc] into mean anomaly for any conic section.
static func _compute_mean_anomaly_static(true_anomaly: float, ecc: float) -> float:
	if is_equal_approx(ecc, 1.0):
		var d: float = tan(clampf(true_anomaly, -PI + 0.01, PI - 0.01) * 0.5)
		return d + (d * d * d) / 3.0
	if ecc > 1.0:
		var max_nu: float = acos(clampf(-1.0 / ecc, -1.0, 1.0)) - 0.01
		var clamped_nu: float = clampf(true_anomaly, -max_nu, max_nu)
		var tanh_half_h: float = clampf(
				sqrt((ecc - 1.0) / (ecc + 1.0)) * tan(clamped_nu * 0.5),
				-0.9999,
				0.9999
		)
		var h: float = log((1.0 + tanh_half_h) / (1.0 - tanh_half_h))
		return ecc * sinh(h) - h
	var half_nu: float = true_anomaly * 0.5
	var e_anom: float = 2.0 * atan2(
			sqrt(maxf(0.0, 1.0 - ecc)) * sin(half_nu),
			sqrt(1.0 + ecc) * cos(half_nu)
	)
	return normalize_angle(e_anom - ecc * sin(e_anom))


## Initializes the orbital shape and orientation caches on construction.
func _init() -> void:
	_mass = mass
	_mass_unit = mass_unit
	_recompute_gravity_strength_from_mass()
	_update_orbital_shape_cache()
	_update_orbital_orientation_cache()


## Initializes child tracking and evaluates the initial epoch orbital state.
func _ready() -> void:
	super._ready()
	_evaluate_standalone_or_epoch_state()


## Refreshes configuration warnings when the node is parented or unparented.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PARENTED or what == NOTIFICATION_UNPARENTED:
		update_configuration_warnings()


## Validates hierarchy placement and conic section validity for the Godot scene tree dock.
func _get_configuration_warnings() -> PackedStringArray:
	var warnings: PackedStringArray = []

	if get_parent() is not JrOrreryElement:
		warnings.append(
				"A JrRailOrbiter must be a child of a JrOrbitDriver "
				+ "or another JrRailOrbiter to be stepped automatically."
		)

	if not is_orbit_valid():
		warnings.append("The orbital parameters do not form a valid conic section.")

	if binary_partner != null and is_instance_valid(binary_partner):
		if binary_partner.get_parent() != get_parent():
			warnings.append(
					"Binary partner must share the same parent barycenter JrOrreryElement."
			)

	return warnings


## Overrides [method JrOrreryElement.is_trajectory_valid] to report whether this
## orbiter's conic parameters are valid.
func is_trajectory_valid() -> bool:
	return is_orbit_valid()


## Overrides [method JrOrreryElement.has_keplerian_gizmo] to enable 2D and 3D conic grabbers.
func has_keplerian_gizmo() -> bool:
	return is_orbit_valid()


## Returns [code]true[/code] if the orbital parameters describe a physically valid
## conic section.
func is_orbit_valid() -> bool:
	if eccentricity < 0.0:
		return false
	if semi_major_axis <= 0.0 and eccentricity < 1.0:
		return false
	elif semi_major_axis >= 0.0 and eccentricity > 1.0:
		return false
	return true


## Returns [code]true[/code] if the orbit is gravitationally bound
## ([code]0.0 <= eccentricity < 1.0[/code]).
func is_bound_orbit() -> bool:
	return eccentricity >= 0.0 and eccentricity < 1.0


## Returns [code]true[/code] if the trajectory is an unbound escape trajectory
## ([code]eccentricity >= 1.0[/code]).
func is_unbound_orbit() -> bool:
	return eccentricity >= 1.0


## Returns [code]true[/code] if the animated editor preview ghost currently has
## a valid evaluated position.
func has_ghost_position() -> bool:
	return _has_ghost_position


## Returns the animated editor preview ghost's position relative to the parent
## focus in the coordinate space of [member _reference_plane].
func get_ghost_relative_position() -> Vector3:
	return _ghost_relative_position


## Returns the animated editor preview ghost's 3D orientation [Basis] in the
## coordinate space of [member _reference_plane].
func get_ghost_body_basis() -> Basis:
	return _ghost_body_basis


## Updates the cached [member _reference_plane] reference and connects its
## [signal JrReferenceDirection.changed] signal.
func set_reference_plane(new_reference_plane: JrReferenceDirection) -> void:
	if new_reference_plane == null:
		return
	if _reference_plane != new_reference_plane:
		if (
				_reference_plane != null
				and _reference_plane.changed.is_connected(_on_reference_plane_changed)
		):
			_reference_plane.changed.disconnect(_on_reference_plane_changed)
		_reference_plane = new_reference_plane
		if not _reference_plane.changed.is_connected(_on_reference_plane_changed):
			_reference_plane.changed.connect(_on_reference_plane_changed)
	if is_locked_to_ecliptic():
		_enforce_ecliptic_lock()


## Converts the given [param true_anomaly] (in radians) into the corresponding
## mean anomaly via Kepler's or Barker's equation for any conic section.
func get_mean_anomaly_from_true_anomaly(true_anomaly: float) -> float:
	var anomaly: float = get_eccentric_anomaly_from_true_anomaly(true_anomaly)
	if is_parabolic_trajectory():
		return anomaly + (anomaly * anomaly * anomaly) / 3.0
	if is_unbound_orbit():
		return eccentricity * sinh(anomaly) - anomaly
	return normalize_angle(anomaly - eccentricity * sin(anomaly))


## Returns the 3D position of the body relative to the central focus in the
## coordinate space of [param reference] (or [member _reference_plane] if omitted)
## at the given [param time].
func get_relative_position(time: float, reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var perifocal_position: Vector3 = get_perifocal_position(time)
	return ref.orbit_transform * (_orbital_transform * perifocal_position)


## Returns the 3D orbital velocity vector of the body relative to its parent focus
## in the coordinate space of [param reference] (or [member _reference_plane] if omitted)
## at the given [param time]. When called with no arguments, returns the cached
## parent-relative velocity from the most recent orbital evaluation.
func get_relative_velocity(time: float = NAN, reference: JrReferenceDirection = null) -> Vector3:
	if is_nan(time) and reference == null:
		return _relative_velocity
	var eval_time: float = _last_time if is_nan(time) else time
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	if ref == null:
		return Vector3.ZERO
	var perifocal_velocity: Vector3 = get_perifocal_velocity(eval_time)
	return ref.orbit_transform.basis * (_orbital_transform.basis * perifocal_velocity)


## Returns the right-handed 3D [Basis] of the orbital perifocal frame in the
## coordinate space of [param reference] (or [member _reference_plane] if omitted),
## prior to applying [member axial_tilt] and sidereal rotation.
func get_orbital_plane_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var orbital_plane_basis: Basis = (
			_orbital_transform.basis
			if ref.positive_is_clockwise
			else _orbital_basis_ccw
	)
	return (ref.rotation_basis * orbital_plane_basis).orthonormalized()


## Returns the right-handed 3D [Basis] of the body in the coordinate space of
## [param reference] (or [member _reference_plane] if omitted) after applying
## [member axial_tilt] along [member axial_tilt_longitude], prior to applying
## sidereal spin around the body's polar [code]+Y[/code] axis.
func get_tilted_body_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var orbital_frame: Basis = get_orbital_plane_basis(ref)
	var tilt_basis := Basis.IDENTITY
	if not is_zero_approx(axial_tilt) and (ref == null or not ref.lock_to_ecliptic):
		var tilt_axis := Vector3.FORWARD.rotated(
				Vector3.UP,
				dir_sign * axial_tilt_longitude
		)
		tilt_basis = Basis(tilt_axis, dir_sign * axial_tilt)
	return (orbital_frame * tilt_basis).orthonormalized()


## Returns the right-handed 3D orientation [Basis] of the body at the given [param time]
## in the coordinate space of [param reference] (or [member _reference_plane] if omitted),
## incorporating the orbital plane orientation, [member axial_tilt] along
## [member axial_tilt_longitude], and sidereal spin.
func get_body_basis(time: float, reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var spin_angle: float = normalize_angle(get_rotation_angle(time))
	var spin_basis := Basis(Vector3.UP, dir_sign * spin_angle)
	return (get_tilted_body_basis(ref) * spin_basis).orthonormalized()


## Returns the 3D [Basis] in [param reference] space oriented to the ascending node
## within the ecliptic plane (applying [member longitude_of_the_ascending_node] only).
func get_nodal_ecliptic_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var cw_lan := Basis(Vector3.UP, -longitude_of_the_ascending_node)
	return ref.orbit_transform.basis * cw_lan


## Returns the 3D [Basis] in [param reference] space after applying both
## [member longitude_of_the_ascending_node] and [member inclination] (prior to
## [member argument_of_periapsis]).
func get_inclined_nodal_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var cw_lan := Basis(Vector3.UP, -longitude_of_the_ascending_node)
	var cw_inc := Basis(Vector3.FORWARD, -inclination)
	return ref.orbit_transform.basis * (cw_lan * cw_inc)


## Returns the unit axis in the orbital plane around which [member axial_tilt]
## rotates the body's spin axis, in the coordinate space of [param reference].
func get_axial_tilt_hinge_axis(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var local_hinge := Vector3.FORWARD.rotated(Vector3.UP, dir_sign * axial_tilt_longitude)
	return (get_orbital_plane_basis(ref) * local_hinge).normalized()


## Returns the unit vector in the orbital plane toward which the body's north
## pole ([code]+Y[/code]) tilts when [member axial_tilt] is positive.
func get_axial_tilt_direction(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var local_dir := Vector3(dir_sign, 0.0, 0.0).rotated(
			Vector3.UP,
			dir_sign * axial_tilt_longitude
	)
	return (get_orbital_plane_basis(ref) * local_dir).normalized()


## Returns the 3D position of the periapsis relative to the central focus (parent)
## in the coordinate space of [param reference].
func get_periapsis_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	return periapsis * (ref.orbit_transform.basis * (-_orbital_transform.basis.z))


## Returns the 3D position of the apoapsis relative to the central focus (parent)
## in the coordinate space of [param reference]. Returns [constant Vector3.INF]
## for unbound ([code]eccentricity >= 1.0[/code]) trajectories.
func get_apoapsis_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	if is_unbound_orbit():
		return Vector3.INF
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	return apoapsis * (ref.orbit_transform.basis * _orbital_transform.basis.z)


## Returns the 3D position of the semi-parameter (latus rectum point at +90 degrees
## true anomaly) relative to the central focus (parent) in the coordinate space
## of [param reference].
func get_semi_parameter_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	return _semi_parameter * (ref.orbit_transform.basis * _orbital_transform.basis.x)


## Returns the radial distance from the parent focus used for the ascending node
## and inclination gizmo grabbers in driver-relative focus space.
func get_orbital_plane_gizmo_radius() -> float:
	return periapsis * ORBITAL_PLANE_GIZMO_RADIUS_RATIO


## Returns the 3D position of the ascending node (RAAN) grabber relative to the
## parent focus in the coordinate space of [member _reference_plane].
func get_ascending_node_gizmo_position() -> Vector3:
	return get_orbital_plane_gizmo_radius() * get_ascending_node_direction(_reference_plane)


## Returns the 3D position of the orbital inclination grabber relative to the
## parent focus in the coordinate space of [member _reference_plane].
func get_inclination_gizmo_position() -> Vector3:
	return get_orbital_plane_gizmo_radius() * get_inclination_direction(_reference_plane)


## Overrides [method JrOrreryElement.get_trajectory_path_points] using this orbiter's
## active [member _reference_plane].
func get_trajectory_path_points(point_count: int = 64) -> PackedVector3Array:
	return get_orbit_path_points(_reference_plane, point_count)


## Overrides [method JrOrreryElement.get_trajectory_path_points_2d] using this
## orbiter's active [member _reference_plane] and unbound scaling.
func get_trajectory_path_points_2d(
		point_count: int = 64,
		scale_factor: float = 1.0
) -> PackedVector2Array:
	return get_orbit_path_points_2d(_reference_plane, point_count, scale_factor)


## Generates a [PackedVector3Array] of points tracing the conic path in the
## coordinate space of [param reference] (or [member _reference_plane] if omitted),
## with point density weighted by [member eccentricity] toward high-curvature regions.
func get_orbit_path_points(
		reference: JrReferenceDirection = null,
		point_count: int = 64,
		close_loop: bool = true,
		max_unbound_distance: float = 500.0
) -> PackedVector3Array:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane
	var count: int = maxi(3, point_count)
	var combined_transform: Transform3D = ref.orbit_transform * _orbital_transform
	if is_bound_orbit():
		return _sample_bound_orbit_points(combined_transform, count, close_loop)
	return _sample_unbound_orbit_points(combined_transform, count, max_unbound_distance)


## Invoked when the owning [JrOrbitDriver] changes its distance or time units.
func _on_driver_units_updated() -> void:
	_refresh_gravity_dependent_motion()


## Called downward by [JrOrbitDriver] to provide the effective central gravitational
## parameter [param parent_mu] of this orbiter's immediate parent element.
func set_parent_effective_mu(parent_mu: float) -> void:
	var clamped: float = maxf(0.0, parent_mu)
	if _parent_effective_mu != clamped:
		_parent_effective_mu = clamped
		_refresh_gravity_dependent_motion()


## Returns the effective gravitational parameter [member _parent_effective_mu] of this
## orbiter's immediate parent element.
func get_parent_effective_mu() -> float:
	return _parent_effective_mu


## Returns [code]true[/code] if the orbit is circular ([code]eccentricity ~= 0.0[/code]).
func is_circular_orbit() -> bool:
	return is_zero_approx(eccentricity) and eccentricity >= 0.0


## Returns [code]true[/code] if the orbit is closed and elliptical
## ([code]0.0 <= eccentricity < 1.0[/code]).
func is_elliptical_orbit() -> bool:
	return eccentricity >= 0.0 and eccentricity < 1.0


## Returns [code]true[/code] if the trajectory is parabolic ([code]eccentricity ~= 1.0[/code]).
func is_parabolic_trajectory() -> bool:
	return is_equal_approx(eccentricity, 1.0) and eccentricity >= 1.0


## Evaluates this orbiter's focus-relative position, parent-relative and global
## orbital velocities, driver-relative basis, parent-canceled local transform, and
## heliocentric driver transform at [param time] using [param reference_plane],
## [param parent_driver_transform], and [param parent_global_velocity].
func evaluate_orbital_state(
		time: float,
		reference_plane: JrReferenceDirection,
		parent_driver_transform: Transform3D = Transform3D.IDENTITY,
		parent_global_velocity: Vector3 = Vector3.ZERO
) -> void:
	_last_time = 0.0 if Engine.is_editor_hint() else time
	_parent_driver_basis = parent_driver_transform.basis
	set_reference_plane(reference_plane)
	if not is_orbit_valid() or _reference_plane == null:
		_relative_velocity = Vector3.ZERO
		_global_velocity = parent_global_velocity
		return
	_compute_orbital_transforms(_last_time, parent_driver_transform.origin, parent_global_velocity)


## Convenience method that evaluates the orbital state at [param time] and
## immediately sets [member _system_transform] and emits [signal transform_updated].
func update_position(
		time: float,
		reference_plane: JrReferenceDirection,
		parent_driver_basis: Basis = Basis.IDENTITY
) -> void:
	evaluate_orbital_state(time, reference_plane, Transform3D(parent_driver_basis, Vector3.ZERO))
	set_system_transform(_driver_relative_transform)


## Updates the animated editor preview ghost's position and orientation at
## [param preview_time] and emits [signal editor_ghost_updated].
func update_editor_ghost(
		preview_time: float,
		reference_plane: JrReferenceDirection,
		parent_driver_basis: Basis = Basis.IDENTITY
) -> void:
	_last_preview_time = preview_time
	_parent_driver_basis = parent_driver_basis
	set_reference_plane(reference_plane)
	_refresh_ghost_state()
	editor_ghost_updated.emit(preview_time)


## Computes the unwrapped mean anomaly (in radians) at the given [param time].
func get_mean_anomaly(time: float) -> float:
	return time * mean_motion + mean_anomaly_at_epoch


## Computes the unwrapped sidereal rotation angle (in radians) at the given [param time].
func get_rotation_angle(time: float) -> float:
	return time * rotational_motion + rotation_at_epoch


## Returns the cyclic preview loop duration (in time units) for unbound trajectories,
## extending [param tail_duration] units past the mean anomaly at epoch
## ([code]time = 0.0[/code]) or periapsis passage ([code]mean_anomaly = 0.0[/code]),
## whichever occurs later.
func get_unbound_preview_loop_duration(
		tail_duration: float = UNBOUND_PREVIEW_TAIL_DURATION
) -> float:
	var periapsis_time: float = 0.0
	if mean_motion > 0.0:
		periapsis_time = -mean_anomaly_at_epoch / mean_motion
	return maxf(0.0, periapsis_time) + maxf(0.001, tail_duration)


## Solves for the conic anomaly at the given [param time]:
## - Bound orbits ([code]0 <= e < 1[/code]): Elliptical eccentric anomaly [code]E[/code] ([code]M =
##   E - e * sin(E)[/code]).
## - Parabolic trajectories ([code]e == 1[/code]): Parabolic anomaly [code]D = tan(nu / 2)[/code]
##   ([code]M = D + D^3 / 3[/code]).
## - Hyperbolic trajectories ([code]e > 1[/code]): Hyperbolic anomaly [code]H[/code] ([code]M = e *
##   sinh(H) - H[/code]).
func solve_eccentric_anomaly(time: float, threshold: float = 1e-10) -> float:
	if time == _cached_time and not is_nan(_cached_eccentric_anomaly):
		return _cached_eccentric_anomaly

	if is_parabolic_trajectory():
		var parabolic_m: float = get_mean_anomaly(time)
		var d: float = _solve_parabolic_anomaly(parabolic_m)
		_cached_time = time
		_cached_mean_anomaly = parabolic_m
		_cached_eccentric_anomaly = d
		_cached_inv_dm_de = 1.0 / (1.0 + d * d)
		return d

	if is_unbound_orbit():
		return _solve_hyperbolic_anomaly(time, threshold)

	var mean_anomaly: float = normalize_angle(get_mean_anomaly(time))
	if is_circular_orbit():
		_cached_time = time
		_cached_mean_anomaly = mean_anomaly
		_cached_eccentric_anomaly = mean_anomaly
		_cached_inv_dm_de = 1.0
		return mean_anomaly

	var ek: float = _get_initial_eccentric_anomaly_guess(mean_anomaly)
	var ek1: float = _iterate_eccentric_anomaly(ek, mean_anomaly)
	var iterations: int = 0
	while absf(ek - ek1) > threshold and iterations < MAX_KEPLER_ITERATIONS:
		ek = ek1
		ek1 = _iterate_eccentric_anomaly(ek, mean_anomaly)
		iterations += 1

	_cached_time = time
	_cached_mean_anomaly = mean_anomaly
	_cached_eccentric_anomaly = ek1
	var denom: float = 1.0 - eccentricity * cos(ek1)
	_cached_inv_dm_de = 1.0 if is_zero_approx(denom) else 1.0 / denom
	return ek1


## Converts the given conic anomaly ([param eccentric_anomaly]) into the true anomaly
## (in radians) for elliptical, parabolic, or hyperbolic trajectories.
func get_true_anomaly(eccentric_anomaly: float) -> float:
	if is_parabolic_trajectory():
		return 2.0 * atan(eccentric_anomaly)
	if is_unbound_orbit():
		var half_h: float = eccentric_anomaly / 2.0
		var hyp_opposite: float = sqrt(eccentricity + 1.0) * sinh(half_h)
		var hyp_adjacent: float = sqrt(eccentricity - 1.0) * cosh(half_h)
		return 2.0 * atan2(hyp_opposite, hyp_adjacent)

	var half_e: float = eccentric_anomaly / 2.0
	var opposite: float = sqrt(1.0 + eccentricity) * sin(half_e)
	var adjacent: float = sqrt(maxf(0.0, 1.0 - eccentricity)) * cos(half_e)
	return 2.0 * atan2(opposite, adjacent)


## Converts the given [param true_anomaly] (in radians) into the corresponding
## conic anomaly (elliptical [code]E[/code], parabolic [code]D[/code], or hyperbolic
## [code]H[/code]).
func get_eccentric_anomaly_from_true_anomaly(true_anomaly: float) -> float:
	if is_parabolic_trajectory():
		var clamped_parabolic_nu: float = clampf(true_anomaly, -PI + 0.01, PI - 0.01)
		return tan(clamped_parabolic_nu / 2.0)
	if is_unbound_orbit():
		var max_nu: float = acos(clampf(-1.0 / eccentricity, -1.0, 1.0)) - 0.01
		var clamped_nu: float = clampf(true_anomaly, -max_nu, max_nu)
		var tanh_half_h: float = clampf(
				sqrt((eccentricity - 1.0) / (eccentricity + 1.0)) * tan(clamped_nu / 2.0),
				-0.9999,
				0.9999
		)
		return log((1.0 + tanh_half_h) / (1.0 - tanh_half_h))

	var half_nu: float = true_anomaly / 2.0
	var opposite: float = sqrt(maxf(0.0, 1.0 - eccentricity)) * sin(half_nu)
	var adjacent: float = sqrt(1.0 + eccentricity) * cos(half_nu)
	return 2.0 * atan2(opposite, adjacent)


## Returns the 3D position of the body in its canonical perifocal coordinate frame
## at the given [param time], where the [code]-Z[/code] ([constant Vector3.FORWARD])
## axis points toward periapsis, the [code]+X[/code] ([constant Vector3.RIGHT]) axis
## points toward +90 degrees of true anomaly, and [code]+Y[/code] ([constant Vector3.UP])
## is the orbital normal.
func get_perifocal_position(time: float) -> Vector3:
	var anomaly: float = solve_eccentric_anomaly(time)
	if is_parabolic_trajectory():
		var major_parabolic: float = periapsis * (1.0 - anomaly * anomaly)
		var minor_parabolic: float = 2.0 * periapsis * anomaly
		return Vector3(minor_parabolic, 0.0, -major_parabolic)
	if is_unbound_orbit():
		var major_hyperbolic: float = _semi_major_axis * (cosh(anomaly) - eccentricity)
		var minor_hyperbolic: float = _semi_minor_axis * sinh(anomaly)
		return Vector3(minor_hyperbolic, 0.0, -major_hyperbolic)
	var major_elliptical: float = _semi_major_axis * (cos(anomaly) - eccentricity)
	var minor_elliptical: float = _semi_minor_axis * sin(anomaly)
	return Vector3(minor_elliptical, 0.0, -major_elliptical)


## Returns the 3D orbital velocity vector of the body in its canonical perifocal
## coordinate frame at the given [param time] (or [member _last_time] if omitted),
## expressed in orbital distance units per unit of simulation time.
func get_perifocal_velocity(time: float = NAN) -> Vector3:
	if not is_orbit_valid() or mean_motion <= 0.0:
		return Vector3.ZERO
	var eval_time: float = _last_time if is_nan(time) else time
	var anomaly: float = solve_eccentric_anomaly(eval_time)
	if is_parabolic_trajectory():
		return _compute_parabolic_perifocal_velocity(anomaly)
	if is_unbound_orbit():
		return _compute_hyperbolic_perifocal_velocity(anomaly)
	return _compute_elliptical_perifocal_velocity(anomaly)


## Returns the unit direction vector pointing from the central focus toward the
## ascending node in the coordinate space of [param reference].
func get_ascending_node_direction(reference: JrReferenceDirection = null) -> Vector3:
	return -get_nodal_ecliptic_basis(reference).z


## Returns the unit normal vector orthogonal to the tilted orbital plane
## in the coordinate space of [param reference].
func get_inclination_direction(reference: JrReferenceDirection = null) -> Vector3:
	return get_inclined_nodal_basis(reference).y


## Returns the 3D position of the axial tilt grabber (at the tip of the north spin pole)
## relative to the parent focus in the coordinate space of [member _reference_plane].
func get_axial_tilt_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var body_pos: Vector3 = get_relative_position(0.0, _reference_plane)
	var north_pole: Vector3 = get_tilted_body_basis(_reference_plane).y
	return body_pos + north_pole * (body_gizmo_radius * AXIAL_TILT_GIZMO_RADIUS_SCALE)


## Returns the 3D position of the axial tilt longitude grabber (in the orbital plane
## around the body) relative to the parent focus in the coordinate space of
## [member _reference_plane].
func get_axial_tilt_longitude_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var body_pos: Vector3 = get_relative_position(0.0, _reference_plane)
	var tilt_dir: Vector3 = get_axial_tilt_direction(_reference_plane)
	return body_pos + tilt_dir * body_gizmo_radius


## Returns the 3D position of the starting day angle ([member rotation_at_epoch])
## grabber on the body's equatorial ring relative to the parent focus in the coordinate
## space of [member _reference_plane].
func get_starting_day_angle_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var body_pos: Vector3 = get_relative_position(0.0, _reference_plane)
	var prime_meridian: Vector3 = -get_body_basis(0.0, _reference_plane).z
	return body_pos + prime_meridian * (body_gizmo_radius * STARTING_DAY_GIZMO_RADIUS_SCALE)


## Generates a [PackedVector2Array] of points projecting the 3D conic path onto
## the top-down X-Z plane ([code]Vector2(x, z) * scale_factor[/code]).
func get_orbit_path_points_2d(
		reference: JrReferenceDirection = null,
		point_count: int = 64,
		scale_factor: float = 1.0,
		close_loop: bool = true
) -> PackedVector2Array:
	var safe_scale: float = maxf(0.0001, absf(scale_factor))
	var max_dist_3d: float = 10000.0 / safe_scale
	var points_3d: PackedVector3Array = get_orbit_path_points(
			reference,
			point_count,
			close_loop,
			max_dist_3d
	)
	var points_2d: PackedVector2Array = []
	points_2d.resize(points_3d.size())
	for i: int in range(points_3d.size()):
		var pt: Vector3 = points_3d[i]
		points_2d[i] = Vector2(pt.x, pt.z) * scale_factor
	return points_2d


## Updates [member mean_motion] and [member binary_partner] coupling when mass,
## orbital geometry, or parent effective gravity changes.
func _refresh_gravity_dependent_motion() -> void:
	if _updating_synced_motion:
		return
	if binary_partner != null and is_instance_valid(binary_partner):
		_sync_with_binary_partner()
		return
	if not sync_motion_to_gravity or not is_orbit_valid():
		return

	var mu_eff: float = _parent_effective_mu + _gravity_strength
	if mu_eff <= 0.0:
		return

	_updating_synced_motion = true
	var derived_n: float = compute_keplerian_mean_motion(periapsis, eccentricity, mu_eff)
	if derived_n > 0.0:
		mean_motion = derived_n
	_updating_synced_motion = false


## Synchronizes coupled orbital elements, mass-weighted periapsis, and shared binary
## mean motion with [member binary_partner].
func _sync_with_binary_partner() -> void:
	if _syncing_binary_partner or binary_partner == null or not is_instance_valid(binary_partner):
		return
	if binary_partner._syncing_binary_partner:
		return

	_syncing_binary_partner = true
	binary_partner._syncing_binary_partner = true

	binary_partner.eccentricity = eccentricity
	binary_partner.inclination = inclination
	binary_partner.longitude_of_the_ascending_node = longitude_of_the_ascending_node
	binary_partner.argument_of_periapsis = normalize_angle(argument_of_periapsis + PI)
	binary_partner.mean_anomaly_at_epoch = mean_anomaly_at_epoch

	var mu_self: float = _gravity_strength
	var mu_partner: float = binary_partner.get_gravity_strength()
	if mu_self > 0.0 and mu_partner > 0.0:
		var target_partner_pe: float = maxf(1e-6, periapsis * (mu_self / mu_partner))
		binary_partner.periapsis = target_partner_pe

	var mu_total: float = mu_self + mu_partner
	if mu_total > 0.0 and is_bound_orbit() and binary_partner.is_bound_orbit():
		var total_a: float = _semi_major_axis + binary_partner.semi_major_axis
		if total_a > 0.0:
			var shared_n: float = sqrt(mu_total / (total_a * total_a * total_a))
			mean_motion = shared_n
			binary_partner.mean_motion = shared_n
	else:
		binary_partner.mean_motion = mean_motion

	binary_partner._syncing_binary_partner = false
	_syncing_binary_partner = false


## Recomputes [member _ghost_relative_position] and [member _ghost_body_basis]
## at [member _last_preview_time].
func _refresh_ghost_state() -> void:
	if not is_orbit_valid() or _reference_plane == null:
		_has_ghost_position = false
		return
	var effective_time: float = _last_preview_time
	if is_unbound_orbit():
		var loop_duration: float = get_unbound_preview_loop_duration()
		effective_time = fposmod(_last_preview_time, loop_duration)
	_ghost_relative_position = get_relative_position(effective_time, _reference_plane)
	_ghost_body_basis = get_body_basis(effective_time, _reference_plane)
	_has_ghost_position = true


## Handles changes to the active [member _reference_plane].
func _on_reference_plane_changed() -> void:
	if is_locked_to_ecliptic():
		_enforce_ecliptic_lock()
	_update_orbital_orientation_cache()
	_refresh_ghost_state()
	orbit_visuals_changed.emit()


## Snaps [member inclination] to flat prograde ([code]0.0[/code]) or flat retrograde
## ([code]PI[/code]) and zeroes [member axial_tilt] when [method is_locked_to_ecliptic] is active.
func _enforce_ecliptic_lock() -> void:
	var target_inc: float = PI if absf(inclination) > PI * 0.5 else 0.0
	var changed_any: bool = false
	if not is_equal_approx(inclination, target_inc):
		inclination = target_inc
		changed_any = true
	if not is_zero_approx(axial_tilt):
		axial_tilt = 0.0
		changed_any = true
	if changed_any:
		_update_orbital_orientation_cache()
		_invalidate_anomaly_cache()


## Computes [member _focus_relative_position], [member _relative_velocity],
## [member _global_velocity], [member _driver_relative_basis],
## [member _local_orbital_transform], and [member _driver_relative_transform]
## at [param eval_time] given [param parent_driver_origin] and [param parent_global_velocity].
func _compute_orbital_transforms(
		eval_time: float,
		parent_driver_origin: Vector3,
		parent_global_velocity: Vector3 = Vector3.ZERO
) -> void:
	_focus_relative_position = get_relative_position(eval_time, _reference_plane)
	_relative_velocity = get_relative_velocity(eval_time, _reference_plane)
	_global_velocity = parent_global_velocity + _relative_velocity
	_driver_relative_basis = get_body_basis(eval_time, _reference_plane)
	var inv_parent: Basis = _parent_driver_basis.transposed()
	var local_pos: Vector3 = inv_parent * _focus_relative_position
	var local_basis: Basis = (inv_parent * _driver_relative_basis).orthonormalized()
	_local_orbital_transform = Transform3D(local_basis, local_pos)
	_driver_relative_transform = Transform3D(
			_driver_relative_basis,
			parent_driver_origin + _focus_relative_position
	)


## Evaluates the orbiter's state when modified in the Inspector or gizmos,
## updating ghost state, warnings, and notifying listeners.
func _notify_orbital_parameters_changed() -> void:
	_evaluate_standalone_or_epoch_state()
	update_configuration_warnings()
	orbit_visuals_changed.emit()


## Evaluates local and fallback system transforms at epoch (in editor) or
## [member _last_time] (at runtime) prior to a full driver pass.
func _evaluate_standalone_or_epoch_state() -> void:
	if not is_orbit_valid() or _reference_plane == null:
		_has_ghost_position = false
		_relative_velocity = Vector3.ZERO
		return
	var target_time: float = 0.0 if Engine.is_editor_hint() else _last_time
	var parent_origin: Vector3 = _driver_relative_transform.origin - _focus_relative_position
	var parent_velocity: Vector3 = _global_velocity - _relative_velocity
	_compute_orbital_transforms(target_time, parent_origin, parent_velocity)
	_system_transform = _driver_relative_transform
	_refresh_ghost_state()
	transform_updated.emit()


## Recalculates the cached conic geometry properties ([member _semi_major_axis],
## [member _semi_minor_axis], and [member _semi_parameter]) from [member periapsis]
## and [member eccentricity].
func _update_orbital_shape_cache() -> void:
	_semi_parameter = periapsis * (1.0 + eccentricity)
	if is_parabolic_trajectory():
		_semi_major_axis = INF
		_semi_minor_axis = INF
	else:
		_semi_major_axis = periapsis / (1.0 - eccentricity)
		_semi_minor_axis = absf(_semi_major_axis) * sqrt(absf(1.0 - eccentricity ** 2))


## Recalculates the cached 3D orbital orientation transforms from
## [member longitude_of_the_ascending_node], [member inclination], and
## [member argument_of_periapsis].
func _update_orbital_orientation_cache() -> void:
	var eff_inc: float = (
			(PI if absf(inclination) > PI * 0.5 else 0.0)
			if is_locked_to_ecliptic()
			else inclination
	)
	var cw_lan := Basis(Vector3.UP, -longitude_of_the_ascending_node)
	var cw_inc := Basis(Vector3.FORWARD, -eff_inc)
	var cw_arg := Basis(Vector3.UP, -argument_of_periapsis)
	var cw_basis: Basis = (cw_lan * cw_inc * cw_arg).orthonormalized()
	_orbital_transform = Transform3D(cw_basis, Vector3.ZERO)

	var ccw_lan := Basis(Vector3.UP, longitude_of_the_ascending_node)
	var ccw_inc := Basis(Vector3.FORWARD, eff_inc)
	var ccw_arg := Basis(Vector3.UP, argument_of_periapsis)
	_orbital_basis_ccw = (ccw_lan * ccw_inc * ccw_arg).orthonormalized()


## Invalidates the cached eccentric anomaly solution so the next query recomputes
## from a cold start.
func _invalidate_anomaly_cache() -> void:
	_cached_time = NAN
	_cached_mean_anomaly = NAN
	_cached_eccentric_anomaly = NAN
	_cached_inv_dm_de = 1.0


## Selects the initial elliptical eccentric anomaly guess (in radians) for a given
## normalized [param mean_anomaly], using first-order Taylor extrapolation from the
## cached solution when the wrapped angular step is within [constant WARM_START_MAX_DELTA_M].
func _get_initial_eccentric_anomaly_guess(mean_anomaly: float) -> float:
	if not is_nan(_cached_eccentric_anomaly):
		var delta_m: float = normalize_angle(mean_anomaly - _cached_mean_anomaly)
		if absf(delta_m) <= WARM_START_MAX_DELTA_M:
			return normalize_angle(_cached_eccentric_anomaly + delta_m * _cached_inv_dm_de)
	return mean_anomaly if eccentricity < 0.8 else PI


## Selects the initial hyperbolic eccentric anomaly guess for an unwrapped
## [param mean_anomaly], using first-order Taylor extrapolation when stepping smoothly.
func _get_initial_hyperbolic_anomaly_guess(mean_anomaly: float) -> float:
	if not is_nan(_cached_eccentric_anomaly):
		var delta_m: float = mean_anomaly - _cached_mean_anomaly
		if absf(delta_m) <= WARM_START_MAX_DELTA_M:
			return _cached_eccentric_anomaly + delta_m * _cached_inv_dm_de
	var abs_m: float = absf(mean_anomaly)
	if abs_m < 1.0:
		return mean_anomaly
	return signf(mean_anomaly) * log(2.0 * abs_m / eccentricity + 1.8)


## Performs a single Newton-Raphson step to refine the elliptical eccentric anomaly
## estimate [param ek] for the given [param mean_anomaly] (in radians).
func _iterate_eccentric_anomaly(ek: float, mean_anomaly: float) -> float:
	var sin_ek: float = sin(ek)
	var cos_ek: float = cos(ek)
	var numerator: float = ek - eccentricity * sin_ek - mean_anomaly
	var denominator: float = 1.0 - eccentricity * cos_ek
	if is_zero_approx(denominator):
		return ek
	return ek - numerator / denominator


## Performs a single Newton-Raphson step to refine the hyperbolic anomaly estimate
## [param hk] for the given unwrapped [param mean_anomaly].
func _iterate_hyperbolic_anomaly(hk: float, mean_anomaly: float) -> float:
	var sinh_hk: float = sinh(hk)
	var cosh_hk: float = cosh(hk)
	var numerator: float = eccentricity * sinh_hk - hk - mean_anomaly
	var denominator: float = eccentricity * cosh_hk - 1.0
	if is_zero_approx(denominator):
		return hk
	return hk - numerator / denominator


## Solves Barker's equation ([code]M = D + D^3 / 3[/code]) analytically for the
## parabolic anomaly [code]D = tan(true_anomaly / 2)[/code].
func _solve_parabolic_anomaly(mean_anomaly: float) -> float:
	var b: float = 1.5 * mean_anomaly
	var a: float = pow(b + sqrt(1.0 + b * b), 1.0 / 3.0)
	return a - 1.0 / a


## Solves the hyperbolic Kepler equation ([code]M = e * sinh(H) - H[/code]) for
## the hyperbolic anomaly [code]H[/code] at [param time].
func _solve_hyperbolic_anomaly(time: float, threshold: float) -> float:
	var mean_anomaly: float = get_mean_anomaly(time)
	var hk: float = _get_initial_hyperbolic_anomaly_guess(mean_anomaly)
	var hk1: float = _iterate_hyperbolic_anomaly(hk, mean_anomaly)
	var iterations: int = 0
	while absf(hk - hk1) > threshold and iterations < MAX_KEPLER_ITERATIONS:
		hk = hk1
		hk1 = _iterate_hyperbolic_anomaly(hk, mean_anomaly)
		iterations += 1

	_cached_time = time
	_cached_mean_anomaly = mean_anomaly
	_cached_eccentric_anomaly = hk1
	var denom: float = eccentricity * cosh(hk1) - 1.0
	_cached_inv_dm_de = 1.0 if is_zero_approx(denom) else 1.0 / denom
	return hk1


## Computes the perifocal orbital velocity vector for a parabolic trajectory
## ([code]eccentricity == 1.0[/code]) at parabolic anomaly [param d].
func _compute_parabolic_perifocal_velocity(d: float) -> Vector3:
	var dd_dt: float = mean_motion / (1.0 + d * d)
	var vx: float = 2.0 * periapsis * dd_dt
	var vz: float = 2.0 * periapsis * d * dd_dt
	return Vector3(vx, 0.0, vz)


## Computes the perifocal orbital velocity vector for a hyperbolic trajectory
## ([code]eccentricity > 1.0[/code]) at hyperbolic anomaly [param h].
func _compute_hyperbolic_perifocal_velocity(h: float) -> Vector3:
	var denom: float = eccentricity * cosh(h) - 1.0
	if is_zero_approx(denom):
		return Vector3.ZERO
	var dh_dt: float = mean_motion / denom
	var vx: float = _semi_minor_axis * cosh(h) * dh_dt
	var vz: float = -_semi_major_axis * sinh(h) * dh_dt
	return Vector3(vx, 0.0, vz)


## Computes the perifocal orbital velocity vector for a bound circular or elliptical
## orbit ([code]0.0 <= eccentricity < 1.0[/code]) at eccentric anomaly [param e_anom].
func _compute_elliptical_perifocal_velocity(e_anom: float) -> Vector3:
	var denom: float = 1.0 - eccentricity * cos(e_anom)
	if is_zero_approx(denom):
		return Vector3.ZERO
	var de_dt: float = mean_motion / denom
	var vx: float = _semi_minor_axis * cos(e_anom) * de_dt
	var vz: float = _semi_major_axis * sin(e_anom) * de_dt
	return Vector3(vx, 0.0, vz)


## Samples a closed circular or elliptical orbit ([code]0.0 <= eccentricity < 1.0[/code]),
## warping the eccentric anomaly step by [member eccentricity] to concentrate points
## around both high-curvature apsides (periapsis and apoapsis).
func _sample_bound_orbit_points(
		combined_transform: Transform3D,
		count: int,
		close_loop: bool
) -> PackedVector3Array:
	var total_vertices: int = count + 1 if close_loop else count
	var points: PackedVector3Array = []
	points.resize(total_vertices)

	var warp_weight: float = 0.45 * eccentricity
	var step: float = TAU / float(count)

	for i: int in range(count):
		var u: float = -PI + float(i) * step
		var eccentric_anomaly: float = u - 0.5 * warp_weight * sin(2.0 * u)
		var major_pos: float = _semi_major_axis * (cos(eccentric_anomaly) - eccentricity)
		var minor_pos: float = _semi_minor_axis * sin(eccentric_anomaly)
		var perifocal_pos := Vector3(minor_pos, 0.0, -major_pos)
		points[i] = combined_transform * perifocal_pos

	if close_loop:
		points[count] = points[0]

	return points


## Samples an open parabolic or hyperbolic trajectory ([code]eccentricity >= 1.0[/code])
## up to [param max_distance] from the focus, concentrating points around the
## high-curvature periapsis turn ([code]true_anomaly = 0.0[/code]).
func _sample_unbound_orbit_points(
		combined_transform: Transform3D,
		count: int,
		max_distance: float
) -> PackedVector3Array:
	var loop_end_time: float = get_unbound_preview_loop_duration()
	var epoch_dist: float = get_perifocal_position(0.0).length()
	var loop_end_dist: float = get_perifocal_position(loop_end_time).length()
	var clamped_max_dist: float = maxf(max_distance, maxf(epoch_dist, loop_end_dist))
	clamped_max_dist = maxf(clamped_max_dist, periapsis * 1.01)
	var cos_limit: float = clampf(
			(_semi_parameter / clamped_max_dist - 1.0) / eccentricity,
			-0.9999,
			1.0
	)
	var max_true_anomaly: float = acos(cos_limit)

	var points: PackedVector3Array = []
	points.resize(count)

	var warp_alpha: float = clampf(0.65 * (1.0 - 1.0 / (1.0 + eccentricity)), 0.0, 0.65)
	var linear_weight: float = 1.0 - warp_alpha
	var inv_last: float = 1.0 / float(count - 1)

	for i: int in range(count):
		var s: float = -1.0 + 2.0 * float(i) * inv_last
		var warped_s: float = linear_weight * s + warp_alpha * (s * s * s)
		var true_anomaly: float = max_true_anomaly * warped_s
		var radius: float = _semi_parameter / (1.0 + eccentricity * cos(true_anomaly))
		var perifocal_pos := Vector3(
				radius * sin(true_anomaly),
				0.0,
				-radius * cos(true_anomaly)
		)
		points[i] = combined_transform * perifocal_pos

	return points
