@tool
@icon("res://addons/jr_orrery/icons/ballistic.svg")

## Simulation [Node] representing a physics-driven ballistic body within a [JrOrbitDriver] hierarchy.
##
## Supports authoring initial conditions as either 3D Cartesian state vectors or
## Keplerian orbital elements, continuous external thrust in Inertial, Body-Local,
## or Orbital (Prograde/Normal/Radial) frames, optional third-body tidal perturbations,
## and automatic or manual dominant gravity-well anchoring. At runtime, [JrOrbitDriver]
## steps all tracked [JrBallisticOrbiter] nodes via a hybrid Universal-Variable
## Kepler Kick-Drift-Kick GPU compute solver ([JrBallisticComputeSolver]) with
## automatic Tier-1 epoch-frozen coasting when unperturbed.
extends JrOrreryElement
class_name JrBallisticOrbiter

## Emitted when this ballistic body's dominant gravitational anchor changes from
## [param previous_body] to [param current_body].
signal dominant_body_changed(previous_body: JrOrreryElement, current_body: JrOrreryElement);

## Determines how the initial epoch state of this ballistic body is authored and edited.
enum InitialStateMode {
	## Authored via Cartesian [member initial_position] and [member initial_velocity] vectors
	## with 2D/3D position and velocity vector gizmo handles.
	STATE_VECTORS = 0,
	## Authored via Keplerian conic elements ([member initial_periapsis], [member initial_eccentricity],
	## etc.) with 2D/3D orbital conic gizmo handles.
	KEPLERIAN_ELEMENTS = 1,
}

## Coordinate frame in which [member external_acceleration] is interpreted.
enum ThrustFrame {
	## Unrotated inertial coordinate frame of the owning [JrOrbitDriver].
	INERTIAL = 0,
	## Body-local manual orientation frame ([member manual_rotation]).
	BODY_LOCAL = 1,
	## Instantaneous orbital frame relative to the active dominant anchor:
	## [code]-Z[/code] ([constant Vector3.FORWARD]) = Prograde,
	## [code]+Y[/code] ([constant Vector3.UP]) = Orbit Normal,
	## [code]+X[/code] ([constant Vector3.RIGHT]) = Radial Outward.
	ORBITAL = 2,
}

## Minimum editor preview duration in simulation time units.
const MIN_PREVIEW_DURATION: float = 0.1;

#region Private Members
## Internal non-tree [JrRailOrbiter] helper used to evaluate Keplerian initial states and gizmo geometry.
var _conic_helper: JrRailOrbiter = null;

## Re-entrancy guard preventing recursive synchronization between Cartesian and Keplerian initial properties.
var _syncing_initial_state: bool = false;

## Effective gravitational parameter (mu) of the spawn/anchor parent supplied downward by [JrOrbitDriver].
var _parent_effective_mu: float = 0.0;

## Current dominant gravitational anchor [JrOrreryElement] relative to which this body is integrated.
var _anchor_element: JrOrreryElement = null;

## Original parent [JrOrreryElement] recorded prior to [method JrOrbitDriver.collect_ballistic_nodes].
var _spawn_parent: JrOrreryElement = null;

## Guard set during [method JrOrbitDriver.collect_ballistic_nodes] so reparenting does not reset live state.
var _suppress_reinit_on_reparent: bool = false;

## Current position offset relative to [member _anchor_element] in the driver's unrotated reference frame.
var _anchor_relative_position: Vector3 = Vector3(0.0, 0.0, -10.0);

## Current velocity vector relative to [member _anchor_element] in the driver's unrotated reference frame.
var _anchor_relative_velocity: Vector3 = Vector3(4.0, 0.0, 0.0);

## Current inertial position of this body in the unshifted [JrOrbitDriver] frame.
var _inertial_position: Vector3 = Vector3(0.0, 0.0, -10.0);

## Current inertial velocity of this body in the unshifted [JrOrbitDriver] frame.
var _inertial_velocity: Vector3 = Vector3(4.0, 0.0, 0.0);

## Whether this body is currently coasting in Tier-1 epoch-frozen analytical Keplerian mode.
var _is_on_rails: bool = false;

## Anchor-relative position frozen when entering Tier-1 on-rails coasting.
var _epoch_anchor_pos: Vector3 = Vector3.ZERO;

## Anchor-relative velocity frozen when entering Tier-1 on-rails coasting.
var _epoch_anchor_vel: Vector3 = Vector3.ZERO;

## Simulation timestamp (64-bit float) at which Tier-1 on-rails coasting began.
var _epoch_time: float = 0.0;

## Analytical orbital period (64-bit float) of the frozen Tier-1 conic (0.0 if unbound).
var _epoch_orbital_period: float = 0.0;

## Analytical maximum distance from [member _anchor_element] below which third-body tides remain culled.
var _epoch_safe_max_radius: float = 0.0;

## Manual right-handed local orientation [Basis] prior to reference-frame rotation.
var _manual_basis: Basis = Basis.IDENTITY;

## Guard preventing recursive Euler/Basis synchronization when updating orientation.
var _syncing_orientation: bool = false;

## Whether runtime state has been initialized from the parent element.
var _has_initialized_state: bool = false;

## Flag requesting re-initialization from the spawn parent's orbital state on next step.
var _needs_parent_reinit: bool = true;

## Most recently supplied inertial origin of this node's immediate scene parent [JrOrreryElement].
var _last_parent_origin: Vector3 = Vector3.ZERO;

## Most recently supplied global inertial velocity of this node's immediate scene parent [JrOrreryElement].
var _last_parent_velocity: Vector3 = Vector3.ZERO;

## Most recently supplied inertial origin of [member _anchor_element].
var _last_anchor_origin: Vector3 = Vector3.ZERO;

## Most recently supplied global inertial velocity of [member _anchor_element].
var _last_anchor_velocity: Vector3 = Vector3.ZERO;

## Offset from the immediate scene parent to [member _anchor_element] ([code]r_anchor - r_parent[/code]).
var _anchor_to_parent_offset: Vector3 = Vector3.ZERO;

## Velocity difference between [member _anchor_element] and the scene parent ([code]v_anchor - v_parent[/code]).
var _anchor_to_parent_velocity: Vector3 = Vector3.ZERO;

## Cached parent-focus-relative trajectory preview vertices for editor visualization.
var _preview_path_points: PackedVector3Array = [];

## Cached live runtime forward-integrated trajectory points in inertial driver space.
var _forecast_inertial_points: PackedVector3Array = [];

## Cached live runtime forward trajectory points relative to the active perspective
## node's future origin (unrotated inertial axes).
var _forecast_unrotated_points: PackedVector3Array = [];

## Cached live runtime forward trajectory points in the active system frame
## (matches [method JrOrreryElement.get_system_transform] origin at index 0).
var _forecast_system_points: PackedVector3Array = [];

## Cached per-point reference gravity-well instance IDs corresponding to [member _forecast_system_points].
var _forecast_reference_ids: PackedInt64Array = [];

## Flag set when velocity, position, thrust, or horizon parameters change at runtime
## so [JrOrbitDriver] recomputes the forecast immediately even when paused.
var _forecast_dirty: bool = true;

## Last editor preview timestamp supplied to [method update_editor_ghost].
var _last_preview_time: float = 0.0;

## Interpolated position of the animated editor ghost relative to the parent focus.
var _ghost_relative_position: Vector3 = Vector3.ZERO;

## Orientation [Basis] of the animated editor ghost in the driver's reference space.
var _ghost_body_basis: Basis = Basis.IDENTITY;

## Whether [member _ghost_relative_position] currently holds a valid preview position.
var _has_ghost_position: bool = false;

## Whether the cached Keplerian elements need to be recomputed from [member initial_position] and [member initial_velocity].
var _keplerian_cache_dirty: bool = false;
#endregion

#region Initial State Exports
@export_group("Initial State")
## Selects whether initial conditions and viewport gizmos use Cartesian state vectors
## ([constant InitialStateMode.STATE_VECTORS]) or Keplerian orbital elements
## ([constant InitialStateMode.KEPLERIAN_ELEMENTS]).
@export
var initial_state_mode: InitialStateMode = InitialStateMode.STATE_VECTORS:
	set(value):
		if initial_state_mode != value:
			initial_state_mode = value;
			if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
				_sync_keplerian_from_vectors();
			else:
				_sync_vectors_from_keplerian();
			notify_property_list_changed();
			orbit_visuals_changed.emit();

## Starting 3D position offset relative to the spawn parent [JrOrreryElement]
## at epoch, expressed in the driver's [JrReferenceDirection] frame.
@export
var initial_position: Vector3 = Vector3(0.0, 0.0, -10.0):
	set(value):
		var target_pos: Vector3 = Vector3(value.x, 0.0, value.z) if is_locked_to_ecliptic() else value;
		if not initial_position.is_equal_approx(target_pos):
			initial_position = target_pos;
			if not _syncing_initial_state:
				_mark_or_sync_keplerian_from_vectors();
				_on_initial_parameters_changed();

## Starting 3D velocity vector relative to the spawn parent [JrOrreryElement]
## at epoch, expressed in driver distance units per driver time unit.
@export
var initial_velocity: Vector3 = Vector3(4.0, 0.0, 0.0):
	set(value):
		var target_vel: Vector3 = Vector3(value.x, 0.0, value.z) if is_locked_to_ecliptic() else value;
		if not initial_velocity.is_equal_approx(target_vel):
			initial_velocity = target_vel;
			if not _syncing_initial_state:
				_mark_or_sync_keplerian_from_vectors();
				_on_initial_parameters_changed();

## Visual scale factor applied to [member initial_velocity] when drawing the
## interactive velocity vector arrow in the 2D and 3D editor viewports.
@export_range(0.05, 20.0, 0.05, "or_greater")
var velocity_gizmo_scale: float = 0.5:
	set(value):
		var clamped: float = maxf(0.01, value);
		if not is_equal_approx(velocity_gizmo_scale, clamped):
			velocity_gizmo_scale = clamped;
			orbit_visuals_changed.emit();

@export_subgroup("Keplerian Initial Orbit")
## Initial conic eccentricity ([code]0[/code] circular, [code](0, 1)[/code] elliptical,
## [code]1[/code] parabolic, [code]> 1[/code] hyperbolic). Synchronized with [member initial_position]
## and [member initial_velocity].
@export_range(0.0, 2.0, 0.0001, "or_greater")
var initial_eccentricity: float = 0.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if not is_equal_approx(initial_eccentricity, clamped):
			initial_eccentricity = clamped;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.eccentricity = clamped;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();

## Initial closest approach distance (periapsis radius) to the spawn parent focus.
@export_range(0.001, 1000000000.0, 0.001, "exp", "or_greater")
var initial_periapsis: float = 10.0:
	set(value):
		var clamped: float = maxf(1e-6, value);
		if not is_equal_approx(initial_periapsis, clamped):
			initial_periapsis = clamped;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.periapsis = clamped;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();

## Initial farthest distance (apoapsis radius) from the spawn parent focus.
## Returns [constant @GDScript.INF] for unbound ([code]initial_eccentricity >= 1.0[/code]) orbits.
@export
var initial_apoapsis: float = 10.0:
	get:
		return _ensure_conic_helper().apoapsis;
	set(value):
		if is_inf(value) or is_nan(value):
			return;
		var clamped_ap: float = maxf(value, initial_periapsis);
		initial_eccentricity = (clamped_ap - initial_periapsis) / (clamped_ap + initial_periapsis);

## Initial orbital inclination in radians within [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var initial_inclination: float = 0.0:
	set(value):
		var normalized: float = JrRailOrbiter._normalize_angle(value);
		if is_locked_to_ecliptic():
			normalized = PI if absf(normalized) > PI * 0.5 else 0.0;
		if not is_equal_approx(initial_inclination, normalized):
			initial_inclination = normalized;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.inclination = normalized;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();

## Initial longitude of the ascending node (RAAN) in radians within [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var initial_longitude_of_the_ascending_node: float = 0.0:
	set(value):
		var normalized: float = JrRailOrbiter._normalize_angle(value);
		if not is_equal_approx(initial_longitude_of_the_ascending_node, normalized):
			initial_longitude_of_the_ascending_node = normalized;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.longitude_of_the_ascending_node = normalized;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();

## Initial argument of periapsis in radians within [code](-PI, PI][/code].
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees")
var initial_argument_of_periapsis: float = 0.0:
	set(value):
		var normalized: float = JrRailOrbiter._normalize_angle(value);
		if not is_equal_approx(initial_argument_of_periapsis, normalized):
			initial_argument_of_periapsis = normalized;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.argument_of_periapsis = normalized;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();

## Initial mean anomaly at epoch in radians.
@export_range(-180.0, 180.0, 0.1, "radians_as_degrees", "or_less", "or_greater")
var initial_mean_anomaly_at_epoch: float = 0.0:
	set(value):
		var target_val: float = JrRailOrbiter._normalize_angle(value) if initial_eccentricity < 1.0 else value;
		if not is_equal_approx(initial_mean_anomaly_at_epoch, target_val):
			initial_mean_anomaly_at_epoch = target_val;
			if _conic_helper != null and is_instance_valid(_conic_helper):
				_conic_helper.mean_anomaly_at_epoch = target_val;
			if not _syncing_initial_state:
				_sync_vectors_from_keplerian();
				_on_initial_parameters_changed();
#endregion

#region Non-Exported Conic Gizmo Property Aliases
## Alias forwarding to [member initial_periapsis] for interactive 2D/3D conic gizmos.
var periapsis: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_periapsis;
	set(value):
		initial_periapsis = value;

## Alias forwarding to [member initial_apoapsis] for interactive 2D/3D conic gizmos.
var apoapsis: float:
	get:
		return initial_apoapsis;
	set(value):
		initial_apoapsis = value;

## Alias forwarding to [member initial_eccentricity] for interactive 2D/3D conic gizmos.
var eccentricity: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_eccentricity;
	set(value):
		initial_eccentricity = value;

## Alias forwarding to [member initial_inclination] for interactive 2D/3D conic gizmos.
var inclination: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_inclination;
	set(value):
		initial_inclination = value;

## Alias forwarding to [member initial_longitude_of_the_ascending_node] for interactive 2D/3D conic gizmos.
var longitude_of_the_ascending_node: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_longitude_of_the_ascending_node;
	set(value):
		initial_longitude_of_the_ascending_node = value;

## Alias forwarding to [member initial_argument_of_periapsis] for interactive 2D/3D conic gizmos.
var argument_of_periapsis: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_argument_of_periapsis;
	set(value):
		initial_argument_of_periapsis = value;

## Alias forwarding to [member initial_mean_anomaly_at_epoch] for interactive 2D/3D conic gizmos.
var mean_anomaly_at_epoch: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_mean_anomaly_at_epoch;
	set(value):
		initial_mean_anomaly_at_epoch = value;

## Alias forwarding to [member initial_mean_anomaly_at_epoch].
var anomaly_at_epoch: float:
	get:
		if _keplerian_cache_dirty and not _syncing_initial_state:
			_sync_keplerian_from_vectors();
		return initial_mean_anomaly_at_epoch;
	set(value):
		initial_mean_anomaly_at_epoch = value;

## Alias forwarding to [member manual_rotation] X (pitch/tilt) for 3D gizmo compatibility.
var axial_tilt: float:
	get:
		return _ensure_conic_helper().axial_tilt;
	set(value):
		_ensure_conic_helper().axial_tilt = value;
		set_manual_basis(_ensure_conic_helper().get_body_basis(0.0, _reference_plane));

## Alias forwarding to [member _conic_helper] axial tilt longitude for 3D gizmo compatibility.
var axial_tilt_longitude: float:
	get:
		return _ensure_conic_helper().axial_tilt_longitude;
	set(value):
		_ensure_conic_helper().axial_tilt_longitude = value;
		set_manual_basis(_ensure_conic_helper().get_body_basis(0.0, _reference_plane));

## Alias forwarding to [member _conic_helper] rotation at epoch for 3D gizmo compatibility.
var rotation_at_epoch: float:
	get:
		return _ensure_conic_helper().rotation_at_epoch;
	set(value):
		_ensure_conic_helper().rotation_at_epoch = value;
		set_manual_basis(_ensure_conic_helper().get_body_basis(0.0, _reference_plane));
#endregion

#region Orientation Exports
@export_group("Orientation")
## Manual 3D Euler orientation (stored in radians, edited in degrees) in the
## driver's reference frame. Orientation is strictly manual and never altered by velocity.
@export_custom(PROPERTY_HINT_RANGE, "-180.0,180.0,0.1,radians_as_degrees")
var manual_rotation: Vector3 = Vector3.ZERO:
	set(value):
		var wrapped: Vector3 = Vector3(
			0.0 if is_locked_to_ecliptic() else JrReferenceDirection._wrap_angle(value.x),
			JrReferenceDirection._wrap_angle(value.y),
			0.0 if is_locked_to_ecliptic() else JrReferenceDirection._wrap_angle(value.z)
		);
		if not manual_rotation.is_equal_approx(wrapped):
			manual_rotation = wrapped;
			if not _syncing_orientation:
				_manual_basis = Basis.from_euler(manual_rotation, manual_euler_order).orthonormalized();
				_on_orientation_changed();

## Euler axis order used when constructing the manual 3D orientation from [member manual_rotation].
@export
var manual_euler_order: EulerOrder = EULER_ORDER_YXZ:
	set(value):
		if manual_euler_order != value:
			manual_euler_order = value;
			if not _syncing_orientation:
				_manual_basis = Basis.from_euler(manual_rotation, manual_euler_order).orthonormalized();
				_on_orientation_changed();
#endregion

#region Propulsion Exports
@export_group("Propulsion")
## Continuous non-gravitational thrust acceleration vector integrated alongside
## gravity each simulation step. Interpreted in [member thrust_frame] and scaled
## by [member acceleration_unit].
@export
var external_acceleration: Vector3 = Vector3.ZERO:
	set(value):
		var target_acc: Vector3 = Vector3(value.x, 0.0, value.z) if is_locked_to_ecliptic() else value;
		if not external_acceleration.is_equal_approx(target_acc):
			external_acceleration = target_acc;
			if external_acceleration != Vector3.ZERO:
				_is_on_rails = false;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();

## Physical unit used to scale [member external_acceleration] into driver units.
@export
var acceleration_unit: JrOrreryUnits.AccelerationUnit = JrOrreryUnits.AccelerationUnit.METERS_PER_SECOND_SQUARED:
	set(value):
		if acceleration_unit != value:
			acceleration_unit = value;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();

## Coordinate frame in which [member external_acceleration] is applied.
## Use [constant ThrustFrame.ORBITAL] for continuous prograde ([code]-Z[/code]),
## retrograde ([code]+Z[/code]), normal ([code]+Y[/code]), or radial ([code]+X[/code]) burns.
@export
var thrust_frame: ThrustFrame = ThrustFrame.INERTIAL:
	set(value):
		if thrust_frame != value:
			thrust_frame = value;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();
#endregion

#region Gravity & Anchor Exports
@export_group("Gravity & Anchor")
## If [code]true[/code], third-body differential tidal forces (Battin's f(q)) that
## exceed [member JrOrbitDriver.tidal_cull_threshold] are integrated alongside the
## dominant anchor's two-body Keplerian drift (enabling Lagrange points and multi-body
## transfers). If [code]false[/code], only the dominant anchor's gravity is applied.
@export
var enable_tidal_forces: bool = true:
	set(value):
		if enable_tidal_forces != value:
			enable_tidal_forces = value;
			if not enable_tidal_forces and external_acceleration == Vector3.ZERO:
				_forecast_dirty = true;
			else:
				_is_on_rails = false;
				_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();

## Optional [JrOrreryElement] to permanently lock as this body's reference anchor,
## bypassing automatic Hill-sphere anchor switching. Leave [code]null[/code] for automatic switching.
@export
var pinned_anchor: JrOrreryElement = null:
	set(value):
		var target: JrOrreryElement = null if value == self else value;
		if pinned_anchor != target:
			pinned_anchor = target;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();
#endregion

#region Forecast & Editor Preview Exports
@export_group("Forecast & Editor Preview")
## If [code]true[/code], [JrOrbitDriver] continuously integrates a forward-looking
## trajectory forecast ahead of this body at runtime using the GPU compute shader.
@export
var runtime_forecast_enabled: bool = true:
	set(value):
		if runtime_forecast_enabled != value:
			runtime_forecast_enabled = value;
			_forecast_dirty = true;

## Duration (in simulation time units) integrated forward from the current time
## when computing both the live runtime trajectory forecast and the editor preview line.
@export_range(0.1, 10000000.0, 0.1, "exp", "or_greater")
var preview_duration: float = 30.0:
	set(value):
		var clamped: float = maxf(MIN_PREVIEW_DURATION, value);
		if not is_equal_approx(preview_duration, clamped):
			preview_duration = clamped;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();

## Number of output segments sampled across [member preview_duration] for both
## the live runtime trajectory forecast and the editor trajectory preview line.
@export_range(8, 2048, 1)
var preview_steps: int = 192:
	set(value):
		var clamped: int = maxi(8, value);
		if preview_steps != clamped:
			preview_steps = clamped;
			_forecast_dirty = true;
			if Engine.is_editor_hint():
				orbit_visuals_changed.emit();
#endregion

#region Lifecycle & Validation
## Initializes manual orientation on construction.
func _init() -> void:
	_manual_basis = Basis.from_euler(manual_rotation, manual_euler_order).orthonormalized();

## Initializes subtree tracking and evaluates fallback epoch transforms.
func _ready() -> void:
	super._ready();
	if get_parent() is JrOrreryElement and _spawn_parent == null:
		_spawn_parent = get_parent() as JrOrreryElement;
	if not _has_initialized_state:
		_apply_standalone_epoch_state();

## Refreshes editor configuration warnings when parented or unparented, and frees
## the internal non-tree helper on deletion.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if _conic_helper != null and is_instance_valid(_conic_helper):
			_conic_helper.free();
			_conic_helper = null;
	elif what == NOTIFICATION_PARENTED or what == NOTIFICATION_UNPARENTED:
		if not _suppress_reinit_on_reparent:
			_needs_parent_reinit = true;
			if what == NOTIFICATION_PARENTED and get_parent() is JrOrreryElement:
				_spawn_parent = get_parent() as JrOrreryElement;
		update_configuration_warnings();

## Validates that this [JrBallisticOrbiter] is parented under a [JrOrreryElement] so it
## is stepped by an owning [JrOrbitDriver].
func _get_configuration_warnings() -> PackedStringArray:
	var warnings: PackedStringArray = [];
	if get_parent() is not JrOrreryElement:
		warnings.append("A JrBallisticOrbiter must be a child of a JrOrbitDriver or JrRailOrbiter to be stepped automatically.");
	return warnings;

## Overrides [method JrOrreryElement.is_trajectory_valid] to enable path and transform
## synchronization on [JrOrbiter3D] and [JrOrbiter2D] followers.
func is_trajectory_valid() -> bool:
	return true;

## Overrides [method JrOrreryElement.has_keplerian_gizmo] to enable the 2D and 3D conic
## grabbers when [member initial_state_mode] is [constant InitialStateMode.KEPLERIAN_ELEMENTS].
func has_keplerian_gizmo() -> bool:
	return initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS and is_orbit_valid();

## Invoked when the owning [JrOrbitDriver] updates its distance or time units.
func _on_driver_units_updated() -> void:
	_forecast_dirty = true;
#endregion

#region Keplerian <-> State Vector Synchronization
## Lazily creates and configures the internal [member _conic_helper].
func _ensure_conic_helper() -> JrRailOrbiter:
	if _keplerian_cache_dirty and not _syncing_initial_state:
		_sync_keplerian_from_vectors();
	if _conic_helper == null or not is_instance_valid(_conic_helper):
		_conic_helper = JrRailOrbiter.new();
		if _reference_plane != null:
			_conic_helper._set_reference_plane(_reference_plane);
		_conic_helper.periapsis = initial_periapsis;
		_conic_helper.eccentricity = initial_eccentricity;
		_conic_helper.inclination = initial_inclination;
		_conic_helper.longitude_of_the_ascending_node = initial_longitude_of_the_ascending_node;
		_conic_helper.argument_of_periapsis = initial_argument_of_periapsis;
		_conic_helper.mean_anomaly_at_epoch = initial_mean_anomaly_at_epoch;
		var effective_mu: float = _parent_effective_mu if _parent_effective_mu > 0.0 else 160.0;
		_conic_helper.mean_motion = maxf(
			1e-12,
			JrRailOrbiter.compute_keplerian_mean_motion(initial_periapsis, initial_eccentricity, effective_mu)
		);
	return _conic_helper;

## Called downward by [JrOrbitDriver] to provide the effective central gravitational
## parameter [param parent_mu] of this body's spawn parent or anchor.
func set_parent_effective_mu(parent_mu: float) -> void:
	var clamped: float = maxf(0.0, parent_mu);
	if _parent_effective_mu == clamped:
		return;
	_parent_effective_mu = clamped;
	if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		_sync_vectors_from_keplerian();
		if Engine.is_editor_hint() or not _has_initialized_state:
			_needs_parent_reinit = true;
			_apply_standalone_epoch_state();
	else:
		_mark_or_sync_keplerian_from_vectors();

## Returns the effective gravitational parameter [member _parent_effective_mu] of the spawn parent.
func get_parent_effective_mu() -> float:
	return _parent_effective_mu;

## Synchronizes Keplerian properties immediately in the editor or when in Keplerian mode,
## or marks them dirty for lazy evaluation at runtime when in State Vectors mode.
func _mark_or_sync_keplerian_from_vectors() -> void:
	if Engine.is_editor_hint() or initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS or _conic_helper != null:
		_sync_keplerian_from_vectors();
	else:
		_keplerian_cache_dirty = true;

## Converts [member initial_position] and [member initial_velocity] into the
## Keplerian initial orbit properties using [member _parent_effective_mu].
func _sync_keplerian_from_vectors() -> void:
	if _syncing_initial_state:
		return;
	_keplerian_cache_dirty = false;
	var effective_mu: float = _parent_effective_mu if _parent_effective_mu > 0.0 else 160.0;

	var cw: bool = _reference_plane.positive_is_clockwise if _reference_plane != null else true;
	var conic: Dictionary[StringName, float] = JrRailOrbiter.compute_conic_from_state_vectors(
		initial_position,
		initial_velocity,
		effective_mu,
		cw
	);

	_syncing_initial_state = true;
	initial_periapsis = conic.get(&"periapsis", 10.0);
	initial_eccentricity = conic.get(&"eccentricity", 0.0);
	initial_inclination = conic.get(&"inclination", 0.0);
	initial_longitude_of_the_ascending_node = conic.get(&"longitude_of_the_ascending_node", 0.0);
	initial_argument_of_periapsis = conic.get(&"argument_of_periapsis", 0.0);
	initial_mean_anomaly_at_epoch = conic.get(&"mean_anomaly_at_epoch", 0.0);
	if _conic_helper != null and is_instance_valid(_conic_helper):
		_conic_helper.mean_motion = maxf(1e-12, conic.get(&"mean_motion", 1.0));
	_syncing_initial_state = false;

## Converts the Keplerian initial orbit properties into [member initial_position]
## and [member initial_velocity] using [member _parent_effective_mu].
func _sync_vectors_from_keplerian() -> void:
	if _syncing_initial_state:
		return;
	var helper: JrRailOrbiter = _ensure_conic_helper();
	if not helper.is_orbit_valid():
		return;

	var effective_mu: float = _parent_effective_mu if _parent_effective_mu > 0.0 else 160.0;
	var derived_n: float = JrRailOrbiter.compute_keplerian_mean_motion(
		initial_periapsis,
		initial_eccentricity,
		effective_mu
	);
	helper.mean_motion = maxf(1e-12, derived_n);

	var canonical_ref: JrReferenceDirection = JrReferenceDirection.new(
		Vector3.ZERO,
		EULER_ORDER_YXZ,
		_reference_plane.positive_is_clockwise if _reference_plane != null else true
	);
	var new_pos: Vector3 = helper.get_relative_position(0.0, canonical_ref);
	var new_vel: Vector3 = helper.get_relative_velocity(0.0, canonical_ref);

	_syncing_initial_state = true;
	initial_position = new_pos;
	initial_velocity = new_vel;
	_syncing_initial_state = false;
#endregion

#region Virtual Keplerian Gizmo Overrides
## Overrides [method JrOrreryElement.is_orbit_valid] for Keplerian gizmo editing.
func is_orbit_valid() -> bool:
	return _ensure_conic_helper().is_orbit_valid();

## Overrides [method JrOrreryElement.is_bound_orbit] for Keplerian gizmo editing.
func is_bound_orbit() -> bool:
	return _ensure_conic_helper().is_bound_orbit();

## Overrides [method JrOrreryElement.is_unbound_orbit] for Keplerian gizmo editing.
func is_unbound_orbit() -> bool:
	return _ensure_conic_helper().is_unbound_orbit();

## Overrides [method JrOrreryElement.get_relative_position] for Keplerian gizmo editing at epoch.
func get_relative_position(time: float, reference: JrReferenceDirection = null) -> Vector3:
	if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
		return _ensure_conic_helper().get_relative_position(time, ref);
	if is_zero_approx(time):
		return get_focus_initial_position();
	return _focus_relative_position;

## Overrides [method JrOrreryElement.get_periapsis_relative_position].
func get_periapsis_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_periapsis_relative_position(ref);

## Overrides [method JrOrreryElement.get_apoapsis_relative_position].
func get_apoapsis_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_apoapsis_relative_position(ref);

## Overrides [method JrOrreryElement.get_semi_parameter_relative_position].
func get_semi_parameter_relative_position(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_semi_parameter_relative_position(ref);

## Overrides [method JrOrreryElement.get_orbital_plane_gizmo_radius].
func get_orbital_plane_gizmo_radius() -> float:
	return _ensure_conic_helper().get_orbital_plane_gizmo_radius();

## Overrides [method JrOrreryElement.get_ascending_node_gizmo_position].
func get_ascending_node_gizmo_position() -> Vector3:
	var helper: JrRailOrbiter = _ensure_conic_helper();
	return helper.get_orbital_plane_gizmo_radius() * helper.get_ascending_node_direction(_reference_plane);

## Overrides [method JrOrreryElement.get_inclination_gizmo_position].
func get_inclination_gizmo_position() -> Vector3:
	var helper: JrRailOrbiter = _ensure_conic_helper();
	return helper.get_orbital_plane_gizmo_radius() * helper.get_inclination_direction(_reference_plane);

## Overrides [method JrOrreryElement.get_nodal_ecliptic_basis].
func get_nodal_ecliptic_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_nodal_ecliptic_basis(ref);

## Overrides [method JrOrreryElement.get_inclined_nodal_basis].
func get_inclined_nodal_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_inclined_nodal_basis(ref);

## Overrides [method JrOrreryElement.get_orbital_plane_basis].
func get_orbital_plane_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_orbital_plane_basis(ref);

## Overrides [method JrOrreryElement.get_tilted_body_basis].
func get_tilted_body_basis(reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_tilted_body_basis(ref);

## Overrides [method JrOrreryElement.get_body_basis].
func get_body_basis(time: float, reference: JrReferenceDirection = null) -> Basis:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		return _ensure_conic_helper().get_body_basis(time, ref);
	return _compute_driver_basis();

## Overrides [method JrOrreryElement.get_axial_tilt_direction].
func get_axial_tilt_direction(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_axial_tilt_direction(ref);

## Overrides [method JrOrreryElement.get_axial_tilt_hinge_axis].
func get_axial_tilt_hinge_axis(reference: JrReferenceDirection = null) -> Vector3:
	var ref: JrReferenceDirection = reference if reference != null else _reference_plane;
	return _ensure_conic_helper().get_axial_tilt_hinge_axis(ref);

## Overrides [method JrOrreryElement.get_mean_anomaly_from_true_anomaly].
func get_mean_anomaly_from_true_anomaly(true_anomaly: float) -> float:
	return _ensure_conic_helper().get_mean_anomaly_from_true_anomaly(true_anomaly);

## Overrides [method JrOrreryElement.get_orbit_path_points].
func get_orbit_path_points(
	_reference: JrReferenceDirection = null,
	_point_count: int = 64,
	_close_loop: bool = true,
	_max_unbound_distance: float = 500.0
) -> PackedVector3Array:
	return get_trajectory_path_points(_point_count);
#endregion

#region Runtime State, Propulsion, Anchor & Orientation API
## Returns [code]true[/code] if this body's runtime state has been initialized
## and does not have a pending reset request.
func is_runtime_state_ready() -> bool:
	return _has_initialized_state and not _needs_parent_reinit;

## Returns [code]true[/code] if this body's runtime trajectory forecast needs recomputation.
func is_forecast_dirty() -> bool:
	return _forecast_dirty;

## Marks this body's runtime trajectory forecast as needing recomputation.
func mark_forecast_dirty() -> void:
	_forecast_dirty = true;

## Returns the current dominant gravitational anchor [JrOrreryElement].
func get_anchor_element() -> JrOrreryElement:
	return _anchor_element;

## Returns the original spawn parent [JrOrreryElement] recorded prior to reparenting.
func get_spawn_parent() -> JrOrreryElement:
	if _spawn_parent != null and is_instance_valid(_spawn_parent):
		return _spawn_parent;
	if get_parent() is JrOrreryElement:
		return get_parent() as JrOrreryElement;
	return null;

## Sets whether reparenting notifications should suppress automatic state resets
## (used during [method JrOrbitDriver.collect_ballistic_nodes]).
func set_suppress_reinit_on_reparent(suppress: bool) -> void:
	_suppress_reinit_on_reparent = suppress;

## Returns this body's current position offset relative to [member _anchor_element]
## in the driver's unrotated reference coordinate space.
func get_anchor_relative_position() -> Vector3:
	return _anchor_relative_position;

## Returns this body's current velocity vector relative to [member _anchor_element]
## in the driver's unrotated reference coordinate space.
func get_anchor_relative_velocity() -> Vector3:
	return _anchor_relative_velocity;

## Returns [code]true[/code] if this body is currently coasting in Tier-1 epoch-frozen on-rails mode.
func is_on_rails() -> bool:
	return _is_on_rails;

## Returns the frozen anchor-relative position at the start of Tier-1 coasting.
func get_epoch_anchor_position() -> Vector3:
	return _epoch_anchor_pos;

## Returns the frozen anchor-relative velocity at the start of Tier-1 coasting.
func get_epoch_anchor_velocity() -> Vector3:
	return _epoch_anchor_vel;

## Returns the 64-bit simulation timestamp at which Tier-1 coasting was frozen.
func get_epoch_time() -> float:
	return _epoch_time;

## Returns the 64-bit analytical orbital period of the frozen Tier-1 conic (or 0.0 if unbound).
func get_epoch_orbital_period() -> float:
	return _epoch_orbital_period;

## Returns the maximum safe distance from [member _anchor_element] for Tier-1 coasting.
func get_epoch_safe_max_radius() -> float:
	return _epoch_safe_max_radius;

## Returns the phase-wrapped elapsed time since [member _epoch_time] at [param current_time],
## wrapping in 64-bit float by [member _epoch_orbital_period] when bound.
func get_wrapped_epoch_elapsed_time(current_time: float) -> float:
	var dt_total: float = current_time - _epoch_time;
	if _epoch_orbital_period > 0.0:
		return fmod(dt_total, _epoch_orbital_period);
	return dt_total;

## Enters Tier-1 epoch-frozen on-rails mode at [param current_time] with [param anchor_mu]
## and analytical wake-up radius [param safe_max_radius].
func enter_on_rails_epoch(current_time: float, anchor_mu: float, safe_max_radius: float) -> void:
	_is_on_rails = true;
	_epoch_anchor_pos = _anchor_relative_position;
	_epoch_anchor_vel = _anchor_relative_velocity;
	_epoch_time = current_time;
	_epoch_safe_max_radius = safe_max_radius;
	_epoch_orbital_period = _compute_bound_orbital_period(_epoch_anchor_pos, _epoch_anchor_vel, anchor_mu);

## Exits Tier-1 epoch-frozen on-rails mode so subsequent steps integrate incrementally.
func exit_on_rails_epoch() -> void:
	_is_on_rails = false;

## Computes the 64-bit orbital period for a bound conic state [param r_vec], [param v_vec]
## around [param mu] (returns 0.0 if unbound or degenerate).
static func _compute_bound_orbital_period(r_vec: Vector3, v_vec: Vector3, mu: float) -> float:
	var r_len: float = r_vec.length();
	if r_len <= 1e-9 or mu <= 0.0:
		return 0.0;
	var alpha: float = (2.0 / r_len) - (v_vec.length_squared() / mu);
	if alpha <= 1e-12:
		return 0.0;
	var semi_major_a: float = 1.0 / alpha;
	var mean_n: float = sqrt(mu / (semi_major_a * semi_major_a * semi_major_a));
	if mean_n <= 0.0:
		return 0.0;
	return TAU / mean_n;

## Returns [member external_acceleration] scaled into driver units (dist_unit / time_unit^2).
## When [member thrust_frame] is [constant ThrustFrame.BODY_LOCAL], rotates the vector by
## this body's manual 3D orientation [Basis]. When [constant ThrustFrame.ORBITAL], returns
## the scaled local vector (which is rotated per-substep by the instantaneous RTN basis).
func get_scaled_external_acceleration() -> Vector3:
	var raw_ext: Vector3 = Vector3(external_acceleration.x, 0.0, external_acceleration.z) if is_locked_to_ecliptic() else external_acceleration;
	var scaled: Vector3 = JrOrreryUnits.convert_acceleration_to_driver_units(
		raw_ext,
		acceleration_unit,
		_driver_distance_unit,
		_driver_time_unit
	);
	if scaled == Vector3.ZERO:
		return Vector3.ZERO;
	if thrust_frame == ThrustFrame.BODY_LOCAL:
		var local_rot: Vector3 = _compute_driver_basis() * scaled;
		return _reference_plane.project_vector_to_ecliptic(local_rot) if is_locked_to_ecliptic() else local_rot;
	if thrust_frame == ThrustFrame.INERTIAL and is_locked_to_ecliptic():
		return _reference_plane.project_vector_to_ecliptic(scaled);
	return scaled;

## Computes the instantaneous orthonormal orbital (Radial / Normal / Prograde) [Basis]
## relative to [member _anchor_element] where [code]-Z[/code] ([constant Vector3.FORWARD])
## is Prograde, [code]+Y[/code] ([constant Vector3.UP]) is Orbit Normal, and [code]+X[/code]
## ([constant Vector3.RIGHT]) is Radial Outward.
func get_orbital_thrust_basis() -> Basis:
	return compute_orbital_rtn_basis(_anchor_relative_position, _anchor_relative_velocity);

## Static helper constructing the orthonormal orbital [Basis] ([code]-Z[/code] Prograde,
## [code]+Y[/code] Orbit Normal, [code]+X[/code] Radial Outward) from [param rel_pos] and [param rel_vel].
static func compute_orbital_rtn_basis(rel_pos: Vector3, rel_vel: Vector3) -> Basis:
	var v_len: float = rel_vel.length();
	var r_len: float = rel_pos.length();
	var prograde: Vector3 = (rel_vel / v_len) if v_len > 1e-8 else (
		(rel_pos / r_len) if r_len > 1e-8 else Vector3.FORWARD
	);
	var h_vec: Vector3 = rel_pos.cross(rel_vel);
	var h_len: float = h_vec.length();
	var orbit_normal: Vector3 = (h_vec / h_len) if h_len > 1e-8 else Vector3.UP;
	if absf(prograde.dot(orbit_normal)) > 0.999:
		var fallback_up: Vector3 = Vector3.RIGHT if absf(prograde.y) > 0.9 else Vector3.UP;
		orbit_normal = fallback_up.cross(prograde).normalized();
	var z_back: Vector3 = -prograde;
	var x_radial: Vector3 = orbit_normal.cross(z_back).normalized();
	var y_normal: Vector3 = z_back.cross(x_radial).normalized();
	return Basis(x_radial, y_normal, z_back);

## Returns the live forward-integrated trajectory forecast points in the active
## system frame (matches [method JrOrreryElement.get_system_transform] origin at index 0).
func get_forecast_points() -> PackedVector3Array:
	return _forecast_system_points;

## Returns the live forward-integrated trajectory forecast points relative to the
## active [member JrOrbitDriver.perspective_node]'s future origin without surface-spin rotation.
func get_forecast_unrotated_points() -> PackedVector3Array:
	return _forecast_unrotated_points;

## Returns the live forward-integrated trajectory forecast points in raw inertial driver space.
func get_forecast_inertial_points() -> PackedVector3Array:
	return _forecast_inertial_points;

## Returns the per-point reference gravity-well instance IDs corresponding to [method get_forecast_points].
## Consecutive points with differing IDs indicate a Sphere-of-Influence reference frame transition.
func get_forecast_reference_ids() -> PackedInt64Array:
	return _forecast_reference_ids;

## Projects [method get_forecast_unrotated_points] onto 2D top-down X-Z coordinates
## scaled by [param pos_scale].
func get_forecast_points_2d(pos_scale: float = 1.0) -> PackedVector2Array:
	var count: int = _forecast_unrotated_points.size();
	var pts_2d: PackedVector2Array = [];
	pts_2d.resize(count);
	for i: int in range(count):
		var p: Vector3 = _forecast_unrotated_points[i];
		pts_2d[i] = Vector2(p.x, p.z) * pos_scale;
	return pts_2d;

## Resets this body's state back to [member initial_position] and [member initial_velocity]
## relative to its spawn parent's orbital state, and schedules re-synchronization
## on the next driver pass.
func reset_state() -> void:
	if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		_sync_vectors_from_keplerian();
	var ref_basis: Basis = _get_reference_rotation_basis();
	_anchor_relative_position = ref_basis * initial_position;
	_anchor_relative_velocity = ref_basis * initial_velocity;
	_inertial_position = _last_parent_origin + _anchor_relative_position;
	_inertial_velocity = _last_parent_velocity + _anchor_relative_velocity;
	_is_on_rails = false;
	_needs_parent_reinit = true;
	_forecast_dirty = true;
	_rebuild_transforms_from_anchor();
	_system_transform = _driver_relative_transform;
	transform_updated.emit();

## Applies an instantaneous velocity change [param delta_v] (in inertial driver units).
func apply_impulse(delta_v: Vector3) -> void:
	var effective_dv: Vector3 = _reference_plane.project_vector_to_ecliptic(delta_v) if is_locked_to_ecliptic() else delta_v;
	_anchor_relative_velocity += effective_dv;
	_inertial_velocity += effective_dv;
	_global_velocity = _inertial_velocity;
	_relative_velocity = _anchor_to_parent_velocity + _anchor_relative_velocity;
	_is_on_rails = false;
	_forecast_dirty = true;

## Applies an instantaneous velocity change [param local_delta_v] expressed in this
## body's manual 3D local orientation frame (for example, [code]Vector3.FORWARD * speed[/code]).
func apply_local_impulse(local_delta_v: Vector3) -> void:
	apply_impulse(_compute_driver_basis() * local_delta_v);

## Applies an instantaneous velocity change [param orbital_delta_v] expressed in this
## body's instantaneous orbital frame ([code]Vector3.FORWARD[/code] = prograde,
## [code]Vector3.UP[/code] = normal, [code]Vector3.RIGHT[/code] = radial outward).
func apply_orbital_impulse(orbital_delta_v: Vector3) -> void:
	apply_impulse(get_orbital_thrust_basis() * orbital_delta_v);

## Returns this body's current inertial position in the unshifted [JrOrbitDriver] frame.
func get_driver_position() -> Vector3:
	return _inertial_position;

## Directly sets this body's inertial position in the unshifted [JrOrbitDriver] frame.
func set_driver_position(new_position: Vector3) -> void:
	_inertial_position = new_position;
	_anchor_relative_position = new_position - _last_anchor_origin;
	_has_initialized_state = true;
	_needs_parent_reinit = false;
	_is_on_rails = false;
	_forecast_dirty = true;
	_rebuild_transforms_from_anchor();

## Returns this body's current inertial velocity in the unshifted [JrOrbitDriver] frame.
func get_driver_velocity() -> Vector3:
	return _inertial_velocity;

## Directly sets this body's inertial velocity in the unshifted [JrOrbitDriver] frame.
func set_driver_velocity(new_velocity: Vector3) -> void:
	_inertial_velocity = new_velocity;
	_anchor_relative_velocity = new_velocity - _last_anchor_velocity;
	_has_initialized_state = true;
	_needs_parent_reinit = false;
	_is_on_rails = false;
	_forecast_dirty = true;
	_global_velocity = _inertial_velocity;
	_relative_velocity = _anchor_to_parent_velocity + _anchor_relative_velocity;

## Returns the manual local 3D orientation [Basis] prior to reference-frame rotation.
func get_manual_basis() -> Basis:
	return _manual_basis;

## Directly sets the manual local 3D orientation [Basis] (orthonormalized) and
## synchronizes [member manual_rotation] without altering velocity.
func set_manual_basis(new_basis: Basis) -> void:
	_manual_basis = new_basis.orthonormalized();
	_syncing_orientation = true;
	manual_rotation = _manual_basis.get_euler(manual_euler_order);
	_syncing_orientation = false;
	_on_orientation_changed();
#endregion

#region Driver Call-Down Integration Hooks
## Initializes this body's anchor-relative and inertial state from [param parent_transform],
## [param parent_velocity], [param reference_plane], and [param initial_anchor].
## Called downward by [JrOrbitDriver].
func initialize_from_parent(
	parent_transform: Transform3D,
	parent_velocity: Vector3,
	reference_plane: JrReferenceDirection,
	initial_anchor: JrOrreryElement = null
) -> void:
	_set_reference_plane(reference_plane);
	if initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		_sync_vectors_from_keplerian();
	else:
		_mark_or_sync_keplerian_from_vectors();

	_parent_driver_basis = parent_transform.basis;
	_last_parent_origin = parent_transform.origin;
	_last_parent_velocity = parent_velocity;
	_last_anchor_origin = parent_transform.origin;
	_last_anchor_velocity = parent_velocity;
	_anchor_to_parent_offset = Vector3.ZERO;
	_anchor_to_parent_velocity = Vector3.ZERO;

	var prev_anchor: JrOrreryElement = _anchor_element;
	_anchor_element = initial_anchor;
	if prev_anchor != _anchor_element and _anchor_element != null:
		dominant_body_changed.emit(prev_anchor, _anchor_element);

	var ref_basis: Basis = _get_reference_rotation_basis();
	_anchor_relative_position = ref_basis * initial_position;
	_anchor_relative_velocity = ref_basis * initial_velocity;
	_inertial_position = _last_anchor_origin + _anchor_relative_position;
	_inertial_velocity = _last_anchor_velocity + _anchor_relative_velocity;

	_is_on_rails = false;
	_has_initialized_state = true;
	_needs_parent_reinit = false;
	_forecast_dirty = true;
	_rebuild_transforms_from_anchor();

## Switches this body's dominant reference anchor to [param new_anchor] using the
## exact tree-relative inter-well offset [param old_to_new_offset] ([code]r_new - r_old[/code])
## and velocity difference [param old_to_new_velocity] ([code]v_new - v_old[/code]).
func switch_anchor_element(
	new_anchor: JrOrreryElement,
	old_to_new_offset: Vector3,
	old_to_new_velocity: Vector3
) -> void:
	if new_anchor == null or new_anchor == _anchor_element:
		return;
	var prev_anchor: JrOrreryElement = _anchor_element;
	_anchor_element = new_anchor;
	_anchor_relative_position -= old_to_new_offset;
	_anchor_relative_velocity -= old_to_new_velocity;
	_is_on_rails = false;
	_forecast_dirty = true;
	dominant_body_changed.emit(prev_anchor, _anchor_element);

## Stores the newly integrated anchor-relative position and velocity produced by
## [JrBallisticComputeSolver]. Called downward by [JrOrbitDriver].
func apply_integrated_anchor_state(new_anchor_rel_pos: Vector3, new_anchor_rel_vel: Vector3) -> void:
	if is_locked_to_ecliptic():
		_anchor_relative_position = _reference_plane.project_vector_to_ecliptic(new_anchor_rel_pos);
		_anchor_relative_velocity = _reference_plane.project_vector_to_ecliptic(new_anchor_rel_vel);
	else:
		_anchor_relative_position = new_anchor_rel_pos;
		_anchor_relative_velocity = new_anchor_rel_vel;
	_inertial_position = _last_anchor_origin + _anchor_relative_position;
	_inertial_velocity = _last_anchor_velocity + _anchor_relative_velocity;
	_forecast_dirty = true;

## Backwards-compatible hook storing newly integrated inertial position and velocity.
func apply_integrated_state(new_inertial_pos: Vector3, new_inertial_vel: Vector3) -> void:
	_inertial_position = new_inertial_pos;
	_inertial_velocity = new_inertial_vel;
	_anchor_relative_position = new_inertial_pos - _last_anchor_origin;
	_anchor_relative_velocity = new_inertial_vel - _last_anchor_velocity;
	_forecast_dirty = true;

## Stores the forward-integrated runtime forecast polylines computed by [JrOrbitDriver].
func set_runtime_forecast_points(
	inertial_pts: PackedVector3Array,
	unrotated_pts: PackedVector3Array,
	system_pts: PackedVector3Array,
	reference_ids: PackedInt64Array = PackedInt64Array()
) -> void:
	_forecast_inertial_points = inertial_pts;
	_forecast_unrotated_points = unrotated_pts;
	_forecast_system_points = system_pts;
	_forecast_reference_ids = reference_ids;
	_forecast_dirty = false;

## Synchronizes focus-relative, local parent-canceled, and heliocentric driver transforms
## using exact LCA offsets between [member _anchor_element] and the scene parent.
func sync_transforms_with_anchor_and_parent(
	parent_transform: Transform3D,
	parent_velocity: Vector3,
	anchor_to_parent_pos: Vector3,
	anchor_to_parent_vel: Vector3
) -> void:
	_parent_driver_basis = parent_transform.basis;
	_last_parent_origin = parent_transform.origin;
	_last_parent_velocity = parent_velocity;
	_anchor_to_parent_offset = anchor_to_parent_pos;
	_anchor_to_parent_velocity = anchor_to_parent_vel;
	_last_anchor_origin = _last_parent_origin + _anchor_to_parent_offset;
	_last_anchor_velocity = _last_parent_velocity + _anchor_to_parent_velocity;
	_rebuild_transforms_from_anchor();

## Recomputes focus-relative, local parent-canceled, and heliocentric driver transforms
## against the parent's updated [param parent_transform] and [param parent_velocity].
func sync_transforms_with_parent(
	parent_transform: Transform3D,
	parent_velocity: Vector3,
	reference_plane: JrReferenceDirection
) -> void:
	_set_reference_plane(reference_plane);
	_parent_driver_basis = parent_transform.basis;
	_last_parent_origin = parent_transform.origin;
	_last_parent_velocity = parent_velocity;
	if _anchor_element == null:
		_last_anchor_origin = _last_parent_origin;
		_last_anchor_velocity = _last_parent_velocity;
		_anchor_to_parent_offset = Vector3.ZERO;
		_anchor_to_parent_velocity = Vector3.ZERO;
	_rebuild_transforms_from_anchor();
#endregion

#region Editor Preview, Ghost & Gizmo Helpers
## Stores parent-focus-relative preview points [param focus_relative_pts] for
## editor path rendering and animated ghost playback.
func set_editor_preview_relative_points(focus_relative_pts: PackedVector3Array) -> void:
	_preview_path_points = focus_relative_pts;
	_refresh_ghost_state();

## Stores forward-integrated inertial preview points [param points_driver] converted
## relative to [param parent_epoch_origin] for editor path rendering and ghost playback.
func set_editor_preview_trajectory(
	points_driver: PackedVector3Array,
	parent_epoch_origin: Vector3
) -> void:
	_preview_path_points = [];
	_preview_path_points.resize(points_driver.size());
	for i: int in range(points_driver.size()):
		_preview_path_points[i] = points_driver[i] - parent_epoch_origin;
	_refresh_ghost_state();

## Updates the animated editor preview ghost along [member _preview_path_points]
## at [param preview_time] and emits [signal JrOrreryElement.editor_ghost_updated].
func update_editor_ghost(
	preview_time: float,
	reference_plane: JrReferenceDirection,
	parent_driver_basis: Basis = Basis.IDENTITY
) -> void:
	_last_preview_time = preview_time;
	_parent_driver_basis = parent_driver_basis;
	_set_reference_plane(reference_plane);
	_refresh_ghost_state();
	editor_ghost_updated.emit(preview_time);

## Overrides [method JrOrreryElement.has_ghost_position].
func has_ghost_position() -> bool:
	return _has_ghost_position;

## Overrides [method JrOrreryElement.get_ghost_relative_position].
func get_ghost_relative_position() -> Vector3:
	return _ghost_relative_position;

## Overrides [method JrOrreryElement.get_ghost_body_basis].
func get_ghost_body_basis() -> Basis:
	return _ghost_body_basis;

## Overrides [method JrOrreryElement.get_trajectory_path_points] to return the
## forward-integrated preview polyline relative to the parent focus.
func get_trajectory_path_points(_point_count: int = 64) -> PackedVector3Array:
	if _preview_path_points.is_empty() and initial_state_mode == InitialStateMode.KEPLERIAN_ELEMENTS:
		return _ensure_conic_helper().get_orbit_path_points(_reference_plane, _point_count);
	return _preview_path_points;

## Returns [member initial_position] transformed into the active driver-focus space.
func get_focus_initial_position() -> Vector3:
	return _get_reference_rotation_basis() * initial_position;

## Sets [member initial_position] from a vector expressed in driver-focus space.
func set_focus_initial_position(focus_pos: Vector3) -> void:
	initial_position = _get_reference_rotation_basis().transposed() * focus_pos;

## Returns [member initial_velocity] transformed into the active driver-focus space.
func get_focus_initial_velocity() -> Vector3:
	return _get_reference_rotation_basis() * initial_velocity;

## Sets [member initial_velocity] from a vector expressed in driver-focus space.
func set_focus_initial_velocity(focus_vel: Vector3) -> void:
	initial_velocity = _get_reference_rotation_basis().transposed() * focus_vel;

## Returns the 3D position of the velocity handle tip in unscaled driver-focus space.
func get_velocity_gizmo_focus_position() -> Vector3:
	return get_focus_initial_position() + get_focus_initial_velocity() * velocity_gizmo_scale;
#endregion

#region Internal Transform & Signal Helpers
## Updates cached [member _reference_plane] and connects its change signal.
func _set_reference_plane(new_reference_plane: JrReferenceDirection) -> void:
	if new_reference_plane == null:
		return;
	if _reference_plane != new_reference_plane:
		if _reference_plane != null and _reference_plane.changed.is_connected(_on_reference_plane_changed):
			_reference_plane.changed.disconnect(_on_reference_plane_changed);
		_reference_plane = new_reference_plane;
		if _conic_helper != null and is_instance_valid(_conic_helper):
			_conic_helper._set_reference_plane(new_reference_plane);
		if not _reference_plane.changed.is_connected(_on_reference_plane_changed):
			_reference_plane.changed.connect(_on_reference_plane_changed);
	if is_locked_to_ecliptic():
		_enforce_ecliptic_lock();

## Handles orientation updates on the shared [JrReferenceDirection].
func _on_reference_plane_changed() -> void:
	if is_locked_to_ecliptic():
		_enforce_ecliptic_lock();
	if Engine.is_editor_hint() or not _has_initialized_state:
		_apply_standalone_epoch_state();
	else:
		_rebuild_transforms_from_anchor();
	orbit_visuals_changed.emit();

## Clamps initial position, velocity, inclination, thrust, manual orientation, and
## active anchor-relative vectors onto the ecliptic plane when [method is_locked_to_ecliptic] is active.
func _enforce_ecliptic_lock() -> void:
	if _conic_helper != null and is_instance_valid(_conic_helper):
		_conic_helper._set_reference_plane(_reference_plane);
	if not is_zero_approx(initial_position.y):
		initial_position = Vector3(initial_position.x, 0.0, initial_position.z);
	if not is_zero_approx(initial_velocity.y):
		initial_velocity = Vector3(initial_velocity.x, 0.0, initial_velocity.z);
	var target_inc: float = PI if absf(initial_inclination) > PI * 0.5 else 0.0;
	if not is_equal_approx(initial_inclination, target_inc):
		initial_inclination = target_inc;
	if not is_zero_approx(external_acceleration.y):
		external_acceleration = Vector3(external_acceleration.x, 0.0, external_acceleration.z);
	if not is_zero_approx(manual_rotation.x) or not is_zero_approx(manual_rotation.z):
		manual_rotation = Vector3(0.0, manual_rotation.y, 0.0);
	_anchor_relative_position = _reference_plane.project_vector_to_ecliptic(_anchor_relative_position);
	_anchor_relative_velocity = _reference_plane.project_vector_to_ecliptic(_anchor_relative_velocity);
	_forecast_dirty = true;

## Returns the pure right-handed rotation [Basis] of [member _reference_plane].
func _get_reference_rotation_basis() -> Basis:
	if _reference_plane != null:
		return _reference_plane.rotation_basis;
	return Basis.IDENTITY;

## Returns the body's manual orientation [Basis] in the driver's reference space.
func _compute_driver_basis() -> Basis:
	return (_get_reference_rotation_basis() * _manual_basis).orthonormalized();

## Recomputes all [JrOrreryElement] cached positions, velocities, and transforms
## directly from [member _anchor_relative_position] and [member _anchor_to_parent_offset]
## without subtracting large heliocentric coordinates.
func _rebuild_transforms_from_anchor() -> void:
	_focus_relative_position = _anchor_to_parent_offset + _anchor_relative_position;
	_relative_velocity = _anchor_to_parent_velocity + _anchor_relative_velocity;
	_inertial_position = _last_anchor_origin + _anchor_relative_position;
	_inertial_velocity = _last_anchor_velocity + _anchor_relative_velocity;
	_global_velocity = _inertial_velocity;
	_driver_relative_basis = _compute_driver_basis();

	var inv_parent: Basis = _parent_driver_basis.transposed();
	var local_pos: Vector3 = inv_parent * _focus_relative_position;
	var local_basis: Basis = (inv_parent * _driver_relative_basis).orthonormalized();
	_local_orbital_transform = Transform3D(local_basis, local_pos);
	_driver_relative_transform = Transform3D(_driver_relative_basis, _inertial_position);

## Evaluates standalone/editor epoch state when edited in the Inspector or gizmos.
func _on_initial_parameters_changed() -> void:
	if not is_inside_tree():
		_needs_parent_reinit = true;
		return;
	if Engine.is_editor_hint() or not _has_initialized_state:
		_needs_parent_reinit = true;
		_apply_standalone_epoch_state();
	orbit_visuals_changed.emit();

## Updates cached transforms and notifies followers when manual orientation changes.
func _on_orientation_changed() -> void:
	_rebuild_transforms_from_anchor();
	if Engine.is_editor_hint() or not _simulation_playing:
		_system_transform = _driver_relative_transform;
	_refresh_ghost_state();
	transform_updated.emit();
	orbit_visuals_changed.emit();

## Computes epoch transforms from cached parent state prior to a full driver pass.
func _apply_standalone_epoch_state() -> void:
	var ref_basis: Basis = _get_reference_rotation_basis();
	_anchor_relative_position = ref_basis * initial_position;
	_anchor_relative_velocity = ref_basis * initial_velocity;
	_anchor_to_parent_offset = Vector3.ZERO;
	_anchor_to_parent_velocity = Vector3.ZERO;
	_last_anchor_origin = _last_parent_origin;
	_last_anchor_velocity = _last_parent_velocity;
	_rebuild_transforms_from_anchor();
	_system_transform = _driver_relative_transform;
	_refresh_ghost_state();
	transform_updated.emit();

## Linearly interpolates [member _ghost_relative_position] along [member _preview_path_points].
func _refresh_ghost_state() -> void:
	var count: int = _preview_path_points.size();
	if count == 0:
		_has_ghost_position = false;
		return;

	_ghost_body_basis = _compute_driver_basis();
	if count == 1 or preview_duration <= 0.0:
		_ghost_relative_position = _preview_path_points[0];
		_has_ghost_position = true;
		return;

	var wrapped_time: float = fposmod(_last_preview_time, maxf(MIN_PREVIEW_DURATION, preview_duration));
	var normalized_t: float = clampf(wrapped_time / maxf(MIN_PREVIEW_DURATION, preview_duration), 0.0, 1.0);
	var scaled_idx: float = normalized_t * float(count - 1);
	var idx_a: int = mini(int(floorf(scaled_idx)), count - 1);
	var idx_b: int = mini(idx_a + 1, count - 1);
	var frac: float = scaled_idx - float(idx_a);
	_ghost_relative_position = _preview_path_points[idx_a].lerp(_preview_path_points[idx_b], frac);
	_has_ghost_position = true;
#endregion
