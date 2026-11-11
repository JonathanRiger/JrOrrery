@tool
@icon("res://addons/jr_orrery/icons/orbit_driver.svg")
class_name JrOrbitDriver
extends JrOrreryElement
## Top-level simulation [Node] controller for a hierarchical orbital and ballistic system.
##
## Generates and maintains a [JrReferenceDirection] ([RefCounted]) from its
## exported reference frame properties, defines the physical unit system
## ([JrOrreryUnits]), tracks all descendant [JrRailOrbiter] and
## [JrBallisticOrbiter] nodes in its subtree, evaluates analytical Keplerian orbits,
## and dispatches the Hybrid Universal-Variable Kick-Drift-Kick GPU compute shader
## ([JrBallisticComputeSolver]) with LCA tree-relative coordinates and Hill-sphere
## anchor switching.

## Controls how the system's 3D orientation is locked when [member perspective_node] is assigned.
enum PerspectiveRotationMode {
	## Locks only the position of [member perspective_node] at the origin; coordinate axes remain
	## inertial.
	INERTIAL = 0,
	## Locks both position and axial spin orientation of [member perspective_node] to the origin.
	BODY_SPIN = 1,
	## Locks position and co-rotates with the orbital radius vector from
	## [member perspective_node]'s
	## parent focus so Lagrange points (L1-L5) remain stationary.
	SYNODIC_ORBITAL = 2,
}

## Hysteresis multiplier on a child body's Hill radius for entering its gravity well.
const HILL_ENTRY_MULTIPLIER: float = 0.98

## Hysteresis multiplier on the current anchor's Hill radius for exiting to its parent well.
const HILL_EXIT_MULTIPLIER: float = 1.05

@export_group("Reference Frame")
## Euler rotation of the reference frame from the default orientation
## ([constant Vector3.FORWARD] vernal point, [constant Vector3.UP] ecliptic north pole),
## stored in radians (edited in degrees in the Inspector) and normalized to the
## range [code](-PI, PI][/code] per component.
@export_custom(PROPERTY_HINT_RANGE, "-180.0,180.0,0.1,radians_as_degrees")
var reference_rotation := Vector3.ZERO:
	set(value):
		var wrapped := Vector3(
				JrReferenceDirection.wrap_angle(value.x),
				JrReferenceDirection.wrap_angle(value.y),
				JrReferenceDirection.wrap_angle(value.z)
		)
		if not reference_rotation.is_equal_approx(wrapped):
			reference_rotation = wrapped
			_sync_reference_direction()

## Euler axis rotation order used when constructing the 3D orientation from
## [member reference_rotation].
@export var euler_order: EulerOrder = EULER_ORDER_YXZ:
	set(value):
		if euler_order != value:
			euler_order = value
			_sync_reference_direction()

## Defines the direction of positive angular progression in the ecliptic plane
## when viewed from the ecliptic north pole ([code]+Y[/code]).
## If [code]true[/code], positive angular progression is clockwise.
## If [code]false[/code], positive angular progression is counter-clockwise.
@export var positive_is_clockwise: bool = true:
	set(value):
		if positive_is_clockwise != value:
			positive_is_clockwise = value
			_sync_reference_direction()

## If [code]true[/code], forces all descendant [JrRailOrbiter] and [JrBallisticOrbiter]
## orbits, ballistic trajectories, thrust vectors, and axial tilts to lie flat on the
## ecliptic plane (allowing both flat prograde [code]0 deg[/code] and flat retrograde
## [code]180 deg[/code] motion), and hides out-of-plane 3D gizmo handles.
@export var lock_to_ecliptic: bool = false:
	set(value):
		if lock_to_ecliptic != value:
			lock_to_ecliptic = value
			_sync_reference_direction()
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree():
				update_orbits()
			orbit_visuals_changed.emit()

@export_group("Simulation Units")
## Physical length unit represented by [code]1.0[/code] spatial unit across this driver's
## hierarchy.
@export var distance_unit := JrOrreryUnits.DistanceUnit.METERS:
	set(value):
		if distance_unit != value:
			distance_unit = value;
			_units_and_mu_dirty = true;
			_propagate_units_and_parent_mu();
			_invalidate_all_ballistic_forecasts();
			if is_inside_tree():
				update_orbits();
			orbit_visuals_changed.emit();

## Physical time unit represented by [code]1.0[/code] unit of [member time] across this driver's
## hierarchy.
@export var time_unit := JrOrreryUnits.TimeUnit.SECONDS:
	set(value):
		if time_unit != value:
			time_unit = value;
			_units_and_mu_dirty = true;
			_propagate_units_and_parent_mu();
			_invalidate_all_ballistic_forecasts();
			if is_inside_tree():
				update_orbits();
			orbit_visuals_changed.emit();

@export_group("Gravity")
## Physical mass of the root focus at the origin of this [JrOrbitDriver] (for example,
## a central star, or [code]0.0[/code] for a massless binary barycenter), expressed in
## [member mass_unit].
@export_range(0.0, 1_000_000.0, 0.001, "or_greater")
var mass: float = 0.0:
	set(value):
		var clamped: float = maxf(0.0, value);
		if mass != clamped:
			mass = clamped;
			_units_and_mu_dirty = true;
			_propagate_units_and_parent_mu();
			_invalidate_all_ballistic_forecasts();
			if is_inside_tree() and Engine.is_editor_hint():
				update_orbits();
			orbit_visuals_changed.emit();

## Unit in which [member mass] is specified. Combined with [member distance_unit] and
## [member time_unit] to compute the root gravitational parameter [code]mu = G * M[/code].
@export var mass_unit := JrOrreryUnits.MassUnit.DRIVER_GM:
	set(value):
		if mass_unit != value:
			mass_unit = value;
			_units_and_mu_dirty = true;
			_propagate_units_and_parent_mu();
			_invalidate_all_ballistic_forecasts();
			if is_inside_tree() and Engine.is_editor_hint():
				update_orbits();
			orbit_visuals_changed.emit();

## Dimensionless tidal ratio threshold [code](|a_tidal| / |a_dominant|)[/code] below which
## distant third-body gravitational perturbations are culled. Bodies with zero thrust and
## zero unculled tides automatically coast on Tier-1 frozen Keplerian conics.
@export_range(0.0, 0.1, 0.00001)
var tidal_cull_threshold: float = 0.0001:
	set(value):
		var clamped: float = maxf(0.0, value)
		if tidal_cull_threshold != clamped:
			tidal_cull_threshold = clamped
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree() and Engine.is_editor_hint():
				update_orbits()

## Maximum number of adaptive Universal-Variable Kick-Drift-Kick substeps executed inside
## each GPU compute shader dispatch per simulation tick.
@export_range(1, 64, 1)
var ballistic_substeps: int = 1:
	set(value):
		ballistic_substeps = maxi(1, value)

## If [code]true[/code], continuously dispatches the GPU trajectory forecast compute
## shader at runtime to predict future paths for [JrBallisticOrbiter] bodies relative to
## the active [member perspective_node].
@export var enable_ballistic_forecasts: bool = true:
	set(value):
		if enable_ballistic_forecasts != value:
			enable_ballistic_forecasts = value
			if is_inside_tree() and not Engine.is_editor_hint():
				update_orbits()

@export_group("Perspective")
## Optional descendant [JrOrreryElement] ([JrRailOrbiter] or [JrBallisticOrbiter]) to lock
## at the system origin at runtime. When [code]null[/code], the driver itself
## remains at the system origin.
@export var perspective_node: JrOrreryElement = null:
	set(value):
		if perspective_node != value:
			perspective_node = value
			if is_inside_tree():
				update_orbits()

## Controls whether the system frame stays inertial ([constant PerspectiveRotationMode.INERTIAL]),
## locks to [member perspective_node]'s axial spin ([constant PerspectiveRotationMode.BODY_SPIN]),
## or co-rotates with [member perspective_node]'s orbital radius vector so Lagrange points L1-L5
## remain fixed ([constant PerspectiveRotationMode.SYNODIC_ORBITAL]).
@export var perspective_rotation_mode := PerspectiveRotationMode.BODY_SPIN:
	set(value):
		if perspective_rotation_mode != value:
			perspective_rotation_mode = value
			if not _syncing_perspective_mode:
				_syncing_perspective_mode = true
				lock_perspective_rotation = (value != PerspectiveRotationMode.INERTIAL)
				_syncing_perspective_mode = false
			if is_inside_tree():
				update_orbits()

## Backwards-compatible boolean toggle for locking [member perspective_node]'s orientation.
## Setting [code]false[/code] selects [constant PerspectiveRotationMode.INERTIAL]; setting
## [code]true[/code] selects [constant PerspectiveRotationMode.BODY_SPIN] if currently inertial.
@export var lock_perspective_rotation: bool = true:
	set(value):
		if lock_perspective_rotation != value:
			lock_perspective_rotation = value
			if not _syncing_perspective_mode:
				_syncing_perspective_mode = true
				if not value:
					perspective_rotation_mode = PerspectiveRotationMode.INERTIAL
				elif perspective_rotation_mode == PerspectiveRotationMode.INERTIAL:
					perspective_rotation_mode = PerspectiveRotationMode.BODY_SPIN
				_syncing_perspective_mode = false
			if is_inside_tree():
				update_orbits()

@export_group("Simulation Time")
## Current simulation time passed to all descendant [JrRailOrbiter] and [JrBallisticOrbiter]
## solvers.
@export var time: float = 0.0:
	set(value):
		if time != value:
			time = value
			if is_inside_tree():
				update_orbits()

## Multiplier applied to frame delta when advancing [member time] automatically.
@export var time_scale: float = 1.0

## Multiplier applied to frame delta when advancing the editor preview clock.
## Set to [code]0.0[/code] to stop editor ghost playback.
@export var editor_play_speed: float = 1.0

## If [code]true[/code], [member time] advances automatically each frame at runtime.
@export var auto_advance: bool = true:
	set(value):
		if auto_advance != value:
			auto_advance = value
			_sync_simulation_playing_state()

## If [code]true[/code], advances orbits during [method Node._physics_process]
## instead of [method Node._process].
@export var use_physics_process: bool = false

## Read-only access to the generated [JrReferenceDirection] ([RefCounted]) instance
## passed down to all descendant [JrRailOrbiter] and [JrBallisticOrbiter] solvers.
var reference_direction: JrReferenceDirection:
	get:
		return _reference_plane

## Non-exported backwards-compatible alias for the computed gravitational parameter
## [code]mu = G * M[/code] in driver units ([code]distance_unit^3 / time_unit^2[/code]).
var gravity_strength: float:
	get:
		_recompute_gravity_strength_from_mass()
		return _gravity_strength
	set(value):
		var clamped: float = maxf(0.0, value)
		_gravity_strength = clamped
		mass = JrOrreryUnits.mu_to_mass(clamped, mass_unit, distance_unit, time_unit)

## Maps each tracked descendant [JrRailOrbiter] to its immediate parent [JrOrreryElement].
var _children: Dictionary[JrRailOrbiter, JrOrreryElement] = {}

## Maps every tracked descendant simulation element ([JrRailOrbiter] or [JrBallisticOrbiter])
## to its immediate scene-tree parent [JrOrreryElement].
var _element_parents: Dictionary[JrOrreryElement, JrOrreryElement] = {}

## Flattened list of all tracked descendant [JrRailOrbiter] nodes in parent-before-child order.
var _orbit_sequence: Array[JrRailOrbiter] = []

## Flattened list of all tracked descendant [JrBallisticOrbiter] nodes in parent-before-child
## order.
var _ballistic_sequence: Array[JrBallisticOrbiter] = [];

## Sparse list of [JrBallisticOrbiter] nodes not currently in the root-parent/root-anchor fast sync state.
var _non_root_sync_ballistics: Array[JrBallisticOrbiter] = [];

## Sparse list of tracked [JrBallisticOrbiter] nodes with [member JrBallisticOrbiter.runtime_forecast_enabled] set.
var _forecast_enabled_ballistics: Array[JrBallisticOrbiter] = [];

## Last broadcast value of [member JrBallisticOrbiter._direct_system_xf] across tracked ballistics.
var _last_direct_system_xf_state: bool = true;

## Batched GPU compute-shader N-body solver for [JrBallisticOrbiter] descendants.
var _ballistic_solver := JrBallisticComputeSolver.new()

## Last runtime [member time] value at which ballistic integration was stepped.
var _last_stepped_time: float = 0.0

## Whether initial runtime orbital state has been evaluated at least once.
var _runtime_initialized: bool = false

## Non-serialized preview timestamp used to animate editor ghost markers without
## modifying actual node transforms in the scene file.
var _editor_preview_time: float = 0.0

## Re-entrancy guard set while [method update_orbits] is actively evaluating descendants.
var _updating_orbits: bool = false

## Nestable batch-modification counter suppressing intermediate [method update_orbits] calls
## during bulk node additions, removals, or reparenting.
var _batch_update_depth: int = 0

## Re-entrancy guard for synchronizing [member perspective_rotation_mode] and
## [member lock_perspective_rotation].
var _syncing_perspective_mode: bool = false

## Cached ordered gravity-well elements from the most recent forecast pass.
var _last_forecast_well_list: Array[JrOrreryElement] = []

## Cached anchor-relative forecast points (`xyz = r_rel, w = anchor_idx`) from the most recent
## forecast pass.
var _last_forecast_anchor_points: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {}

## Optional telemetry sink receiving microsecond phase timings when non-null.
var _telemetry: JrOrreryTelemetry = null;

## Dirty flag indicating that units or parent effective gravitational parameters need propagation.
var _units_and_mu_dirty: bool = true;

## Whether [member _last_synced_playing] has been broadcast across all current descendants.
var _has_synced_playing: bool = false;

## Last simulation-playing boolean broadcast to tracked descendants.
var _last_synced_playing: bool = false;

## Reusable ancestor element stack for allocation-free LCA queries.
var _lca_b_elems: Array[JrOrreryElement] = [];

## Reusable ancestor vector stack paired with [member _lca_b_elems] for allocation-free LCA queries.
var _lca_b_vecs: PackedVector3Array = PackedVector3Array();

## Monotonic pass stamp identifying the active perspective offset table evaluation pass.
var _persp_pass_stamp: int = 0;

## Cached flag indicating whether the active [member perspective_node] resolves to heliocentric root.
var _persp_is_heliocentric: bool = true;

## Reusable per-step map of parent elements to their massive direct child [JrRailOrbiter] nodes.
var _hill_children_by_parent: Dictionary[JrOrreryElement, Array] = {};

## Reusable per-step focus-relative position cache for massive [JrRailOrbiter] nodes.
var _hill_orbiter_offset: Dictionary[JrRailOrbiter, Vector3] = {};

## Reusable per-step squared Hill entry radii for massive [JrRailOrbiter] nodes.
var _hill_orbiter_entry_sq: Dictionary[JrRailOrbiter, float] = {};

## Reusable per-step squared Hill exit radii for massive [JrRailOrbiter] nodes.
var _hill_orbiter_exit_sq: Dictionary[JrRailOrbiter, float] = {};

## Reusable per-step eligibility flag for Hill-sphere exit checks per [JrRailOrbiter].
var _hill_orbiter_can_exit: Dictionary[JrRailOrbiter, bool] = {};

## Active `(count, step_dt)` key for the shared per-frame forecast perspective projection cache.
var _fc_proj_grid_key: Vector2 = Vector2(-1.0, -1.0);

## Cached synodic inverse rotation bases across the active forecast grid.
var _fc_proj_synodic_bases: Array[Basis] = [];

## Cached common ancestor gravity well per forecast well index.
var _fc_proj_well_common: Array[JrOrreryElement] = [];

## Cached current-time LCA offset from [member perspective_node] to each forecast well's common well.
var _fc_proj_well_common_now: PackedVector3Array = PackedVector3Array();

## Cached `system_basis * common_now` per forecast well index.
var _fc_proj_well_sys_common_now: PackedVector3Array = PackedVector3Array();

## Cached boolean byte (`1` or `0`) indicating synodic rotation per forecast well index.
var _fc_proj_well_is_synodic: PackedByteArray = PackedByteArray();

## Cached common ancestor instance ID per forecast well index.
var _fc_proj_well_common_id: PackedInt64Array = PackedInt64Array();

## Lazy readiness flags for [member _fc_proj_well_origins] per forecast well index.
var _fc_proj_well_origins_ready: PackedByteArray = PackedByteArray();

## Lazily evaluated inertial origins across forecast steps per forecast well index.
var _fc_proj_well_origins: Array[PackedVector3Array] = [];

## Lazy readiness flags for [member _fc_proj_well_offsets] per forecast well index.
var _fc_proj_well_offsets_ready: PackedByteArray = PackedByteArray();

## Lazily evaluated offsets from common ancestor well across forecast steps per forecast well index.
var _fc_proj_well_offsets: Array[PackedVector3Array] = [];

## Tracks whether any [JrBallisticOrbiter] currently holds non-empty runtime forecast polylines.
var _any_ballistic_has_forecast_points: bool = false;

## Coalesces multiple subtree removals within a single frame into one deferred orbital update.
var _deferred_orbit_update_queued: bool = false;

## Dirty flag indicating one or more [JrBallisticOrbiter] nodes require initial state setup.
var _pending_ballistic_init: bool = true;

## Dirty flag indicating one or more [JrBallisticOrbiter] nodes require pre-step anchor validation or pinning.
var _ballistic_anchors_dirty: bool = true;

## Sparse queue of [JrRailOrbiter] nodes whose GPU B-spline lookup tables need baking.
var _dirty_rail_splines: Array[JrRailOrbiter] = [];


## Initializes the generated [JrReferenceDirection], connects subtree signals,
## scans existing descendant elements, and performs an initial orbital update.
func _ready() -> void:
	_last_stepped_time = time
	_initialize_driver_hierarchy()


## Advances editor ghost markers in the Godot Editor, or advances runtime simulation
## time during idle frames when [member use_physics_process] is [code]false[/code].
func _process(delta: float) -> void:
	if _orbit_sequence.is_empty() and _ballistic_sequence.is_empty() and get_child_count() > 0:
		_initialize_driver_hierarchy()
	if Engine.is_editor_hint():
		_step_editor_ghosts(delta)
		return
	if not use_physics_process and _should_step_time():
		step_time(delta)


## Advances runtime simulation time during physics frames when [member use_physics_process] is
## [code]true[/code].
func _physics_process(delta: float) -> void:
	if not Engine.is_editor_hint() and use_physics_process and _should_step_time():
		step_time(delta)


## Releases GPU compute shader resources when leaving the scene tree.
func _exit_tree() -> void:
	if _ballistic_solver != null:
		_ballistic_solver.dispose()


## Ensures GPU compute shader resources are freed before object deletion.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _ballistic_solver != null:
		_ballistic_solver.dispose()


## Overrides [method JrOrreryElement.get_effective_parent_mu] so children of a massless
## binary barycenter driver inherit the combined gravitational parameter of its binary star pair.
func get_effective_parent_mu() -> float:
	_recompute_gravity_strength_from_mass();
	var own_mu: float = _gravity_strength;
	if own_mu > 0.0:
		return own_mu;
	var binary_sum: float = 0.0;
	if not _orbit_sequence.is_empty():
		for child_orb: JrRailOrbiter in _orbit_sequence:
			if (
					_children.get(child_orb, null) == self
					and child_orb.binary_partner != null
					and child_orb.binary_partner.get_parent() == self
			):
				binary_sum += child_orb.get_gravity_strength();
		return binary_sum;
	for child_node: Node in get_children():
		if child_node is JrRailOrbiter:
			var child_orb := child_node as JrRailOrbiter;
			if child_orb.binary_partner != null and child_orb.binary_partner.get_parent() == self:
				binary_sum += child_orb.get_gravity_strength();
	return binary_sum;


## Overrides [method JrOrreryElement._recompute_gravity_strength_from_mass] using
## this driver's [member mass], [member mass_unit], [member distance_unit], and [member time_unit].
func _recompute_gravity_strength_from_mass() -> void:
	_driver_distance_unit = distance_unit
	_driver_time_unit = time_unit
	_gravity_strength = JrOrreryUnits.compute_mu(mass, mass_unit, distance_unit, time_unit)


## Advances [member time] by [code]delta * time_scale[/code] and updates all orbits and ballistics.
func step_time(delta: float) -> void:
	time += delta * time_scale


## Begins a batch hierarchy/parameter update, suppressing intermediate [method update_orbits]
## passes until the matching [method end_batch_updates] call.
func begin_batch_updates() -> void:
	_batch_update_depth += 1


## Ends a batch hierarchy/parameter update started by [method begin_batch_updates] and
## performs a single consolidated [method update_orbits] pass when the outermost batch finishes.
func end_batch_updates() -> void:
	_batch_update_depth = maxi(0, _batch_update_depth - 1);
	if _batch_update_depth == 0 and is_inside_tree():
		_units_and_mu_dirty = true;
		_pending_ballistic_init = true;
		_ballistic_anchors_dirty = true;
		_ballistic_solver.invalidate_resident_step_buffer();
		_propagate_units_and_parent_mu();
		update_orbits();


## Reparents all descendant [JrBallisticOrbiter] nodes in this hierarchy to be direct
## children of this [JrOrbitDriver] without resetting their initialized orbital state
## or losing their original spawn focus reference.
func collect_ballistic_nodes() -> void:
	if _orbit_sequence.is_empty() and _ballistic_sequence.is_empty() and get_child_count() > 0:
		_initialize_driver_hierarchy();
	begin_batch_updates();
	_units_and_mu_dirty = true;
	_propagate_units_and_parent_mu();
	var eval_t: float = 0.0 if Engine.is_editor_hint() else time
	_evaluate_all_orbiters_at(eval_t, _should_step_time())

	var to_collect: Array[JrBallisticOrbiter] = _ballistic_sequence.duplicate()
	for ballistic: JrBallisticOrbiter in to_collect:
		if ballistic == null or not ballistic.is_inside_tree():
			continue
		if ballistic.get_parent() == self:
			continue
		if not ballistic.is_runtime_state_ready():
			_initialize_single_ballistic(ballistic, _should_step_time())
		ballistic.set_suppress_reinit_on_reparent(true)
		ballistic.reparent(self, false)
		ballistic.set_suppress_reinit_on_reparent(false)

	end_batch_updates()


## Alias for [method collect_ballistic_nodes].
func collect_balistic_nodes() -> void:
	collect_ballistic_nodes()


## Evaluates the current LCA-relative position offset of [param element] from
## [member perspective_node] without subtracting shared ancestor orbits.
func evaluate_current_element_offset_from_perspective(element: JrOrreryElement) -> Vector3:
	return _compute_lca_position_difference(element, perspective_node)


## Evaluates the perspective-relative unrotated offset of an anchor-relative trail sample
## [param anchor_rel_pos] anchored to [param anchor_elem] at [param sample_time]
## in reference to the earliest (lowest) common ancestor of [param anchor_elem] and
## [member perspective_node].
func evaluate_trail_sample_relative_to_perspective(
		anchor_elem: JrOrreryElement,
		anchor_rel_pos: Vector3,
		sample_time: float,
		_ref_ballistic_anchor_pos: Vector3 = Vector3.ZERO
) -> Vector3:
	var selected_well: JrOrreryElement = _get_perspective_gravity_well()
	var common_well: JrOrreryElement = _find_common_ancestor_well(anchor_elem, selected_well)
	var offset_from_common: Vector3 = (
			_compute_lca_well_position_difference_at(anchor_elem, common_well, sample_time)
			+ anchor_rel_pos
	)
	var common_from_persp_now: Vector3 = _compute_lca_position_difference(
			common_well,
			perspective_node
	)

	if (
			perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
			and _should_apply_synodic_rotation_to_well(common_well)
	):
		var synodic_step_inv: Basis = _compute_synodic_perspective_basis_at(sample_time)
		var synodic_now_basis: Basis = get_system_transform().basis.transposed()
		return (
				common_from_persp_now
				+ (synodic_now_basis * (synodic_step_inv * offset_from_common))
		)

	return common_from_persp_now + offset_from_common


## Evaluates Keplerian positions for all tracked [JrRailOrbiter] descendants, steps or
## previews all tracked [JrBallisticOrbiter] descendants via [JrBallisticComputeSolver],
## computes the active perspective transform using LCA tree-relative coordinates, and
## emits [signal JrOrreryElement.transform_updated].
func update_orbits() -> void:
	if _updating_orbits or _batch_update_depth > 0 or _reference_plane == null:
		return;

	_owning_driver = self;
	if not child_element_added.is_connected(_on_subtree_element_added) or (
			_orbit_sequence.is_empty() and _ballistic_sequence.is_empty() and get_child_count() > 0
	):
		_reference_plane.configure(
				reference_rotation,
				euler_order,
				positive_is_clockwise,
				lock_to_ecliptic
		);
		if not child_element_added.is_connected(_on_subtree_element_added):
			child_element_added.connect(_on_subtree_element_added);
		if not child_element_removed.is_connected(_on_subtree_element_removed):
			child_element_removed.connect(_on_subtree_element_removed);
		refresh_subtree_tracking();
		_units_and_mu_dirty = true;
		_has_synced_playing = false;
		_pending_ballistic_init = true;
		_ballistic_anchors_dirty = true;

	if _telemetry != null:
		_update_orbits_instrumented();
		return;

	_updating_orbits = true;
	if _units_and_mu_dirty:
		_propagate_units_and_parent_mu();
	var playing: bool = _should_step_time();
	_sync_playing_if_changed(playing);

	_relative_velocity = Vector3.ZERO;
	_global_velocity = Vector3.ZERO;

	if Engine.is_editor_hint():
		_evaluate_editor_system_state(playing);
	else:
		_evaluate_runtime_system_state(playing);

	var persp_basis: Basis = _compute_perspective_basis();
	var is_identity_basis: bool = (persp_basis == Basis.IDENTITY);
	_build_perspective_offset_table();

	var root_rel_pos: Vector3 = Vector3.ZERO if _persp_is_heliocentric else _persp_offset_cached;
	var driver_origin: Vector3 = (
			root_rel_pos if is_identity_basis else (persp_basis * root_rel_pos)
	);
	var driver_system_transform := Transform3D(persp_basis, driver_origin);
	_local_orbital_transform = driver_system_transform;
	_driver_relative_transform = Transform3D.IDENTITY;
	_system_transform = driver_system_transform;
	emit_signal(&"transform_updated");

	if _persp_is_heliocentric and is_identity_basis:
		var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
		for r_idx: int in range(r_seq.size()):
			var orbiter: JrRailOrbiter = r_seq[r_idx];
			orbiter._system_transform = orbiter._driver_relative_transform;
			orbiter.emit_signal(&"transform_updated");
	else:
		var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
		for r_idx: int in range(r_seq.size()):
			var orbiter: JrRailOrbiter = r_seq[r_idx];
			var drv_xf: Transform3D = orbiter._driver_relative_transform;
			var rel_pos: Vector3 = (
					drv_xf.origin
					if _persp_is_heliocentric
					else orbiter._persp_offset_cached
			);
			if is_identity_basis:
				orbiter._system_transform = Transform3D(drv_xf.basis, rel_pos);
			else:
				orbiter._system_transform = Transform3D(
						persp_basis * drv_xf.basis,
						persp_basis * rel_pos
				);
			orbiter.emit_signal(&"transform_updated");

	if not Engine.is_editor_hint():
		_update_runtime_ballistic_forecasts(persp_basis);

	var use_direct_ballistic_xf: bool = _persp_is_heliocentric and is_identity_basis;
	_sync_ballistic_direct_system_xf_flag(use_direct_ballistic_xf);
	if use_direct_ballistic_xf:
		for nr_idx: int in range(_non_root_sync_ballistics.size()):
			var nr_b: JrBallisticOrbiter = _non_root_sync_ballistics[nr_idx];
			nr_b._system_transform = nr_b._driver_relative_transform;
		for i: int in range(_ballistic_sequence.size()):
			_ballistic_sequence[i].emit_signal(&"transform_updated");
	else:
		for ballistic: JrBallisticOrbiter in _ballistic_sequence:
			var rel_pos: Vector3 = _lookup_ballistic_perspective_offset(ballistic);
			var drv_basis: Basis = ballistic._driver_relative_basis;
			if is_identity_basis:
				ballistic._system_transform = Transform3D(drv_basis, rel_pos);
			else:
				ballistic._system_transform = Transform3D(
						persp_basis * drv_basis,
						persp_basis * rel_pos
				);
			ballistic.emit_signal(&"transform_updated");

	if Engine.is_editor_hint():
		_update_editor_ghosts();

	_updating_orbits = false;


## Evaluates the exact analytical inertial driver origin of a tracked [param element]
## at arbitrary simulation timestamp [param eval_time] without mutating live transforms.
func evaluate_element_driver_origin_at(element: JrOrreryElement, eval_time: float) -> Vector3:
	if element == null or element == self or not _element_parents.has(element):
		return Vector3.ZERO
	if element is JrBallisticOrbiter:
		var b_node := element as JrBallisticOrbiter
		var anchor: JrOrreryElement = b_node.get_anchor_element()
		if anchor != null and _is_tracked_or_self(anchor):
			return (
					evaluate_element_driver_origin_at(anchor, eval_time)
					+ b_node.get_anchor_relative_position()
			)
		return b_node.get_driver_position()

	var origin := Vector3.ZERO
	var curr: JrOrreryElement = element
	while curr is JrRailOrbiter and _children.has(curr as JrRailOrbiter):
		var orb := curr as JrRailOrbiter
		if orb.is_orbit_valid():
			origin += orb.get_relative_position(eval_time, _reference_plane)
		curr = _children.get(orb, null)
	return origin


## Returns [code]true[/code] if [param orbiter] is currently tracked within this
## driver's hierarchy.
func has_orbiter(orbiter: JrRailOrbiter) -> bool:
	return _children.has(orbiter)


## Returns [code]true[/code] if [param element] ([JrRailOrbiter] or [JrBallisticOrbiter]) is
## currently tracked within this driver's hierarchy.
func has_element(element: JrOrreryElement) -> bool:
	return _element_parents.has(element)


## Returns the immediate parent [JrOrreryElement] of a tracked [param orbiter],
## or [code]null[/code] if not tracked.
func get_orbiter_parent(orbiter: JrRailOrbiter) -> JrOrreryElement:
	return _children.get(orbiter, null)


## Returns the immediate parent [JrOrreryElement] of any tracked [param element],
## or [code]null[/code] if not tracked.
func get_element_parent(element: JrOrreryElement) -> JrOrreryElement:
	return _element_parents.get(element, null)


## Returns an array of all tracked descendant [JrBallisticOrbiter] nodes.
func get_ballistic_bodies() -> Array[JrBallisticOrbiter]:
	return _ballistic_sequence.duplicate()


## Returns [code]true[/code] if the stepping GPU compute shader pipeline is active.
func is_ballistic_gpu_active() -> bool:
	return _ballistic_solver.is_gpu_active()


## Returns [code]true[/code] if the trajectory forecast GPU compute shader pipeline is active.
func is_ballistic_forecast_gpu_active() -> bool:
	return _ballistic_solver.is_forecast_gpu_active()


## Attaches or detaches a [JrOrreryTelemetry] collector for microsecond profiling of this driver
## and its [JrBallisticComputeSolver].
func set_telemetry(telemetry: JrOrreryTelemetry) -> void:
	_telemetry = telemetry;
	_ballistic_solver.set_telemetry(telemetry);


## Returns the currently attached [JrOrreryTelemetry] collector, or [code]null[/code] if disabled.
func get_telemetry() -> JrOrreryTelemetry:
	return _telemetry;


## Forces [member _ballistic_solver] to execute CPU fallback stepping and forecasting when
## [param force_cpu] is [code]true[/code].
func set_force_cpu_solver(force_cpu: bool) -> void:
	_ballistic_solver.set_force_cpu(force_cpu);


## Returns [code]true[/code] when [member _ballistic_solver] is forced onto its CPU fallback path.
func is_force_cpu_solver() -> bool:
	return _ballistic_solver.is_force_cpu();


## Connects subtree signals, configures [member _reference_plane], scans all
## descendant elements, and evaluates initial orbital and ballistic transforms.
func _initialize_driver_hierarchy() -> void:
	_owning_driver = self;
	_reference_plane.configure(
			reference_rotation,
			euler_order,
			positive_is_clockwise,
			lock_to_ecliptic
	);
	if not child_element_added.is_connected(_on_subtree_element_added):
		child_element_added.connect(_on_subtree_element_added);
	if not child_element_removed.is_connected(_on_subtree_element_removed):
		child_element_removed.connect(_on_subtree_element_removed);
	refresh_subtree_tracking();
	_units_and_mu_dirty = true;
	_has_synced_playing = false;
	_pending_ballistic_init = true;
	_ballistic_anchors_dirty = true;
	_ballistic_solver.invalidate_resident_step_buffer();
	_propagate_units_and_parent_mu();
	update_orbits();


## Advances the editor-only ghost preview clock by [code]delta * editor_play_speed[/code]
## and updates ghost markers on all tracked descendant orbiters and ballistic bodies.
func _step_editor_ghosts(delta: float) -> void:
	if _reference_plane == null or is_zero_approx(editor_play_speed):
		return
	_editor_preview_time += delta * editor_play_speed
	_update_editor_ghosts()


## Updates the editor ghost markers on all tracked descendant orbiters and ballistic
## bodies at the current [member _editor_preview_time].
func _update_editor_ghosts() -> void:
	if not Engine.is_editor_hint() or _reference_plane == null:
		return
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_basis: Basis = _get_parent_driver_transform_for(orbiter).basis
		orbiter.update_editor_ghost(_editor_preview_time, _reference_plane, parent_basis)
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var ballistic_parent_basis: Basis = _get_parent_driver_transform_for(ballistic).basis
		ballistic.update_editor_ghost(
				_editor_preview_time,
				_reference_plane,
				ballistic_parent_basis
		)
	editor_ghost_updated.emit(_editor_preview_time)


## Propagates [member distance_unit], [member time_unit], and parent effective gravitational
## parameters downward to all tracked [JrRailOrbiter] and [JrBallisticOrbiter] nodes.
func _propagate_units_and_parent_mu() -> void:
	set_driver_units(distance_unit, time_unit);
	for orbiter: JrRailOrbiter in _orbit_sequence:
		orbiter.set_driver_units(distance_unit, time_unit);
	var eff_mu_cache: Dictionary[JrOrreryElement, float] = {};
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_elem: JrOrreryElement = _children.get(orbiter, self);
		var eff_mu: float = 0.0;
		if parent_elem != null:
			if not eff_mu_cache.has(parent_elem):
				eff_mu_cache[parent_elem] = parent_elem.get_effective_parent_mu();
			eff_mu = eff_mu_cache[parent_elem];
		orbiter.set_parent_effective_mu(eff_mu);
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_driver_units(distance_unit, time_unit);
		var spawn_parent: JrOrreryElement = _resolve_ballistic_spawn_parent(ballistic);
		var eff_mu: float = 0.0;
		if spawn_parent != null:
			if not eff_mu_cache.has(spawn_parent):
				eff_mu_cache[spawn_parent] = spawn_parent.get_effective_parent_mu();
			eff_mu = eff_mu_cache[spawn_parent];
		ballistic.set_parent_effective_mu(eff_mu);
	_units_and_mu_dirty = false;


## Marks all tracked [JrBallisticOrbiter] forecasts as dirty when system gravity or units change.
func _invalidate_all_ballistic_forecasts() -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.mark_forecast_dirty()


## Evaluates epoch Keplerian states and forward-integrated ballistic trajectory
## previews when running inside the Godot Editor.
func _evaluate_editor_system_state(playing: bool) -> void:
	if not _dirty_rail_splines.is_empty():
		_flush_dirty_rail_splines();
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_xf: Transform3D = _get_parent_driver_transform_for(orbiter);
		var parent_vel: Vector3 = _get_parent_global_velocity_for(orbiter);
		orbiter.set_simulation_playing(playing);
		orbiter.evaluate_orbital_state(0.0, _reference_plane, parent_xf, parent_vel);

	if _ballistic_sequence.is_empty():
		return

	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		_initialize_single_ballistic(ballistic, playing)
		_sync_single_ballistic_transforms(ballistic)

	var forecast_cfg: JrBallisticForecastConfig = _build_forecast_config(_ballistic_sequence, 0.0)
	var well_list: Array[JrOrreryElement] = _ballistic_solver.get_forecast_well_elements(
			forecast_cfg
	)
	var editor_rel_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			_ballistic_solver.compute_forecasts_anchor_relative(forecast_cfg)
	)

	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var pts4: PackedVector4Array = editor_rel_map.get(ballistic, PackedVector4Array())
		var parent_elem: JrOrreryElement = _element_parents.get(ballistic, self)
		var rel_pts: PackedVector3Array = _convert_anchor_forecast_to_element_relative(
				ballistic,
				pts4,
				well_list,
				parent_elem
		)
		ballistic.set_editor_preview_relative_points(rel_pts)


## Evaluates runtime Keplerian states and dispatches GPU Hybrid Universal-Variable
## Kick-Drift-Kick integration across all tracked [JrBallisticOrbiter] bodies.
func _evaluate_runtime_system_state(playing: bool) -> void:
	var dt: float = time - _last_stepped_time if _runtime_initialized else 0.0;
	var prev_local_positions: Dictionary[JrRailOrbiter, Vector3] = {};

	if not _runtime_initialized:
		_evaluate_all_orbiters_at(time, playing);
		if _pending_ballistic_init:
			_initialize_pending_ballistics(playing);
		_sync_all_ballistics_transforms(playing);
		_runtime_initialized = true;
		_last_stepped_time = time;
		return;

	if _pending_ballistic_init:
		_initialize_pending_ballistics(playing);
	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		prev_local_positions = _capture_orbiter_local_positions();

	_evaluate_all_orbiters_at(time, playing);

	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		if _ballistic_anchors_dirty:
			_update_ballistic_anchors_before_step();
		var well_elements: Array[JrOrreryElement] = _build_runtime_well_elements();
		var packed_wells: PackedFloat32Array = _pack_runtime_gravity_wells(
				well_elements,
				prev_local_positions
		);
		_ballistic_solver.configure_step_context(time, tidal_cull_threshold, well_elements);
		_ballistic_solver.step_bodies(_ballistic_sequence, packed_wells, dt, ballistic_substeps);
		for f_idx: int in range(_forecast_enabled_ballistics.size()):
			_forecast_enabled_ballistics[f_idx]._forecast_dirty = true;
		_update_ballistic_anchors_after_step(well_elements);

	_last_stepped_time = time;
	_sync_all_ballistics_transforms(playing);


## Executes [method update_orbits] while recording microsecond sub-phase timings into
## [member _telemetry] and isolating LCA coordinate math from follower signal emission.
func _update_orbits_instrumented() -> void:
	var frame_t0: int = Time.get_ticks_usec();
	_telemetry.begin_frame();
	_updating_orbits = true;

	var t_units: int = Time.get_ticks_usec();
	if _units_and_mu_dirty:
		_propagate_units_and_parent_mu();
	_telemetry.add_phase_usec(
			JrOrreryTelemetry.PHASE_UNITS_AND_MU,
			Time.get_ticks_usec() - t_units
	);

	var playing: bool = _should_step_time();
	_sync_playing_if_changed(playing);
	_relative_velocity = Vector3.ZERO;
	_global_velocity = Vector3.ZERO;

	if Engine.is_editor_hint():
		_evaluate_editor_system_state(playing);
	else:
		_evaluate_runtime_system_state_instrumented(playing);

	var persp_basis: Basis = _apply_driver_and_rail_system_transforms_instrumented();

	if not Engine.is_editor_hint():
		_update_runtime_ballistic_forecasts_instrumented(persp_basis);

	_apply_ballistic_system_transforms_instrumented(persp_basis);

	if Engine.is_editor_hint():
		_update_editor_ghosts();

	_updating_orbits = false;
	_telemetry.end_frame(Time.get_ticks_usec() - frame_t0);
	_record_telemetry_snapshot();


## Computes driver and [JrRailOrbiter] system transforms, recording LCA math separately from
## [method JrOrreryElement.set_system_transform] follower signal emission.
func _apply_driver_and_rail_system_transforms_instrumented() -> Basis:
	var t_math0: int = Time.get_ticks_usec();
	var persp_basis: Basis = _compute_perspective_basis();
	var is_identity_basis: bool = (persp_basis == Basis.IDENTITY);
	_build_perspective_offset_table();

	var root_rel_pos: Vector3 = Vector3.ZERO if _persp_is_heliocentric else _persp_offset_cached;
	var driver_origin: Vector3 = (
			root_rel_pos if is_identity_basis else (persp_basis * root_rel_pos)
	);
	var driver_system_transform := Transform3D(persp_basis, driver_origin);
	_local_orbital_transform = driver_system_transform;
	_driver_relative_transform = Transform3D.IDENTITY;

	var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
	var rail_count: int = r_seq.size();
	var use_direct_driver_xf: bool = _persp_is_heliocentric and is_identity_basis;
	var rail_transforms: Array[Transform3D] = [];
	if not use_direct_driver_xf:
		rail_transforms.resize(rail_count);
		for i: int in range(rail_count):
			var orbiter: JrRailOrbiter = r_seq[i];
			var drv_xf: Transform3D = orbiter._driver_relative_transform;
			var rel_pos: Vector3 = (
					drv_xf.origin
					if _persp_is_heliocentric
					else orbiter._persp_offset_cached
			);
			if is_identity_basis:
				rail_transforms[i] = Transform3D(drv_xf.basis, rel_pos);
			else:
				rail_transforms[i] = Transform3D(
						persp_basis * drv_xf.basis,
						persp_basis * rel_pos
				);
	var t_math1: int = Time.get_ticks_usec();

	_system_transform = driver_system_transform;
	emit_signal(&"transform_updated");
	if use_direct_driver_xf:
		for i: int in range(rail_count):
			var orbiter: JrRailOrbiter = r_seq[i];
			orbiter._system_transform = orbiter._driver_relative_transform;
			orbiter.emit_signal(&"transform_updated");
	else:
		for i: int in range(rail_count):
			var orbiter: JrRailOrbiter = r_seq[i];
			orbiter._system_transform = rail_transforms[i];
			orbiter.emit_signal(&"transform_updated");
	var t_sig1: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SYSTEM_LCA_MATH, t_math1 - t_math0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_TRANSFORM_SIGNAL_EMIT, t_sig1 - t_math1);
	return persp_basis;


## Computes [JrBallisticOrbiter] system transforms, recording LCA math separately from
## [method JrOrreryElement.set_system_transform] follower signal emission.
func _apply_ballistic_system_transforms_instrumented(persp_basis: Basis) -> void:
	var b_seq: Array[JrBallisticOrbiter] = _ballistic_sequence;
	var b_count: int = b_seq.size();
	if b_count == 0:
		return;

	var t_math0: int = Time.get_ticks_usec();
	var is_identity_basis: bool = (persp_basis == Basis.IDENTITY);
	var use_direct_driver_xf: bool = _persp_is_heliocentric and is_identity_basis;
	_sync_ballistic_direct_system_xf_flag(use_direct_driver_xf);
	var ballistic_transforms: Array[Transform3D] = [];
	if use_direct_driver_xf:
		for nr_idx: int in range(_non_root_sync_ballistics.size()):
			var nr_b: JrBallisticOrbiter = _non_root_sync_ballistics[nr_idx];
			nr_b._system_transform = nr_b._driver_relative_transform;
	else:
		ballistic_transforms.resize(b_count);
		for i: int in range(b_count):
			var ballistic: JrBallisticOrbiter = b_seq[i];
			var rel_pos: Vector3 = _lookup_ballistic_perspective_offset(ballistic);
			var drv_basis: Basis = ballistic._driver_relative_basis;
			if is_identity_basis:
				ballistic_transforms[i] = Transform3D(drv_basis, rel_pos);
			else:
				ballistic_transforms[i] = Transform3D(
						persp_basis * drv_basis,
						persp_basis * rel_pos
				);
	var t_math1: int = Time.get_ticks_usec();

	if use_direct_driver_xf:
		for i: int in range(b_count):
			b_seq[i].emit_signal(&"transform_updated");
	else:
		for i: int in range(b_count):
			var ballistic: JrBallisticOrbiter = b_seq[i];
			ballistic._system_transform = ballistic_transforms[i];
			ballistic.emit_signal(&"transform_updated");
	var t_sig1: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SYSTEM_LCA_MATH, t_math1 - t_math0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_TRANSFORM_SIGNAL_EMIT, t_sig1 - t_math1);


## Broadcasts [param use_direct_xf] downward to tracked ballistics only when perspective mode changes.
func _sync_ballistic_direct_system_xf_flag(use_direct_xf: bool) -> void:
	if use_direct_xf == _last_direct_system_xf_state:
		return;
	_last_direct_system_xf_state = use_direct_xf;
	for i: int in range(_ballistic_sequence.size()):
		_ballistic_sequence[i]._direct_system_xf = use_direct_xf;


## Precomputes cancellation-free LCA offsets from [member perspective_node] for all gravity wells
## in a single top-down pass so per-orbiter and per-ballistic lookups run in `O(1)`.
func _build_perspective_offset_table() -> void:
	_persp_is_heliocentric = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	if _persp_is_heliocentric:
		return;

	_persp_pass_stamp += 1;
	var stamp: int = _persp_pass_stamp;
	var self_elem: JrOrreryElement = self;

	var curr_p: JrOrreryElement = perspective_node;
	var accum_p := Vector3.ZERO;
	while curr_p != null:
		curr_p._persp_offset_cached = -accum_p;
		curr_p._persp_offset_stamp = stamp;
		if curr_p == self_elem:
			break;
		accum_p += _get_element_kinematic_local_offset(curr_p);
		curr_p = _get_element_kinematic_parent(curr_p);

	if _persp_offset_stamp != stamp:
		_persp_offset_cached = -accum_p;
		_persp_offset_stamp = stamp;
	var root_offset: Vector3 = _persp_offset_cached;

	var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
	for i: int in range(r_seq.size()):
		var orbiter: JrRailOrbiter = r_seq[i];
		if orbiter._persp_offset_stamp == stamp:
			continue;
		var parent_elem: JrOrreryElement = orbiter._tracked_scene_parent;
		var parent_offset: Vector3 = (
				parent_elem._persp_offset_cached
				if (parent_elem != null and parent_elem._persp_offset_stamp == stamp)
				else root_offset
		);
		orbiter._persp_offset_cached = parent_offset + orbiter._focus_relative_position;
		orbiter._persp_offset_stamp = stamp;


## Returns the cancellation-free LCA offset of [param ballistic] from [member perspective_node]
## in `O(1)` using [member JrOrreryElement._persp_offset_cached].
func _lookup_ballistic_perspective_offset(ballistic: JrBallisticOrbiter) -> Vector3:
	if _persp_is_heliocentric:
		return ballistic.get_driver_position();
	var stamp: int = _persp_pass_stamp;
	if ballistic._persp_offset_stamp == stamp:
		return ballistic._persp_offset_cached;
	var kin_parent: JrOrreryElement = _get_element_kinematic_parent(ballistic);
	var parent_offset: Vector3 = (
			kin_parent._persp_offset_cached
			if (kin_parent != null and kin_parent._persp_offset_stamp == stamp)
			else _persp_offset_cached
	);
	return parent_offset + ballistic.get_anchor_relative_position();


## Executes [method _evaluate_runtime_system_state] while recording microsecond sub-phase timings
## into [member _telemetry].
func _evaluate_runtime_system_state_instrumented(playing: bool) -> void:
	var dt: float = time - _last_stepped_time if _runtime_initialized else 0.0;
	var prev_local_positions: Dictionary[JrRailOrbiter, Vector3] = {};

	if not _runtime_initialized:
		var t_init_rail: int = Time.get_ticks_usec();
		_evaluate_all_orbiters_at(time, playing);
		var t_init_b: int = Time.get_ticks_usec();
		if _pending_ballistic_init:
			_initialize_pending_ballistics(playing);
		var t_init_sync: int = Time.get_ticks_usec();
		_sync_all_ballistics_transforms(playing);
		var t_init_done: int = Time.get_ticks_usec();
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_RAIL_EVAL, t_init_b - t_init_rail);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_INIT, t_init_sync - t_init_b);
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_BALLISTIC_TRANSFORM_SYNC,
				t_init_done - t_init_sync
		);
		_runtime_initialized = true;
		_last_stepped_time = time;
		return;

	var t0: int = Time.get_ticks_usec();
	if _pending_ballistic_init:
		_initialize_pending_ballistics(playing);
	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		prev_local_positions = _capture_orbiter_local_positions();
	var t1: int = Time.get_ticks_usec();

	_evaluate_all_orbiters_at(time, playing);
	var t2: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_INIT, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_RAIL_EVAL, t2 - t1);

	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		var t_pre0: int = Time.get_ticks_usec();
		if _ballistic_anchors_dirty:
			_update_ballistic_anchors_before_step();
		var t_pack0: int = Time.get_ticks_usec();

		var well_elements: Array[JrOrreryElement] = _build_runtime_well_elements();
		var packed_wells: PackedFloat32Array = _pack_runtime_gravity_wells(
				well_elements,
				prev_local_positions
		);
		_ballistic_solver.configure_step_context(time, tidal_cull_threshold, well_elements);
		var t_step0: int = Time.get_ticks_usec();

		_ballistic_solver.step_bodies(_ballistic_sequence, packed_wells, dt, ballistic_substeps);
		for f_idx: int in range(_forecast_enabled_ballistics.size()):
			_forecast_enabled_ballistics[f_idx]._forecast_dirty = true;
		var t_post0: int = Time.get_ticks_usec();

		_update_ballistic_anchors_after_step(well_elements);
		var t_post1: int = Time.get_ticks_usec();

		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_PRE, t_pack0 - t_pre0);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_WELL_PACK, t_step0 - t_pack0);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_POST, t_post1 - t_post0);

	_last_stepped_time = time;
	var t_sync0: int = Time.get_ticks_usec();
	_sync_all_ballistics_transforms(playing);
	_telemetry.add_phase_usec(
			JrOrreryTelemetry.PHASE_BALLISTIC_TRANSFORM_SYNC,
			Time.get_ticks_usec() - t_sync0
	);


## Populates diagnostic entity counts and GPU flags on [member _telemetry] outside frame timing.
func _record_telemetry_snapshot() -> void:
	var on_rails_count: int = 0;
	var unpinned_count: int = 0;
	var forecast_targets: Array[JrBallisticOrbiter] = [];
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if ballistic.is_on_rails():
			on_rails_count += 1;
		if ballistic.pinned_anchor == null:
			unpinned_count += 1;
		if enable_ballistic_forecasts and ballistic.runtime_forecast_enabled:
			forecast_targets.append(ballistic);

	var valid_well_count: int = 1;
	for orbiter: JrRailOrbiter in _orbit_sequence:
		if orbiter.is_orbit_valid():
			valid_well_count += 1;

	var step_uses_gpu: bool = (
			_ballistic_solver.is_gpu_active()
			and _ballistic_sequence.size() >= JrBallisticComputeSolver.MIN_GPU_STEP_BODIES
	);
	var forecast_uses_gpu: bool = _ballistic_solver.should_use_gpu_for_forecast(
			forecast_targets,
			valid_well_count
	);
	_telemetry.record_body_counts(
			_orbit_sequence.size(),
			_ballistic_sequence.size(),
			on_rails_count,
			unpinned_count
	);
	_telemetry.record_solver_state(
			valid_well_count,
			forecast_targets.size(),
			step_uses_gpu,
			forecast_uses_gpu
	);


## Broadcasts [param playing] to tracked descendants only when the playing state or hierarchy changed.
func _sync_playing_if_changed(playing: bool) -> void:
	set_simulation_playing(playing);
	if _has_synced_playing and _last_synced_playing == playing:
		return;
	for orbiter: JrRailOrbiter in _orbit_sequence:
		orbiter.set_simulation_playing(playing);
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_simulation_playing(playing);
	_last_synced_playing = playing;
	_has_synced_playing = true;


## Evaluates all tracked [JrRailOrbiter] nodes in [member _orbit_sequence] at [param eval_time].
func _evaluate_all_orbiters_at(eval_time: float, _playing: bool) -> void:
	if not _dirty_rail_splines.is_empty():
		_flush_dirty_rail_splines();
	var ref_plane: JrReferenceDirection = _reference_plane;
	var self_elem: JrOrreryElement = self;
	var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
	for i: int in range(r_seq.size()):
		var orbiter: JrRailOrbiter = r_seq[i];
		var parent_elem: JrOrreryElement = orbiter._tracked_scene_parent;
		if parent_elem == null or parent_elem == self_elem:
			orbiter.evaluate_runtime_root_rail(eval_time, ref_plane);
		else:
			orbiter.evaluate_runtime_child_rail(
					eval_time,
					ref_plane,
					parent_elem._driver_relative_transform,
					parent_elem._global_velocity
			);


## Flushes all queued [JrRailOrbiter] nodes in [member _dirty_rail_splines] through the GPU/CPU
## periodic cubic B-spline baker on [member _ballistic_solver].
func _flush_dirty_rail_splines() -> void:
	var batch: Array[JrRailOrbiter] = _dirty_rail_splines.duplicate();
	_dirty_rail_splines.clear();
	_ballistic_solver.bake_rail_splines(batch);


## Initializes any [JrBallisticOrbiter] nodes that were newly added or requested a state reset.
func _initialize_pending_ballistics(playing: bool) -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if not ballistic.is_runtime_state_ready():
			_initialize_single_ballistic(ballistic, playing);
	_pending_ballistic_init = false;


## Initializes a single [param ballistic] body relative to its spawn parent and initial anchor.
func _initialize_single_ballistic(ballistic: JrBallisticOrbiter, playing: bool) -> void:
	ballistic.set_simulation_playing(playing);
	var spawn_parent: JrOrreryElement = _resolve_ballistic_spawn_parent(ballistic);
	var spawn_xf: Transform3D = _get_element_driver_transform(spawn_parent);
	var spawn_vel: Vector3 = _get_element_global_velocity(spawn_parent);
	var init_anchor: JrOrreryElement = spawn_parent;
	if ballistic.pinned_anchor != null and _is_tracked_or_self(ballistic.pinned_anchor):
		init_anchor = ballistic.pinned_anchor;
	ballistic.initialize_from_parent(spawn_xf, spawn_vel, _reference_plane, init_anchor);
	if init_anchor != spawn_parent:
		var spawn_to_anchor_pos: Vector3 = _compute_lca_position_difference(
				init_anchor,
				spawn_parent
		);
		var spawn_to_anchor_vel: Vector3 = _compute_lca_velocity_difference(
				init_anchor,
				spawn_parent
		);
		ballistic.apply_integrated_anchor_state(
				ballistic.get_anchor_relative_position() - spawn_to_anchor_pos,
				ballistic.get_anchor_relative_velocity() - spawn_to_anchor_vel
		);


## Synchronizes transforms on all non-root-fast [JrBallisticOrbiter] nodes in `O(K)` and
## graduates root-parent/root-anchor bodies to lazy [member JrBallisticOrbiter._fast_root_sync_ready].
func _sync_all_ballistics_transforms(_playing: bool) -> void:
	var nr_count: int = _non_root_sync_ballistics.size();
	if nr_count == 0:
		return;

	var self_elem: JrOrreryElement = self;
	var write_idx: int = 0;

	for r: int in range(nr_count):
		var ballistic: JrBallisticOrbiter = _non_root_sync_ballistics[r];
		if not is_instance_valid(ballistic) or ballistic._owning_driver != self_elem:
			if is_instance_valid(ballistic):
				ballistic._in_non_root_sync_list = false;
			continue;

		var scene_parent: JrOrreryElement = ballistic._tracked_scene_parent;
		if (
				(scene_parent != self_elem and scene_parent != null)
				or ballistic._driver_basis_dirty
				or not ballistic._parent_basis_is_identity
		):
			_sync_single_ballistic_transforms(ballistic);
			ballistic._in_non_root_sync_list = true;
			_non_root_sync_ballistics[write_idx] = ballistic;
			write_idx += 1;
			continue;

		var anchor: JrOrreryElement = ballistic._anchor_element;
		if anchor != null and anchor != self_elem and anchor._owning_driver == self_elem:
			var anchor_pos: Vector3 = anchor.get_driver_relative_transform().origin;
			var anchor_vel: Vector3 = anchor.get_global_velocity();
			var pos: Vector3 = anchor_pos + ballistic._anchor_relative_position;
			var vel: Vector3 = anchor_vel + ballistic._anchor_relative_velocity;
			ballistic._last_parent_origin = Vector3.ZERO;
			ballistic._last_parent_velocity = Vector3.ZERO;
			ballistic._anchor_to_parent_offset = anchor_pos;
			ballistic._anchor_to_parent_velocity = anchor_vel;
			ballistic._last_anchor_origin = anchor_pos;
			ballistic._last_anchor_velocity = anchor_vel;
			ballistic._root_zero_offsets_cached = false;
			var xf := Transform3D(ballistic._driver_relative_basis, pos);
			ballistic._focus_relative_position = pos;
			ballistic._relative_velocity = vel;
			ballistic._global_velocity = vel;
			ballistic._local_orbital_transform = xf;
			ballistic._driver_relative_transform = xf;
			ballistic._in_non_root_sync_list = true;
			_non_root_sync_ballistics[write_idx] = ballistic;
			write_idx += 1;
		else:
			if not ballistic._root_zero_offsets_cached:
				ballistic._last_parent_origin = Vector3.ZERO;
				ballistic._last_parent_velocity = Vector3.ZERO;
				ballistic._anchor_to_parent_offset = Vector3.ZERO;
				ballistic._anchor_to_parent_velocity = Vector3.ZERO;
				ballistic._last_anchor_origin = Vector3.ZERO;
				ballistic._last_anchor_velocity = Vector3.ZERO;
				ballistic._root_zero_offsets_cached = true;
			ballistic._in_non_root_sync_list = false;
			ballistic._fast_root_sync_ready = true;

	if write_idx != nr_count:
		_non_root_sync_ballistics.resize(write_idx);


## Synchronizes a single [param ballistic] node's transforms with its anchor and scene parent.
func _sync_single_ballistic_transforms(ballistic: JrBallisticOrbiter) -> void:
	var scene_parent: JrOrreryElement = ballistic._tracked_scene_parent;
	if scene_parent == null or scene_parent._owning_driver != self:
		scene_parent = self;
	var anchor: JrOrreryElement = ballistic._anchor_element;
	if anchor == null or anchor._owning_driver != self:
		anchor = scene_parent;

	if scene_parent == null or scene_parent == self:
		if anchor == null or anchor == self:
			ballistic.sync_transforms_with_anchor_and_parent(
					Transform3D.IDENTITY,
					Vector3.ZERO,
					Vector3.ZERO,
					Vector3.ZERO
			);
		else:
			ballistic.sync_transforms_with_anchor_and_parent(
					Transform3D.IDENTITY,
					Vector3.ZERO,
					anchor.get_driver_relative_transform().origin,
					anchor.get_global_velocity()
			);
		return;

	var parent_xf: Transform3D = scene_parent.get_driver_relative_transform();
	var parent_vel: Vector3 = scene_parent.get_global_velocity();
	if anchor == scene_parent:
		ballistic.sync_transforms_with_anchor_and_parent(
				parent_xf,
				parent_vel,
				Vector3.ZERO,
				Vector3.ZERO
		);
		return;

	var anchor_to_parent_pos: Vector3 = _compute_lca_position_difference(anchor, scene_parent);
	var anchor_to_parent_vel: Vector3 = _compute_lca_velocity_difference(anchor, scene_parent);
	ballistic.sync_transforms_with_anchor_and_parent(
			parent_xf,
			parent_vel,
			anchor_to_parent_pos,
			anchor_to_parent_vel
	);


## Records the current parent-local orbital offset of each [JrRailOrbiter] prior to stepping.
func _capture_orbiter_local_positions() -> Dictionary[JrRailOrbiter, Vector3]:
	var positions: Dictionary[JrRailOrbiter, Vector3] = {};
	for orbiter: JrRailOrbiter in _orbit_sequence:
		if orbiter.is_orbit_valid():
			positions[orbiter] = orbiter.get_focus_relative_position();
	return positions;


## Builds the ordered list of gravity-well [JrOrreryElement] nodes (index 0 = driver,
## followed by all valid [JrRailOrbiter] nodes in topological parent-before-child order).
func _build_runtime_well_elements() -> Array[JrOrreryElement]:
	_runtime_well_index = 0;
	var wells: Array[JrOrreryElement] = [self];
	for orbiter: JrRailOrbiter in _orbit_sequence:
		if orbiter.is_orbit_valid():
			orbiter._runtime_well_index = wells.size();
			wells.append(orbiter);
		else:
			orbiter._runtime_well_index = -1;
	return wells;


## Packs start-of-step and end-of-step parent-local positions, gravitational parameters,
## parent well indices, and Hill-sphere entry/exit squared radii for all [param well_elements].
func _pack_runtime_gravity_wells(
		well_elements: Array[JrOrreryElement],
		prev_local_positions: Dictionary[JrRailOrbiter, Vector3]
) -> PackedFloat32Array:
	var w_count: int = well_elements.size();
	var packed := PackedFloat32Array();
	packed.resize(w_count * JrBallisticComputeSolver.FLOATS_PER_WELL);

	var eff_mu_by_idx := PackedFloat32Array();
	eff_mu_by_idx.resize(w_count);
	eff_mu_by_idx[0] = get_effective_parent_mu();

	packed[3] = _gravity_strength;
	packed[7] = -1.0;

	var self_elem: JrOrreryElement = self;
	for i: int in range(1, w_count):
		var orbiter := well_elements[i] as JrRailOrbiter;
		eff_mu_by_idx[i] = orbiter.get_effective_parent_mu();
		var mu: float = orbiter.get_gravity_strength();
		var end_local: Vector3 = orbiter.get_focus_relative_position();
		var start_local: Vector3 = prev_local_positions.get(orbiter, end_local);
		var parent_elem: JrOrreryElement = _children.get(orbiter, self_elem);
		var parent_idx: int = (
				parent_elem._runtime_well_index
				if (parent_elem != null and parent_elem._runtime_well_index >= 0)
				else 0
		);

		var entry_sq: float = 0.0;
		var exit_sq: float = 0.0;
		var can_exit: float = 0.0;
		var mu_parent: float = eff_mu_by_idx[parent_idx];
		if mu > 0.0 and mu_parent > 0.0:
			var d_len: float = end_local.length();
			var r_hill: float = d_len * pow(mu / (3.0 * mu_parent), 1.0 / 3.0);
			var exit_r: float = HILL_EXIT_MULTIPLIER * r_hill;
			exit_sq = exit_r * exit_r;
			can_exit = 1.0;
			if d_len > 1e-6:
				var entry_r: float = HILL_ENTRY_MULTIPLIER * r_hill;
				entry_sq = entry_r * entry_r;

		var base: int = i * JrBallisticComputeSolver.FLOATS_PER_WELL;
		packed[base + 0] = start_local.x;
		packed[base + 1] = start_local.y;
		packed[base + 2] = start_local.z;
		packed[base + 3] = mu;
		packed[base + 4] = end_local.x;
		packed[base + 5] = end_local.y;
		packed[base + 6] = end_local.z;
		packed[base + 7] = float(parent_idx);
		packed[base + 8] = entry_sq;
		packed[base + 9] = exit_sq;
		packed[base + 10] = can_exit;
		packed[base + 11] = 0.0;

	return packed;


## Synchronizes any pinned anchor assignments before stepping [member _ballistic_sequence].
func _update_ballistic_anchors_before_step() -> void:
	var self_elem: JrOrreryElement = self;
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var curr_anchor: JrOrreryElement = ballistic._anchor_element;
		var anchor_valid: bool = curr_anchor != null and curr_anchor._owning_driver == self_elem;
		if not anchor_valid:
			curr_anchor = _resolve_ballistic_spawn_parent(ballistic);
			var spawn_p: JrOrreryElement = ballistic._tracked_scene_parent;
			if spawn_p == null or spawn_p._owning_driver != self_elem:
				spawn_p = self_elem;
			var diff_pos: Vector3 = _compute_lca_position_difference(curr_anchor, spawn_p);
			var diff_vel: Vector3 = _compute_lca_velocity_difference(curr_anchor, spawn_p);
			ballistic.switch_anchor_element(curr_anchor, diff_pos, diff_vel);

		var pinned: JrOrreryElement = ballistic.pinned_anchor;
		if pinned == null or pinned == curr_anchor:
			continue;
		if pinned._owning_driver == self_elem:
			var old_to_new_pos: Vector3 = _compute_lca_position_difference(pinned, curr_anchor);
			var old_to_new_vel: Vector3 = _compute_lca_velocity_difference(pinned, curr_anchor);
			ballistic.switch_anchor_element(pinned, old_to_new_pos, old_to_new_vel);
	_ballistic_anchors_dirty = false;


## Evaluates Hill-sphere hysteresis (`0.98x` entry / `1.05x` exit) after stepping
## for all unpinned [JrBallisticOrbiter] nodes.
func _update_ballistic_anchors_after_step(well_elements: Array[JrOrreryElement]) -> void:
	if _orbit_sequence.is_empty() or _ballistic_sequence.is_empty():
		return;
	if (
			_ballistic_solver.has_gpu_step_handoffs()
			and well_elements.size() == _orbit_sequence.size() + 1
	):
		_apply_gpu_step_handoffs(well_elements);
		return;

	var tables_built: bool = false;
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if ballistic.pinned_anchor != null:
			continue;
		if not tables_built:
			_build_step_hill_handoff_tables();
			tables_built = true;
		_evaluate_hill_sphere_handoff_fast(ballistic);


## Applies only the sparse Hill-sphere anchor transitions reported by the GPU stepping shader.
func _apply_gpu_step_handoffs(well_elements: Array[JrOrreryElement]) -> void:
	var body_indices: PackedInt32Array = _ballistic_solver.get_gpu_handoff_body_indices();
	if body_indices.is_empty():
		return;
	var target_wells: PackedInt32Array = _ballistic_solver.get_gpu_handoff_target_wells();
	var well_count: int = well_elements.size();
	var b_count: int = _ballistic_sequence.size();
	var self_elem: JrOrreryElement = self;

	for h: int in range(body_indices.size()):
		var b_idx: int = body_indices[h];
		var w_idx: int = target_wells[h];
		if b_idx < 0 or b_idx >= b_count or w_idx < 0 or w_idx >= well_count:
			continue;
		var ballistic: JrBallisticOrbiter = _ballistic_sequence[b_idx];
		if ballistic.pinned_anchor != null:
			continue;
		var curr_anchor: JrOrreryElement = ballistic.get_anchor_element();
		var target_anchor: JrOrreryElement = well_elements[w_idx];
		if curr_anchor == null or target_anchor == null or curr_anchor == target_anchor:
			continue;

		if target_anchor is JrRailOrbiter and _children.get(target_anchor as JrRailOrbiter, self_elem) == curr_anchor:
			var child_orb := target_anchor as JrRailOrbiter;
			ballistic.switch_anchor_element(
					child_orb,
					child_orb.get_focus_relative_position(),
					child_orb.get_relative_velocity()
			);
		elif curr_anchor is JrRailOrbiter and _children.get(curr_anchor as JrRailOrbiter, self_elem) == target_anchor:
			var parent_orb := curr_anchor as JrRailOrbiter;
			ballistic.switch_anchor_element(
					target_anchor,
					-parent_orb.get_focus_relative_position(),
					-parent_orb.get_relative_velocity()
			);
		else:
			ballistic.switch_anchor_element(
					target_anchor,
					_compute_lca_position_difference(target_anchor, curr_anchor),
					_compute_lca_velocity_difference(target_anchor, curr_anchor)
			);


## Precomputes squared Hill-sphere entry/exit radii and massive child lookup lists once per step.
func _build_step_hill_handoff_tables() -> void:
	_hill_children_by_parent.clear();
	_hill_orbiter_offset.clear();
	_hill_orbiter_entry_sq.clear();
	_hill_orbiter_exit_sq.clear();
	_hill_orbiter_can_exit.clear();

	var eff_mu_map: Dictionary[JrOrreryElement, float] = {};
	var self_elem: JrOrreryElement = self;
	eff_mu_map[self_elem] = get_effective_parent_mu();

	for orbiter: JrRailOrbiter in _orbit_sequence:
		var mu_orb: float = orbiter.get_gravity_strength();
		if mu_orb <= 0.0:
			continue;
		var parent_elem: JrOrreryElement = _children.get(orbiter, self);
		if parent_elem == null:
			continue;
		if not eff_mu_map.has(parent_elem):
			eff_mu_map[parent_elem] = parent_elem.get_effective_parent_mu();
		var mu_parent: float = eff_mu_map[parent_elem];
		if mu_parent <= 0.0:
			continue;

		var offset: Vector3 = orbiter.get_focus_relative_position();
		var d_len: float = offset.length();
		var r_hill: float = d_len * pow(mu_orb / (3.0 * mu_parent), 1.0 / 3.0);
		var exit_r: float = HILL_EXIT_MULTIPLIER * r_hill;
		_hill_orbiter_offset[orbiter] = offset;
		_hill_orbiter_exit_sq[orbiter] = exit_r * exit_r;
		_hill_orbiter_can_exit[orbiter] = true;

		if not orbiter.is_orbit_valid() or d_len <= 1e-6:
			continue;
		var entry_r: float = HILL_ENTRY_MULTIPLIER * r_hill;
		_hill_orbiter_entry_sq[orbiter] = entry_r * entry_r;
		if not _hill_children_by_parent.has(parent_elem):
			var child_arr: Array[JrRailOrbiter] = [];
			_hill_children_by_parent[parent_elem] = child_arr;
		var children_list: Array = _hill_children_by_parent[parent_elem];
		children_list.append(orbiter);


## Evaluates whether [param ballistic] should enter a direct massive child well of its current
## anchor (`< 2.4 * r_Hill`) or exit to its current anchor's parent well (`> 2.6 * r_Hill`)
## using precomputed squared radii.
func _evaluate_hill_sphere_handoff_fast(ballistic: JrBallisticOrbiter) -> void:
	var curr_anchor: JrOrreryElement = ballistic.get_anchor_element();
	if curr_anchor == null or not _is_tracked_or_self(curr_anchor):
		return;

	var r_rel: Vector3 = ballistic.get_anchor_relative_position();
	if _hill_children_by_parent.has(curr_anchor):
		var child_list: Array = _hill_children_by_parent[curr_anchor];
		var best_child: JrRailOrbiter = null;
		var best_dist_sq: float = INF;
		for i: int in range(child_list.size()):
			var orbiter := child_list[i] as JrRailOrbiter;
			var child_offset: Vector3 = _hill_orbiter_offset[orbiter];
			var dist_sq: float = (r_rel - child_offset).length_squared();
			if dist_sq < _hill_orbiter_entry_sq[orbiter] and dist_sq < best_dist_sq:
				best_dist_sq = dist_sq;
				best_child = orbiter;
		if best_child != null:
			var old_to_new_pos: Vector3 = _hill_orbiter_offset[best_child];
			var old_to_new_vel: Vector3 = best_child.get_relative_velocity();
			ballistic.switch_anchor_element(best_child, old_to_new_pos, old_to_new_vel);
			return;

	if curr_anchor is JrRailOrbiter:
		var anchor_orb := curr_anchor as JrRailOrbiter;
		if not _hill_orbiter_can_exit.get(anchor_orb, false):
			return;
		if r_rel.length_squared() > _hill_orbiter_exit_sq[anchor_orb]:
			var parent_anchor: JrOrreryElement = _children.get(anchor_orb, self);
			if parent_anchor == null:
				return;
			var anchor_offset: Vector3 = _hill_orbiter_offset[anchor_orb];
			var old_to_new_vel: Vector3 = -anchor_orb.get_relative_velocity();
			ballistic.switch_anchor_element(parent_anchor, -anchor_offset, old_to_new_vel);


## Computes the exact position difference [code]r_a - r_b[/code] between [param elem_a]
## and [param elem_b] by summing tree-relative offsets only up to their Lowest Common Ancestor,
## avoiding cancellation of shared ancestor distances (such as 1 AU).
func _compute_lca_position_difference(elem_a: JrOrreryElement, elem_b: JrOrreryElement) -> Vector3:
	var target_a: JrOrreryElement = (
			elem_a
			if (elem_a != null and _is_tracked_or_self(elem_a))
			else self
	);
	var target_b: JrOrreryElement = (
			elem_b
			if (elem_b != null and _is_tracked_or_self(elem_b))
			else self
	);
	if target_a == target_b:
		return Vector3.ZERO;

	if target_b == self:
		var curr_only_a: JrOrreryElement = target_a;
		var accum_only_a := Vector3.ZERO;
		while curr_only_a != null and curr_only_a != self:
			accum_only_a += _get_element_kinematic_local_offset(curr_only_a);
			curr_only_a = _get_element_kinematic_parent(curr_only_a);
		return accum_only_a;

	if target_a == self:
		var curr_only_b: JrOrreryElement = target_b;
		var accum_only_b := Vector3.ZERO;
		while curr_only_b != null and curr_only_b != self:
			accum_only_b += _get_element_kinematic_local_offset(curr_only_b);
			curr_only_b = _get_element_kinematic_parent(curr_only_b);
		return -accum_only_b;

	_lca_b_elems.clear();
	_lca_b_vecs.clear();
	var curr_b: JrOrreryElement = target_b;
	var accum_b := Vector3.ZERO;
	while curr_b != null:
		_lca_b_elems.append(curr_b);
		_lca_b_vecs.append(accum_b);
		if curr_b == self:
			break;
		accum_b += _get_element_kinematic_local_offset(curr_b);
		curr_b = _get_element_kinematic_parent(curr_b);

	var curr_a: JrOrreryElement = target_a;
	var accum_a := Vector3.ZERO;
	while curr_a != null:
		var match_idx: int = _lca_b_elems.find(curr_a);
		if match_idx >= 0:
			return accum_a - _lca_b_vecs[match_idx];
		if curr_a == self:
			break;
		accum_a += _get_element_kinematic_local_offset(curr_a);
		curr_a = _get_element_kinematic_parent(curr_a);

	return accum_a - accum_b;


## Computes the exact velocity difference [code]v_a - v_b[/code] between [param elem_a]
## and [param elem_b] by summing tree-relative velocities only up to their Lowest Common Ancestor.
func _compute_lca_velocity_difference(elem_a: JrOrreryElement, elem_b: JrOrreryElement) -> Vector3:
	var target_a: JrOrreryElement = (
			elem_a
			if (elem_a != null and _is_tracked_or_self(elem_a))
			else self
	);
	var target_b: JrOrreryElement = (
			elem_b
			if (elem_b != null and _is_tracked_or_self(elem_b))
			else self
	);
	if target_a == target_b:
		return Vector3.ZERO;

	if target_b == self:
		var curr_only_a: JrOrreryElement = target_a;
		var accum_only_a := Vector3.ZERO;
		while curr_only_a != null and curr_only_a != self:
			accum_only_a += _get_element_kinematic_local_velocity(curr_only_a);
			curr_only_a = _get_element_kinematic_parent(curr_only_a);
		return accum_only_a;

	if target_a == self:
		var curr_only_b: JrOrreryElement = target_b;
		var accum_only_b := Vector3.ZERO;
		while curr_only_b != null and curr_only_b != self:
			accum_only_b += _get_element_kinematic_local_velocity(curr_only_b);
			curr_only_b = _get_element_kinematic_parent(curr_only_b);
		return -accum_only_b;

	_lca_b_elems.clear();
	_lca_b_vecs.clear();
	var curr_b: JrOrreryElement = target_b;
	var accum_b := Vector3.ZERO;
	while curr_b != null:
		_lca_b_elems.append(curr_b);
		_lca_b_vecs.append(accum_b);
		if curr_b == self:
			break;
		accum_b += _get_element_kinematic_local_velocity(curr_b);
		curr_b = _get_element_kinematic_parent(curr_b);

	var curr_a: JrOrreryElement = target_a;
	var accum_a := Vector3.ZERO;
	while curr_a != null:
		var match_idx: int = _lca_b_elems.find(curr_a);
		if match_idx >= 0:
			return accum_a - _lca_b_vecs[match_idx];
		if curr_a == self:
			break;
		accum_a += _get_element_kinematic_local_velocity(curr_a);
		curr_a = _get_element_kinematic_parent(curr_a);

	return accum_a - accum_b;


## Computes the analytical position difference [code]r_a(eval_time) - r_b(eval_time)[/code]
## between two gravity-well elements at [param eval_time] using LCA traversal.
func _compute_lca_well_position_difference_at(
		well_a: JrOrreryElement,
		well_b: JrOrreryElement,
		eval_time: float
) -> Vector3:
	var target_a: JrOrreryElement = (
			well_a
			if (well_a != null and _is_tracked_or_self(well_a))
			else self
	);
	var target_b: JrOrreryElement = (
			well_b
			if (well_b != null and _is_tracked_or_self(well_b))
			else self
	);
	if target_a == target_b:
		return Vector3.ZERO;

	if target_b == self:
		return evaluate_element_driver_origin_at(target_a, eval_time);
	if target_a == self:
		return -evaluate_element_driver_origin_at(target_b, eval_time);

	_lca_b_elems.clear();
	_lca_b_vecs.clear();
	var curr_b: JrOrreryElement = target_b;
	var accum_b := Vector3.ZERO;
	while curr_b != null:
		_lca_b_elems.append(curr_b);
		_lca_b_vecs.append(accum_b);
		if curr_b == self:
			break;
		if curr_b is JrRailOrbiter:
			var orb_b := curr_b as JrRailOrbiter;
			if orb_b.is_orbit_valid():
				accum_b += orb_b.get_relative_position(eval_time, _reference_plane);
			curr_b = _children.get(orb_b, self);
		else:
			curr_b = _element_parents.get(curr_b, self);

	var curr_a: JrOrreryElement = target_a;
	var accum_a := Vector3.ZERO;
	while curr_a != null:
		var match_idx: int = _lca_b_elems.find(curr_a);
		if match_idx >= 0:
			return accum_a - _lca_b_vecs[match_idx];
		if curr_a == self:
			break;
		if curr_a is JrRailOrbiter:
			var orb_a := curr_a as JrRailOrbiter;
			if orb_a.is_orbit_valid():
				accum_a += orb_a.get_relative_position(eval_time, _reference_plane);
			curr_a = _children.get(orb_a, self);
		else:
			curr_a = _element_parents.get(curr_a, self);

	return accum_a - accum_b;


## Returns the immediate kinematic parent of [param element] in the gravity-well tree
## (using [method JrBallisticOrbiter.get_anchor_element] for ballistic bodies).
func _get_element_kinematic_parent(element: JrOrreryElement) -> JrOrreryElement:
	if element == null or element == self:
		return null
	if element is JrBallisticOrbiter:
		var anchor: JrOrreryElement = (element as JrBallisticOrbiter).get_anchor_element()
		if anchor != null and _is_tracked_or_self(anchor):
			return anchor
		return _element_parents.get(element, self)
	if element is JrRailOrbiter:
		return _children.get(element as JrRailOrbiter, self)
	return _element_parents.get(element, self)


## Returns [param element]'s position offset relative to [method _get_element_kinematic_parent]
## in unrotated driver axes.
func _get_element_kinematic_local_offset(element: JrOrreryElement) -> Vector3:
	if element == null or element == self:
		return Vector3.ZERO
	if element is JrBallisticOrbiter:
		return (element as JrBallisticOrbiter).get_anchor_relative_position()
	return element.get_focus_relative_position()


## Returns [param element]'s velocity vector relative to [method _get_element_kinematic_parent]
## in unrotated driver axes.
func _get_element_kinematic_local_velocity(element: JrOrreryElement) -> Vector3:
	if element == null or element == self:
		return Vector3.ZERO
	if element is JrBallisticOrbiter:
		return (element as JrBallisticOrbiter).get_anchor_relative_velocity()
	return element.get_relative_velocity()


## Constructs a [JrBallisticForecastConfig] for [param bodies] starting at [param start_time].
func _build_forecast_config(
		bodies: Array[JrBallisticOrbiter],
		start_time: float
) -> JrBallisticForecastConfig:
	var cfg := JrBallisticForecastConfig.new()
	cfg.bodies = bodies
	cfg.start_time = start_time
	cfg.driver_gravity_strength = _gravity_strength
	cfg.tidal_cull_threshold = tidal_cull_threshold
	cfg.driver_element = self
	cfg.reference_plane = _reference_plane
	cfg.orbit_sequence = _orbit_sequence
	cfg.orbiter_parents = _children
	return cfg


## Converts an anchor-relative forecast polyline [param pts4] (`xyz = r_rel, w = anchor_idx`)
## into coordinates relative to [param ref_elem] at each preview step using LCA offsets.
func _convert_anchor_forecast_to_element_relative(
		ballistic: JrBallisticOrbiter,
		pts4: PackedVector4Array,
		well_list: Array[JrOrreryElement],
		ref_elem: JrOrreryElement
) -> PackedVector3Array:
	var count: int = pts4.size()
	if count == 0:
		return PackedVector3Array()

	var target_ref: JrOrreryElement = (
			ref_elem
			if (ref_elem != null and _is_tracked_or_self(ref_elem))
			else self
	)
	var step_dt: float = ballistic.preview_duration / float(maxi(1, count - 1))
	var rel_pts: PackedVector3Array = []
	rel_pts.resize(count)

	for k: int in range(count):
		var pt4: Vector4 = pts4[k]
		var anchor_idx: int = clampi(int(roundf(pt4.w)), 0, maxi(0, well_list.size() - 1))
		var anchor_elem: JrOrreryElement = (
				well_list[anchor_idx]
				if not well_list.is_empty()
				else self
		)
		var lca_offset: Vector3 = _compute_lca_well_position_difference_at(
				anchor_elem,
				target_ref,
				float(k) * step_dt
		)
		rel_pts[k] = lca_offset + Vector3(pt4.x, pt4.y, pt4.z)

	return rel_pts


## Computes batched forward-looking anchor-relative trajectories and transforms them into
## the active [member perspective_node] frame for all tracked [JrBallisticOrbiter] bodies.
func _update_runtime_ballistic_forecasts(system_basis: Basis) -> void:
	if _ballistic_sequence.is_empty():
		return;

	if not enable_ballistic_forecasts:
		if _any_ballistic_has_forecast_points:
			for ballistic: JrBallisticOrbiter in _ballistic_sequence:
				if ballistic.has_runtime_forecast_points():
					ballistic.set_runtime_forecast_points(
							PackedVector3Array(),
							PackedVector3Array(),
							PackedVector3Array()
					);
			_any_ballistic_has_forecast_points = false;
		return;

	var targets: Array[JrBallisticOrbiter] = _collect_forecast_targets();
	if targets.is_empty():
		if _any_ballistic_has_forecast_points:
			for ballistic: JrBallisticOrbiter in _ballistic_sequence:
				if ballistic.has_runtime_forecast_points():
					ballistic.set_runtime_forecast_points(
							PackedVector3Array(),
							PackedVector3Array(),
							PackedVector3Array()
					);
			_any_ballistic_has_forecast_points = false;
		return;

	var anchor_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			_resolve_anchor_relative_forecasts(targets)
	);
	_begin_forecast_projection_pass(system_basis);
	for target: JrBallisticOrbiter in targets:
		var pts4: PackedVector4Array = anchor_map.get(target, PackedVector4Array());
		_project_forecast_to_perspective(target, pts4, system_basis);


## Executes [method _update_runtime_ballistic_forecasts] while recording perspective projection
## microsecond durations into [member _telemetry].
func _update_runtime_ballistic_forecasts_instrumented(system_basis: Basis) -> void:
	if _ballistic_sequence.is_empty():
		return;

	if not enable_ballistic_forecasts:
		if _any_ballistic_has_forecast_points:
			var t_clear0: int = Time.get_ticks_usec();
			for ballistic: JrBallisticOrbiter in _ballistic_sequence:
				if ballistic.has_runtime_forecast_points():
					ballistic.set_runtime_forecast_points(
							PackedVector3Array(),
							PackedVector3Array(),
							PackedVector3Array()
					);
			_any_ballistic_has_forecast_points = false;
			_telemetry.add_phase_usec(
					JrOrreryTelemetry.PHASE_FORECAST_PERSPECTIVE_PROJECT,
					Time.get_ticks_usec() - t_clear0
			);
		return;

	var targets: Array[JrBallisticOrbiter] = _collect_forecast_targets();
	if targets.is_empty():
		if _any_ballistic_has_forecast_points:
			var t_empty0: int = Time.get_ticks_usec();
			for ballistic: JrBallisticOrbiter in _ballistic_sequence:
				if ballistic.has_runtime_forecast_points():
					ballistic.set_runtime_forecast_points(
							PackedVector3Array(),
							PackedVector3Array(),
							PackedVector3Array()
					);
			_any_ballistic_has_forecast_points = false;
			_telemetry.add_phase_usec(
					JrOrreryTelemetry.PHASE_FORECAST_PERSPECTIVE_PROJECT,
					Time.get_ticks_usec() - t_empty0
			);
		return;

	var anchor_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			_resolve_anchor_relative_forecasts(targets)
	);
	var t_proj0: int = Time.get_ticks_usec();
	_begin_forecast_projection_pass(system_basis);
	for target: JrBallisticOrbiter in targets:
		var pts4: PackedVector4Array = anchor_map.get(target, PackedVector4Array());
		_project_forecast_to_perspective(target, pts4, system_basis);
	_telemetry.add_phase_usec(
			JrOrreryTelemetry.PHASE_FORECAST_PERSPECTIVE_PROJECT,
			Time.get_ticks_usec() - t_proj0
	);


## Collects all [JrBallisticOrbiter] nodes requiring forecast evaluation in `O(K_forecast)`.
func _collect_forecast_targets() -> Array[JrBallisticOrbiter]:
	return _forecast_enabled_ballistics;


## Re-integrates anchor-relative forecasts on the GPU/CPU when any target is dirty, or
## reuses cached anchor-relative forecasts when only perspective settings changed.
func _resolve_anchor_relative_forecasts(
		targets: Array[JrBallisticOrbiter]
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var any_dirty: bool = _last_forecast_well_list.is_empty();
	if not any_dirty:
		for target: JrBallisticOrbiter in targets:
			if target.is_forecast_dirty() or not _last_forecast_anchor_points.has(target):
				any_dirty = true;
				break;

	if any_dirty:
		var cfg: JrBallisticForecastConfig = _build_forecast_config(targets, time);
		_last_forecast_well_list = _ballistic_solver.get_forecast_well_elements(cfg);
		_last_forecast_anchor_points = _ballistic_solver.compute_forecasts_anchor_relative(cfg);

	return _last_forecast_anchor_points;


## Initializes per-frame common-ancestor metadata for all forecast gravity wells so
## [method _project_forecast_to_perspective] can share timeline evaluations across bodies.
func _begin_forecast_projection_pass(system_basis: Basis) -> void:
	_fc_proj_grid_key = Vector2(-1.0, -1.0);
	_fc_proj_synodic_bases.clear();

	var well_count: int = maxi(1, _last_forecast_well_list.size());
	_fc_proj_well_common.resize(well_count);
	_fc_proj_well_common_now.resize(well_count);
	_fc_proj_well_sys_common_now.resize(well_count);
	_fc_proj_well_is_synodic.resize(well_count);
	_fc_proj_well_common_id.resize(well_count);
	_fc_proj_well_origins.resize(well_count);
	_fc_proj_well_offsets.resize(well_count);
	_fc_proj_well_origins_ready.resize(well_count);
	_fc_proj_well_origins_ready.fill(0);
	_fc_proj_well_offsets_ready.resize(well_count);
	_fc_proj_well_offsets_ready.fill(0);

	var selected_well: JrOrreryElement = _get_perspective_gravity_well();
	var is_heliocentric: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	var is_synodic: bool = (
			not is_heliocentric
			and perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
	);

	for w: int in range(well_count):
		var anchor_elem: JrOrreryElement = _get_forecast_well_element(w);
		var common_well: JrOrreryElement = _find_common_ancestor_well(anchor_elem, selected_well);
		var common_now: Vector3 = _compute_lca_position_difference(common_well, perspective_node);
		var apply_syn: bool = is_synodic and _should_apply_synodic_rotation_to_well(common_well);
		_fc_proj_well_common[w] = common_well;
		_fc_proj_well_common_now[w] = common_now;
		_fc_proj_well_sys_common_now[w] = system_basis * common_now;
		_fc_proj_well_is_synodic[w] = 1 if apply_syn else 0;
		_fc_proj_well_common_id[w] = int(common_well.get_instance_id());


## Ensures shared synodic inverse bases are populated for `(count, step_dt)` when [param need_synodic]
## is active, invalidating cached well step arrays only if the grid stride changed.
func _ensure_forecast_projection_grid(
		count: int,
		step_dt: float,
		need_synodic: bool
) -> void:
	var key := Vector2(float(count), step_dt);
	if _fc_proj_grid_key != key:
		_fc_proj_grid_key = key;
		_fc_proj_synodic_bases.clear();
		_fc_proj_well_origins_ready.fill(0);
		_fc_proj_well_offsets_ready.fill(0);

	if need_synodic and _fc_proj_synodic_bases.size() != count:
		_fc_proj_synodic_bases.resize(count);
		for k: int in range(count):
			var eval_t: float = time + float(k) * step_dt;
			_fc_proj_synodic_bases[k] = _compute_synodic_perspective_basis_at(eval_t);


## Lazily evaluates and caches inertial origins and LCA offsets across `count` forecast steps
## for [param well_idx].
func _ensure_forecast_well_arrays(well_idx: int, count: int, step_dt: float) -> void:
	var anchor_elem: JrOrreryElement = _get_forecast_well_element(well_idx);
	if _fc_proj_well_origins_ready[well_idx] == 0:
		var origins := PackedVector3Array();
		origins.resize(count);
		if anchor_elem != null and anchor_elem != self:
			for k: int in range(count):
				var eval_t: float = time + float(k) * step_dt;
				origins[k] = evaluate_element_driver_origin_at(anchor_elem, eval_t);
		_fc_proj_well_origins[well_idx] = origins;
		_fc_proj_well_origins_ready[well_idx] = 1;

	if _fc_proj_well_offsets_ready[well_idx] == 0:
		var common_well: JrOrreryElement = _fc_proj_well_common[well_idx];
		if anchor_elem == common_well:
			var zero_offsets := PackedVector3Array();
			zero_offsets.resize(count);
			_fc_proj_well_offsets[well_idx] = zero_offsets;
		elif common_well == self:
			_fc_proj_well_offsets[well_idx] = _fc_proj_well_origins[well_idx];
		else:
			var offsets := PackedVector3Array();
			offsets.resize(count);
			for k: int in range(count):
				var eval_t: float = time + float(k) * step_dt;
				offsets[k] = _compute_lca_well_position_difference_at(
						anchor_elem,
						common_well,
						eval_t
				);
			_fc_proj_well_offsets[well_idx] = offsets;
		_fc_proj_well_offsets_ready[well_idx] = 1;


## Projects [param pts4] at each forecast step into the reference frame of the
## earliest (lowest) common ancestor of the step's SOI anchor and [member perspective_node].
func _project_forecast_to_perspective(
		ballistic: JrBallisticOrbiter,
		pts4: PackedVector4Array,
		system_basis: Basis
) -> void:
	var count: int = pts4.size();
	if count == 0:
		if ballistic.has_runtime_forecast_points():
			ballistic.set_runtime_forecast_points(
					PackedVector3Array(),
					PackedVector3Array(),
					PackedVector3Array()
			);
		return;

	if _fc_proj_well_common.is_empty():
		_begin_forecast_projection_pass(system_basis);

	var well_limit: int = maxi(0, _fc_proj_well_common.size() - 1);
	var step_dt: float = ballistic.preview_duration / float(maxi(1, count - 1));
	var is_heliocentric: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	var identity_basis: bool = (
			system_basis == Basis.IDENTITY or system_basis.is_equal_approx(Basis.IDENTITY)
	);
	var is_synodic: bool = (
			not is_heliocentric
			and perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
	);
	_ensure_forecast_projection_grid(count, step_dt, is_synodic);

	var inertial_pts := PackedVector3Array();
	var ref_ids := PackedInt64Array();
	inertial_pts.resize(count);
	ref_ids.resize(count);

	var last_well_idx: int = -1;
	var curr_origins := PackedVector3Array();
	var curr_offsets := PackedVector3Array();
	var curr_common_now := Vector3.ZERO;
	var curr_sys_common_now := Vector3.ZERO;
	var curr_synodic: bool = false;
	var curr_common_id: int = 0;

	if is_heliocentric and identity_basis:
		for k: int in range(count):
			var pt4: Vector4 = pts4[k];
			var well_idx: int = clampi(int(roundf(pt4.w)), 0, well_limit);
			if well_idx != last_well_idx:
				last_well_idx = well_idx;
				_ensure_forecast_well_arrays(well_idx, count, step_dt);
				curr_origins = _fc_proj_well_origins[well_idx];
				curr_common_id = _fc_proj_well_common_id[well_idx];
			inertial_pts[k] = curr_origins[k] + Vector3(pt4.x, pt4.y, pt4.z);
			ref_ids[k] = curr_common_id;
		ballistic.set_runtime_forecast_points(inertial_pts, inertial_pts, inertial_pts, ref_ids);
		_any_ballistic_has_forecast_points = true;
		return;

	var synodic_now_basis: Basis = system_basis.transposed() if is_synodic else Basis.IDENTITY;
	var unrotated_pts := PackedVector3Array();
	var system_pts := PackedVector3Array();
	unrotated_pts.resize(count);
	system_pts.resize(count);

	for k: int in range(count):
		var pt4: Vector4 = pts4[k];
		var well_idx: int = clampi(int(roundf(pt4.w)), 0, well_limit);
		if well_idx != last_well_idx:
			last_well_idx = well_idx;
			_ensure_forecast_well_arrays(well_idx, count, step_dt);
			curr_origins = _fc_proj_well_origins[well_idx];
			curr_offsets = _fc_proj_well_offsets[well_idx];
			curr_common_now = _fc_proj_well_common_now[well_idx];
			curr_sys_common_now = _fc_proj_well_sys_common_now[well_idx];
			curr_synodic = (_fc_proj_well_is_synodic[well_idx] != 0);
			curr_common_id = _fc_proj_well_common_id[well_idx];

		var r_rel := Vector3(pt4.x, pt4.y, pt4.z);
		inertial_pts[k] = curr_origins[k] + r_rel;
		ref_ids[k] = curr_common_id;

		var offset_from_common_at_eval: Vector3 = curr_offsets[k] + r_rel;
		if curr_synodic:
			var synodic_step_inv: Basis = _fc_proj_synodic_bases[k];
			var synodic_pt: Vector3 = (
					curr_sys_common_now + (synodic_step_inv * offset_from_common_at_eval)
			);
			system_pts[k] = synodic_pt;
			unrotated_pts[k] = synodic_now_basis * synodic_pt;
		else:
			var unrot_pt: Vector3 = curr_common_now + offset_from_common_at_eval;
			unrotated_pts[k] = unrot_pt;
			system_pts[k] = unrot_pt if identity_basis else (system_basis * unrot_pt);

	ballistic.set_runtime_forecast_points(inertial_pts, unrotated_pts, system_pts, ref_ids);
	_any_ballistic_has_forecast_points = true;


## Returns the gravity-well element ([JrOrbitDriver] or [JrRailOrbiter]) associated
## with [member perspective_node].
func _get_perspective_gravity_well() -> JrOrreryElement:
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return self;
	if perspective_node is JrRailOrbiter:
		return perspective_node;
	if perspective_node is JrBallisticOrbiter:
		var b_node := perspective_node as JrBallisticOrbiter;
		var b_anchor: JrOrreryElement = b_node.get_anchor_element();
		if b_anchor != null and _is_tracked_or_self(b_anchor):
			return b_anchor;
		return _resolve_ballistic_spawn_parent(b_node);
	return self;


## Returns the lowest (earliest) common ancestor gravity well of [param well_a]
## and [param well_b] in this driver's orbital hierarchy.
func _find_common_ancestor_well(
		well_a: JrOrreryElement,
		well_b: JrOrreryElement
) -> JrOrreryElement:
	var target_a: JrOrreryElement = (
			well_a
			if (well_a != null and _is_tracked_or_self(well_a))
			else self
	);
	var target_b: JrOrreryElement = (
			well_b
			if (well_b != null and _is_tracked_or_self(well_b))
			else self
	);
	if target_a == self or target_b == self:
		return self;
	if target_a == target_b:
		return target_a;

	_lca_b_elems.clear();
	var curr_b: JrOrreryElement = target_b;
	while curr_b != null:
		_lca_b_elems.append(curr_b);
		if curr_b == self:
			break;
		curr_b = _get_element_kinematic_parent(curr_b);

	var curr_a: JrOrreryElement = target_a;
	while curr_a != null:
		if _lca_b_elems.has(curr_a):
			return curr_a;
		if curr_a == self:
			break;
		curr_a = _get_element_kinematic_parent(curr_a);

	return self;


## Returns [code]true[/code] when [param common_well] lies within the subtree of
## [member perspective_node]'s parent gravity well for synodic frame projection.
func _should_apply_synodic_rotation_to_well(common_well: JrOrreryElement) -> bool:
	if common_well == null or perspective_node == null or perspective_node == self:
		return false
	var synodic_primary: JrOrreryElement = _get_element_kinematic_parent(perspective_node)
	if synodic_primary == null:
		return false
	if common_well == synodic_primary:
		return true
	return _find_common_ancestor_well(common_well, synodic_primary) == synodic_primary


## Returns the [JrOrreryElement] at [param well_idx] in [member _last_forecast_well_list].
func _get_forecast_well_element(well_idx: int) -> JrOrreryElement:
	if _last_forecast_well_list.is_empty():
		return self
	var clamped: int = clampi(well_idx, 0, _last_forecast_well_list.size() - 1)
	return _last_forecast_well_list[clamped]


## Returns [code]true[/code] if [param element] is either this driver itself or a tracked
## descendant.
func _is_tracked_or_self(element: JrOrreryElement) -> bool:
	return element != null and (element._owning_driver == self or element == self);


## Updates the generated [JrReferenceDirection] from the exported properties
## and triggers an orbital update.
func _sync_reference_direction() -> void:
	_reference_plane.configure(
			reference_rotation,
			euler_order,
			positive_is_clockwise,
			lock_to_ecliptic
	)
	if is_inside_tree():
		update_orbits()
	orbit_visuals_changed.emit()


## Resolves the original spawn parent [JrOrreryElement] for [param ballistic] (preserving
## its orbital focus even after [method collect_ballistic_nodes] reparents it to this driver).
func _resolve_ballistic_spawn_parent(ballistic: JrBallisticOrbiter) -> JrOrreryElement:
	var spawn_p: JrOrreryElement = ballistic.get_spawn_parent()
	if spawn_p != null and _is_tracked_or_self(spawn_p):
		return spawn_p
	return _element_parents.get(ballistic, self)


## Returns the heliocentric driver-relative [Transform3D] of [param element], or
## [constant Transform3D.IDENTITY] when [param element] is this driver.
func _get_element_driver_transform(element: JrOrreryElement) -> Transform3D:
	if element != null and element != self:
		return element.get_driver_relative_transform()
	return Transform3D.IDENTITY


## Returns the global inertial velocity of [param element], or [constant Vector3.ZERO]
## when [param element] is this driver.
func _get_element_global_velocity(element: JrOrreryElement) -> Vector3:
	if element != null and element != self:
		return element.get_global_velocity()
	return Vector3.ZERO


## Returns the heliocentric driver-relative [Transform3D] of [param element]'s
## immediate parent, or [constant Transform3D.IDENTITY] when the parent is this driver.
func _get_parent_driver_transform_for(element: JrOrreryElement) -> Transform3D:
	var parent_elem: JrOrreryElement = _element_parents.get(element, null)
	return _get_element_driver_transform(parent_elem)


## Returns the accumulated global inertial velocity vector of [param element]'s
## immediate parent in this driver's reference frame, or [constant Vector3.ZERO]
## when the parent is this driver.
func _get_parent_global_velocity_for(element: JrOrreryElement) -> Vector3:
	var parent_elem: JrOrreryElement = _element_parents.get(element, null)
	return _get_element_global_velocity(parent_elem)


## Computes the active perspective rotation [Basis] at the current simulation [member time].
func _compute_perspective_basis() -> Basis:
	if Engine.is_editor_hint():
		return Basis.IDENTITY
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return Basis.IDENTITY

	if perspective_rotation_mode == PerspectiveRotationMode.INERTIAL:
		return Basis.IDENTITY
	if perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL:
		return _compute_synodic_perspective_basis_at(time)

	var target_transform: Transform3D = perspective_node.get_driver_relative_transform()
	return target_transform.basis.transposed().orthonormalized()


## Computes the inverse synodic orbital [Basis] at [param eval_time] that locks the
## primary-to-[member perspective_node] radius vector along the positive X-axis and the
## orbital angular momentum along the ecliptic north pole (+Y).
func _compute_synodic_perspective_basis_at(eval_time: float) -> Basis:
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return Basis.IDENTITY

	var r_orb := Vector3.ZERO
	var v_orb := Vector3.ZERO

	if perspective_node is JrRailOrbiter:
		var orb := perspective_node as JrRailOrbiter
		if is_equal_approx(eval_time, time):
			r_orb = orb.get_focus_relative_position()
			v_orb = orb.get_relative_velocity()
		else:
			r_orb = orb.get_relative_position(eval_time, _reference_plane)
			v_orb = orb.get_relative_velocity(eval_time, _reference_plane)
	elif perspective_node is JrBallisticOrbiter:
		var b_node := perspective_node as JrBallisticOrbiter
		r_orb = b_node.get_anchor_relative_position()
		v_orb = b_node.get_anchor_relative_velocity()

	var r_len: float = r_orb.length()
	if r_len <= 1e-8:
		return Basis.IDENTITY

	var z_back: Vector3 = -r_orb / r_len
	var ref_up: Vector3 = (
			_reference_plane.ecliptic_north_pole
			if _reference_plane != null
			else Vector3.UP
	)
	var h_vec: Vector3 = r_orb.cross(v_orb)
	var y_normal: Vector3 = h_vec.normalized() if h_vec.length_squared() > 1e-12 else ref_up
	if y_normal.dot(ref_up) < 0.0:
		y_normal = -y_normal
	if absf(z_back.dot(y_normal)) > 0.999:
		y_normal = Vector3.RIGHT if absf(z_back.y) > 0.9 else Vector3.UP
	var x_right: Vector3 = y_normal.cross(z_back).normalized()
	y_normal = z_back.cross(x_right).normalized()
	var synodic_frame := Basis(x_right, y_normal, z_back)
	return synodic_frame.transposed()


## Returns [code]true[/code] when automatic runtime time progression is active.
func _should_step_time() -> bool:
	return auto_advance and not Engine.is_editor_hint()


## Synchronizes the active simulation-playing flag across all tracked descendant elements.
func _sync_simulation_playing_state() -> void:
	_has_synced_playing = false;
	_sync_playing_if_changed(_should_step_time());


## Registers a newly added descendant simulation [param child] ([JrRailOrbiter] or
## [JrBallisticOrbiter]) and its [param parent] into the appropriate flat traversal sequence.
func _on_subtree_element_added(parent: JrOrreryElement, child: JrOrreryElement) -> void:
	if child == null or parent == null or child == self:
		return;

	_owning_driver = self;
	var already_tracked: bool = _element_parents.has(child);
	_element_parents[child] = parent;
	child._owning_driver = self;
	child._tracked_scene_parent = parent;
	_ballistic_anchors_dirty = true;
	_ballistic_solver.invalidate_resident_step_buffer();
	if child is JrRailOrbiter:
		var orbiter := child as JrRailOrbiter;
		var already_rail: bool = _children.has(orbiter);
		_children[orbiter] = parent;
		if not already_rail:
			_orbit_sequence.append(orbiter);
		if not orbiter.rail_spline_dirtied.is_connected(_on_rail_spline_dirtied):
			orbiter.rail_spline_dirtied.connect(_on_rail_spline_dirtied);
		_queue_dirty_rail_spline(orbiter);
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter;
		ballistic._fast_root_sync_ready = false;
		ballistic._direct_system_xf = _last_direct_system_xf_state;
		_queue_non_root_sync_ballistic(ballistic);
		if not already_tracked:
			ballistic._ballistic_seq_index = _ballistic_sequence.size();
			_ballistic_sequence.append(ballistic);
			if ballistic.runtime_forecast_enabled:
				_forecast_enabled_ballistics.append(ballistic);
		_pending_ballistic_init = true;
		if not ballistic.ballistic_runtime_dirtied.is_connected(_on_ballistic_runtime_dirtied):
			ballistic.ballistic_runtime_dirtied.connect(_on_ballistic_runtime_dirtied);
		if not ballistic.ballistic_gpu_dirtied.is_connected(_on_ballistic_gpu_dirtied):
			ballistic.ballistic_gpu_dirtied.connect(_on_ballistic_gpu_dirtied);
		if not ballistic.ballistic_sync_dirtied.is_connected(_on_ballistic_sync_dirtied):
			ballistic.ballistic_sync_dirtied.connect(_on_ballistic_sync_dirtied);
		if not ballistic.ballistic_forecast_toggle_changed.is_connected(
				_on_ballistic_forecast_toggle_changed
		):
			ballistic.ballistic_forecast_toggle_changed.connect(
					_on_ballistic_forecast_toggle_changed
			);

	_units_and_mu_dirty = true;
	_has_synced_playing = false;

	if not child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
		child.orbit_visuals_changed.connect(_on_descendant_orbit_visuals_changed);

	if is_inside_tree() and not _updating_orbits:
		update_orbits();


## Removes an exiting descendant simulation [param child] from the lookup tables
## and flat traversal sequences.
func _on_subtree_element_removed(_parent: JrOrreryElement, child: JrOrreryElement) -> void:
	if child == null:
		return;

	if child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
		child.orbit_visuals_changed.disconnect(_on_descendant_orbit_visuals_changed);

	_element_parents.erase(child);
	child._owning_driver = null;
	child._tracked_scene_parent = null;
	child._runtime_well_index = -1;
	_ballistic_solver.invalidate_resident_step_buffer();
	if child is JrRailOrbiter:
		var orbiter := child as JrRailOrbiter;
		orbiter._in_dirty_spline_queue = false;
		if orbiter.rail_spline_dirtied.is_connected(_on_rail_spline_dirtied):
			orbiter.rail_spline_dirtied.disconnect(_on_rail_spline_dirtied);
		_children.erase(orbiter);
		if (
				not _orbit_sequence.is_empty()
				and _orbit_sequence[_orbit_sequence.size() - 1] == orbiter
		):
			_orbit_sequence.pop_back();
		else:
			_orbit_sequence.erase(orbiter);
		_units_and_mu_dirty = true;
		_has_synced_playing = false;
		_ballistic_anchors_dirty = true;
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter;
		ballistic._fast_root_sync_ready = false;
		ballistic._in_non_root_sync_list = false;
		ballistic._ballistic_seq_index = -1;
		if ballistic.ballistic_runtime_dirtied.is_connected(_on_ballistic_runtime_dirtied):
			ballistic.ballistic_runtime_dirtied.disconnect(_on_ballistic_runtime_dirtied);
		if ballistic.ballistic_gpu_dirtied.is_connected(_on_ballistic_gpu_dirtied):
			ballistic.ballistic_gpu_dirtied.disconnect(_on_ballistic_gpu_dirtied);
		if ballistic.ballistic_sync_dirtied.is_connected(_on_ballistic_sync_dirtied):
			ballistic.ballistic_sync_dirtied.disconnect(_on_ballistic_sync_dirtied);
		if ballistic.ballistic_forecast_toggle_changed.is_connected(
				_on_ballistic_forecast_toggle_changed
		):
			ballistic.ballistic_forecast_toggle_changed.disconnect(
					_on_ballistic_forecast_toggle_changed
			);
		_remove_ballistic_from_sparse_list(_non_root_sync_ballistics, ballistic);
		_remove_ballistic_from_sparse_list(_forecast_enabled_ballistics, ballistic);
		if (
				not _ballistic_sequence.is_empty()
				and _ballistic_sequence[_ballistic_sequence.size() - 1] == ballistic
		):
			_ballistic_sequence.pop_back();
		else:
			_ballistic_sequence.erase(ballistic);
			for seq_idx: int in range(_ballistic_sequence.size()):
				_ballistic_sequence[seq_idx]._ballistic_seq_index = seq_idx;
		_last_forecast_anchor_points.erase(ballistic);
		if child == perspective_node and is_inside_tree() and not _updating_orbits:
			update_orbits();
		return;

	if child == perspective_node and is_inside_tree() and not _updating_orbits:
		update_orbits();
	else:
		_schedule_deferred_orbit_update();


## Marks ballistic initialization and pre-step anchor synchronization dirty when signaled upward.
func _on_ballistic_runtime_dirtied() -> void:
	_pending_ballistic_init = true;
	_ballistic_anchors_dirty = true;


## Queues [param orbiter] into [member _dirty_rail_splines] when [signal JrRailOrbiter.rail_spline_dirtied] fires.
func _on_rail_spline_dirtied(orbiter: JrRailOrbiter) -> void:
	_queue_dirty_rail_spline(orbiter);


## Queues [param orbiter] into [member _dirty_rail_splines] if it requires a B-spline bake.
func _queue_dirty_rail_spline(orbiter: JrRailOrbiter) -> void:
	if orbiter.needs_nu_spline_bake() and not orbiter._in_dirty_spline_queue:
		orbiter._in_dirty_spline_queue = true;
		_dirty_rail_splines.append(orbiter);


## Notifies [member _ballistic_solver] that [param ballistic] changed state on the CPU.
func _on_ballistic_gpu_dirtied(ballistic: JrBallisticOrbiter) -> void:
	_ballistic_solver.record_dirty_gpu_body(ballistic);


## Queues [param ballistic] into [member _non_root_sync_ballistics] when signaled upward.
func _on_ballistic_sync_dirtied(ballistic: JrBallisticOrbiter) -> void:
	_queue_non_root_sync_ballistic(ballistic);


## Appends [param ballistic] to [member _non_root_sync_ballistics] if not already queued.
func _queue_non_root_sync_ballistic(ballistic: JrBallisticOrbiter) -> void:
	if not ballistic._in_non_root_sync_list:
		ballistic._in_non_root_sync_list = true;
		_non_root_sync_ballistics.append(ballistic);


## Updates [member _forecast_enabled_ballistics] when [member JrBallisticOrbiter.runtime_forecast_enabled] changes.
func _on_ballistic_forecast_toggle_changed(ballistic: JrBallisticOrbiter) -> void:
	if ballistic.runtime_forecast_enabled:
		if not _forecast_enabled_ballistics.has(ballistic):
			_forecast_enabled_ballistics.append(ballistic);
	else:
		_remove_ballistic_from_sparse_list(_forecast_enabled_ballistics, ballistic);
		if ballistic.has_runtime_forecast_points():
			ballistic.set_runtime_forecast_points(
					PackedVector3Array(),
					PackedVector3Array(),
					PackedVector3Array()
			);


## Removes [param target] from [param list] with a fast `pop_back()` path duringLIFO teardown.
func _remove_ballistic_from_sparse_list(
		list: Array[JrBallisticOrbiter],
		target: JrBallisticOrbiter
) -> void:
	if list.is_empty():
		return;
	if list[list.size() - 1] == target:
		list.pop_back();
	else:
		list.erase(target);


## Schedules a single coalesced [method update_orbits] pass at the end of the frame.
func _schedule_deferred_orbit_update() -> void:
	if _deferred_orbit_update_queued or not is_inside_tree():
		return;
	_deferred_orbit_update_queued = true;
	call_deferred(&"_flush_deferred_orbit_update");


## Executes a coalesced [method update_orbits] pass if this driver is still inside the scene tree.
func _flush_deferred_orbit_update() -> void:
	_deferred_orbit_update_queued = false;
	if is_inside_tree() and not _updating_orbits:
		update_orbits();


## Re-evaluates system state when a descendant element's parameters change in
## the editor or at runtime.
func _on_descendant_orbit_visuals_changed() -> void:
	_units_and_mu_dirty = true;
	if is_inside_tree() and not _updating_orbits:
		update_orbits();
