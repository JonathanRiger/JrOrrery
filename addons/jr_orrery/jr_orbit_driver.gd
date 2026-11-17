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
const HILL_EXIT_MULTIPLIER: float = 1.05;

## Number of staggered trajectory-tape refill cohorts partitioning [member _ballistic_sequence].
const TAPE_COHORT_COUNT: int = 64;

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
			if is_inside_tree():
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
			if is_inside_tree():
				update_orbits();
			orbit_visuals_changed.emit();

## Dimensionless tidal ratio threshold [code](|a_tidal| / |a_dominant|)[/code] below which
## distant third-body gravitational perturbations are culled. Bodies with zero thrust and
## zero unculled tides automatically coast on Tier-1 frozen Keplerian conics.
@export_range(0.0, 0.1, 0.00001)
var tidal_cull_threshold: float = 0.0001:
	set(value):
		var clamped: float = maxf(0.0, value);
		if tidal_cull_threshold != clamped:
			tidal_cull_threshold = clamped;
			_invalidate_all_ballistic_forecasts();
			if is_inside_tree():
				update_orbits();

## Maximum number of adaptive Universal-Variable Kick-Drift-Kick substeps executed inside
## each GPU compute shader dispatch per simulation tick.
@export_range(1, 64, 1)
var ballistic_substeps: int = 1:
	set(value):
		var clamped: int = maxi(1, value);
		if ballistic_substeps != clamped:
			ballistic_substeps = clamped;
			invalidate_all_ballistic_trajectories();

## If [code]true[/code], continuously dispatches the GPU trajectory forecast compute
## shader at runtime to predict future paths for [JrBallisticOrbiter] bodies relative to
## the active [member perspective_node].
@export var enable_ballistic_forecasts: bool = true:
	set(value):
		if enable_ballistic_forecasts != value:
			enable_ballistic_forecasts = value;
			if enable_ballistic_forecasts:
				for ballistic: JrBallisticOrbiter in _forecast_enabled_ballistics:
					ballistic.invalidate_trajectory();
					_queue_dirty_trajectory_ballistic(ballistic);
			if is_inside_tree() and not Engine.is_editor_hint():
				update_orbits();

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

## Parallel array of [JrBallisticTrajectoryTape] instances matching [member _ballistic_sequence].
var _ballistic_tapes: Array[JrBallisticTrajectoryTape] = [];

## Shared 2-float `[eval_time, pass_id]` cell supplied downward to tracked [JrBallisticOrbiter] nodes.
var _sampled_time_cell: PackedFloat64Array = PackedFloat64Array([0.0, 0.0]);

## Shared single-element holder `[PackedVector3Array]` exposing GPU-sampled positions to tracked [JrBallisticOrbiter] nodes.
var _sampled_pos_holder: Array = [PackedVector3Array()];

## 64-bucket staggered refill cohort lists partitioning [member _ballistic_sequence].
var _tape_cohorts: Array[Array] = [
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
	[], [], [], [], [], [], [], [],
];

## Earliest [member JrBallisticOrbiter._next_refill_time] per cohort in [member _tape_cohorts].
var _cohort_min_refill: PackedFloat64Array = PackedFloat64Array([
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
	INF, INF, INF, INF, INF, INF, INF, INF,
]);

## Absolute `dt` of the most recent steady-state forward trajectory-tape scheduling pass.
var _last_schedule_abs_dt: float = 0.0;

## Sparse list of [JrBallisticOrbiter] nodes not currently in the root-parent/root-anchor fast sync state.
var _non_root_sync_ballistics: Array[JrBallisticOrbiter] = [];

## Preallocated batch job list reused across [_bake_scheduled_candidate_batch] calls.
var _batch_jobs: Array[JrBallisticOrbiter] = [];

## Preallocated SIMT coasting partition scratch buffer.
var _batch_coasting_buf: Array[JrBallisticOrbiter] = [];

## Preallocated SIMT perturbed partition scratch buffer.
var _batch_perturbed_buf: Array[JrBallisticOrbiter] = [];

## Preallocated per-job start timestamps reused across [_bake_scheduled_candidate_batch] calls.
var _batch_start_times := PackedFloat64Array();

## Preallocated per-job knot intervals reused across [_bake_scheduled_candidate_batch] calls.
var _batch_knot_dts := PackedFloat32Array();

## Preallocated per-job step counts reused across [_bake_scheduled_candidate_batch] calls.
var _batch_step_counts := PackedInt32Array();

## Preallocated per-job substep counts reused across [_bake_scheduled_candidate_batch] calls.
var _batch_substeps := PackedInt32Array();

## Preallocated per-job append modes reused across [_bake_scheduled_candidate_batch] calls.
var _batch_modes := PackedByteArray();

## Preallocated per-job loop periods reused across [_bake_scheduled_candidate_batch] calls.
var _batch_loop_periods := PackedFloat64Array();

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

## Sparse queue of [JrBallisticOrbiter] nodes whose rolling trajectory tapes require a full bake.
var _dirty_trajectory_ballistics: Array[JrBallisticOrbiter] = [];

## Minimum simulation timestamp at which any clean tracked [JrBallisticOrbiter] requires a tail refill.
var _earliest_tape_refill_time: float = -INF;

## Maximum `start_time` across all valid non-periodic [JrBallisticTrajectoryTape] instances.
var _latest_tape_start_time: float = -INF;

## Dirty flag indicating [member _runtime_well_elements] and safe radii need rebuilding.
var _runtime_wells_dirty: bool = true;

## Cached ordered gravity-well elements (`[self, ...valid JrRailOrbiters]`).
var _runtime_well_elements: Array[JrOrreryElement] = [];

## Per-well worst-case analytical periapsis tidal safe radius for all-time periodic loop detection.
var _well_all_time_safe_radii: PackedFloat32Array = PackedFloat32Array();

## Per-well list of direct valid child [JrRailOrbiter] nodes with `mu > 0.0`.
var _massive_child_orbiters_by_well: Array[Array] = [];

## Monotonic epoch incremented when perspective target, perspective anchor, or rotation mode changes.
var _persp_cache_epoch: int = 1;

## Instance ID of the active [member perspective_node] on the previous forecast projection pass.
var _last_persp_target_id: int = 0;

## Instance ID of [method _get_perspective_gravity_well] on the previous forecast projection pass.
var _last_persp_well_id: int = 0;

## Active effective [enum PerspectiveRotationMode] on the previous forecast projection pass.
var _last_persp_rot_mode: int = -1;

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

## Start timestamp of the shared per-pass forecast perspective projection step grid.
var _fc_grid_start_time: float = NAN;

## Knot time step of the shared per-pass forecast perspective projection step grid.
var _fc_grid_knot_dt: float = -1.0;

## Scratch inertial tip position computed by [method _evaluate_tape_tip_points].
var _tip_inertial_pt: Vector3 = Vector3.ZERO;

## Scratch unrotated perspective tip position computed by [method _evaluate_tape_tip_points].
var _tip_unrot_pt: Vector3 = Vector3.ZERO;

## Scratch system-frame tip position computed by [method _evaluate_tape_tip_points].
var _tip_sys_pt: Vector3 = Vector3.ZERO;

## Scratch common-ancestor well instance ID computed by [method _evaluate_tape_tip_points].
var _tip_common_id: int = 0;

## Cached synodic inverse rotation bases across the active forecast grid.
var _fc_proj_synodic_bases: Array[Basis] = [];

## Pending batch jobs awaiting deferred finalization after an asynchronous GPU bake+sample pass.
var _pending_gpu_finalize_jobs: Array[JrBallisticOrbiter] = [];

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

## Sparse list of [JrBallisticOrbiter] nodes that have non-empty [member JrBallisticOrbiter.scheduled_burns].
var _burn_scheduled_ballistics: Array[JrBallisticOrbiter] = [];


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
		_runtime_wells_dirty = true;
		_persp_cache_epoch += 1;
		_pending_ballistic_init = true;
		_ballistic_anchors_dirty = true;
		if not _ballistic_sequence.is_empty():
			_rebuild_ballistic_tape_cohorts();
			_recompute_tape_schedule_bounds();
		_ballistic_solver.invalidate_resident_step_buffer();
		_invalidate_solver_tape_atlas_safely();
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
	transform_updated.emit();

	if _persp_is_heliocentric and is_identity_basis:
		var r_seq: Array[JrRailOrbiter] = _orbit_sequence;
		for r_idx: int in range(r_seq.size()):
			var orbiter: JrRailOrbiter = r_seq[r_idx];
			orbiter._system_transform = orbiter._driver_relative_transform;
			orbiter.transform_updated.emit();
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
			orbiter.transform_updated.emit();

	if not Engine.is_editor_hint():
		_update_runtime_ballistic_forecasts(persp_basis);

	var use_direct_ballistic_xf: bool = _persp_is_heliocentric and is_identity_basis;
	_sync_ballistic_direct_system_xf_flag(use_direct_ballistic_xf);
	if use_direct_ballistic_xf:
		if not _ballistic_sequence.is_empty() and (
				not _non_root_sync_ballistics.is_empty()
				or _ballistic_sequence[0].has_connections(&"transform_updated")
		):
			_flush_pending_gpu_tape_sample();
		for nr_idx: int in range(_non_root_sync_ballistics.size()):
			var nr_b: JrBallisticOrbiter = _non_root_sync_ballistics[nr_idx];
			nr_b._system_transform = nr_b._driver_relative_transform;
		for i: int in range(_ballistic_sequence.size()):
			_ballistic_sequence[i].transform_updated.emit();
		_flush_pending_gpu_tape_sample();
	else:
		_flush_pending_gpu_tape_sample();
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
			ballistic.transform_updated.emit();

	if Engine.is_editor_hint():
		_update_editor_ghosts();

	_updating_orbits = false;


## Evaluates the exact analytical inertial driver origin of a tracked [param element]
## at arbitrary simulation timestamp [param eval_time] without mutating live transforms.
func evaluate_element_driver_origin_at(element: JrOrreryElement, eval_time: float) -> Vector3:
	if element == null or element == self or not _element_parents.has(element):
		return Vector3.ZERO;
	if element is JrBallisticOrbiter:
		var b_node := element as JrBallisticOrbiter;
		if not is_equal_approx(eval_time, time):
			var b_tape: JrBallisticTrajectoryTape = b_node.get_trajectory_tape();
			if b_tape.contains_time(eval_time):
				var sampled4: Vector4 = b_node.sample_tape_pos_and_well_at(eval_time);
				if sampled4.w >= -0.5:
					var w_idx: int = int(roundf(sampled4.w));
					var w_elem: JrOrreryElement = _get_forecast_well_element(w_idx);
					return (
							evaluate_element_driver_origin_at(w_elem, eval_time)
							+ Vector3(sampled4.x, sampled4.y, sampled4.z)
					);
		var anchor: JrOrreryElement = b_node.get_anchor_element();
		if anchor != null and _is_tracked_or_self(anchor):
			return (
					evaluate_element_driver_origin_at(anchor, eval_time)
					+ b_node.get_anchor_relative_position()
			);
		return b_node.get_driver_position();

	var origin := Vector3.ZERO;
	var curr: JrOrreryElement = element;
	while curr is JrRailOrbiter and _children.has(curr as JrRailOrbiter):
		var orb := curr as JrRailOrbiter;
		if orb.is_orbit_valid():
			origin += orb.get_relative_position(eval_time, _reference_plane);
		curr = _children.get(orb, null);
	return origin;


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


## Invalidates the rolling trajectory-horizon tapes of all tracked [JrBallisticOrbiter]
## descendants so they recompute their full forward trajectories on the next update pass.
func invalidate_all_ballistic_trajectories() -> void:
	_runtime_wells_dirty = true;
	_persp_cache_epoch += 1;
	_earliest_tape_refill_time = -INF;
	_last_schedule_abs_dt = 0.0;
	_reset_cohort_min_refill(-INF);
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.invalidate_trajectory();
		_queue_dirty_trajectory_ballistic(ballistic);
	_ballistic_solver.invalidate_tape_atlas();


## Materializes any GPU-atlas-only tapes back to CPU memory before invalidating the solver atlas.
func _invalidate_solver_tape_atlas_safely() -> void:
	if _ballistic_solver.has_valid_tape_atlas() and not _ballistic_tapes.is_empty():
		_ballistic_solver.materialize_all_tapes_from_gpu_atlas(_ballistic_tapes);
	_ballistic_solver.invalidate_tape_atlas();


## Resets all [constant TAPE_COHORT_COUNT] entries in [member _cohort_min_refill] to [param fill_val].
func _reset_cohort_min_refill(fill_val: float) -> void:
	_cohort_min_refill.resize(TAPE_COHORT_COUNT);
	_cohort_min_refill.fill(fill_val);


## Returns the minimum refill timestamp across all [constant TAPE_COHORT_COUNT] entries in [member _cohort_min_refill].
func _min_across_cohort_refills() -> float:
	var min_t: float = INF;
	for c: int in range(TAPE_COHORT_COUNT):
		min_t = minf(min_t, _cohort_min_refill[c]);
	return min_t;


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
	if _ballistic_solver.is_force_cpu() != force_cpu:
		_ballistic_solver.set_force_cpu(force_cpu);
		invalidate_all_ballistic_trajectories();


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
	_runtime_wells_dirty = true;
	_persp_cache_epoch += 1;
	_has_synced_playing = false;
	_pending_ballistic_init = true;
	_ballistic_anchors_dirty = true;
	_ballistic_solver.invalidate_resident_step_buffer();
	_ballistic_solver.invalidate_tape_atlas();
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


## Marks all tracked [JrBallisticOrbiter] trajectories and forecasts as dirty when system gravity or units change.
func _invalidate_all_ballistic_forecasts() -> void:
	invalidate_all_ballistic_trajectories();


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
		return;

	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		_initialize_single_ballistic(ballistic, playing);
		_sync_single_ballistic_transforms(ballistic);

	var forecast_cfg: JrBallisticForecastConfig = _build_forecast_config(_ballistic_sequence, 0.0);
	var well_list: Array[JrOrreryElement] = _ballistic_solver.get_forecast_well_elements(
			forecast_cfg
	);
	var editor_rel_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			_ballistic_solver.compute_forecasts_anchor_relative(forecast_cfg)
	);

	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var pts4: PackedVector4Array = editor_rel_map.get(ballistic, PackedVector4Array());
		var parent_elem: JrOrreryElement = _element_parents.get(ballistic, self);
		var rel_pts: PackedVector3Array = _convert_anchor_forecast_to_element_relative(
				ballistic,
				pts4,
				well_list,
				parent_elem
		);
		ballistic.set_editor_preview_relative_points(rel_pts);


## Evaluates runtime Keplerian states, maintains rolling Cubic Hermite trajectory-horizon
## tapes for dirty or expiring [JrBallisticOrbiter] cohorts, and samples all ballistic
## positions and velocities in `O(1)`.
func _evaluate_runtime_system_state(playing: bool) -> void:
	var dt: float = time - _last_stepped_time if _runtime_initialized else 0.0;
	var prev_local_positions: Dictionary[JrRailOrbiter, Vector3] = {};

	if not _runtime_initialized:
		_evaluate_all_orbiters_at(time, playing);
		if _pending_ballistic_init:
			_initialize_pending_ballistics(playing);
		if not _ballistic_sequence.is_empty():
			if _ballistic_anchors_dirty:
				_update_ballistic_anchors_before_step();
			if _runtime_wells_dirty:
				_rebuild_runtime_well_cache();
			if enable_ballistic_forecasts and not _forecast_enabled_ballistics.is_empty():
				_schedule_and_bake_trajectory_tapes(0.0, prev_local_positions);
		_sync_all_ballistics_transforms(playing);
		_runtime_initialized = true;
		_last_stepped_time = time;
		return;

	if _pending_ballistic_init:
		_initialize_pending_ballistics(playing);
	if (
			dt < 0.0
			and not _ballistic_sequence.is_empty()
			and (time < _latest_tape_start_time or not _dirty_trajectory_ballistics.is_empty())
	):
		prev_local_positions = _capture_orbiter_local_positions();

	_evaluate_all_orbiters_at(time, playing);

	if not _ballistic_sequence.is_empty():
		if _ballistic_anchors_dirty:
			_update_ballistic_anchors_before_step();
		if _runtime_wells_dirty:
			_rebuild_runtime_well_cache();

		if is_zero_approx(dt):
			if (
					enable_ballistic_forecasts
					and not _dirty_trajectory_ballistics.is_empty()
					and not _forecast_enabled_ballistics.is_empty()
			):
				_schedule_and_bake_trajectory_tapes(0.0, prev_local_positions);
		else:
			var sampled_in_bake: bool = false;
			if (
					not _dirty_trajectory_ballistics.is_empty()
					or time >= _earliest_tape_refill_time
					or time < _latest_tape_start_time
			):
				var want_fuse: bool = (
						_ballistic_sequence.size() >= JrBallisticComputeSolver.MIN_GPU_TAPE_ATLAS_BODIES
						and (_reference_plane == null or not _reference_plane.lock_to_ecliptic)
						and _ballistic_solver.is_tape_sample_gpu_active()
				);
				sampled_in_bake = _schedule_and_bake_trajectory_tapes(
						dt,
						prev_local_positions,
						want_fuse
				);
			if not sampled_in_bake:
				_sample_all_ballistics_from_tapes();
			if dt > 0.0 and not _burn_scheduled_ballistics.is_empty():
				_emit_scheduled_burn_events(_last_stepped_time, time);

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
	transform_updated.emit();
	if use_direct_driver_xf:
		for i: int in range(rail_count):
			var orbiter: JrRailOrbiter = r_seq[i];
			orbiter._system_transform = orbiter._driver_relative_transform;
			orbiter.transform_updated.emit();
	else:
		for i: int in range(rail_count):
			var orbiter: JrRailOrbiter = r_seq[i];
			orbiter._system_transform = rail_transforms[i];
			orbiter.transform_updated.emit();
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
	if not use_direct_driver_xf or not _non_root_sync_ballistics.is_empty() or b_seq[0].has_connections(&"transform_updated"):
		_flush_pending_gpu_tape_sample();
		t_math0 = Time.get_ticks_usec();
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
			b_seq[i].transform_updated.emit();
	else:
		for i: int in range(b_count):
			var ballistic: JrBallisticOrbiter = b_seq[i];
			ballistic._system_transform = ballistic_transforms[i];
			ballistic.transform_updated.emit();
	var t_sig1: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SYSTEM_LCA_MATH, t_math1 - t_math0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_TRANSFORM_SIGNAL_EMIT, t_sig1 - t_math1);
	_flush_pending_gpu_tape_sample();


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
		var t_init_pre: int = Time.get_ticks_usec();
		if not _ballistic_sequence.is_empty():
			if _ballistic_anchors_dirty:
				_update_ballistic_anchors_before_step();
			if _runtime_wells_dirty:
				_rebuild_runtime_well_cache();
			if enable_ballistic_forecasts and not _forecast_enabled_ballistics.is_empty():
				_schedule_and_bake_trajectory_tapes(0.0, prev_local_positions);
		var t_init_sync: int = Time.get_ticks_usec();
		_sync_all_ballistics_transforms(playing);
		var t_init_done: int = Time.get_ticks_usec();
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_RAIL_EVAL, t_init_b - t_init_rail);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_INIT, t_init_pre - t_init_b);
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
	if (
			dt < 0.0
			and not _ballistic_sequence.is_empty()
			and (time < _latest_tape_start_time or not _dirty_trajectory_ballistics.is_empty())
	):
		prev_local_positions = _capture_orbiter_local_positions();
	var t1: int = Time.get_ticks_usec();

	_evaluate_all_orbiters_at(time, playing);
	var t2: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_INIT, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_RAIL_EVAL, t2 - t1);

	if not _ballistic_sequence.is_empty():
		var t_pre0: int = Time.get_ticks_usec();
		if _ballistic_anchors_dirty:
			_update_ballistic_anchors_before_step();
		if _runtime_wells_dirty:
			_rebuild_runtime_well_cache();
		var t_pre1: int = Time.get_ticks_usec();
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_PRE, t_pre1 - t_pre0);

		if is_zero_approx(dt):
			if (
					enable_ballistic_forecasts
					and not _dirty_trajectory_ballistics.is_empty()
					and not _forecast_enabled_ballistics.is_empty()
			):
				_schedule_and_bake_trajectory_tapes(0.0, prev_local_positions);
		else:
			var sampled_in_bake: bool = false;
			if (
					not _dirty_trajectory_ballistics.is_empty()
					or time >= _earliest_tape_refill_time
					or time < _latest_tape_start_time
			):
				var want_fuse: bool = (
						_ballistic_sequence.size() >= JrBallisticComputeSolver.MIN_GPU_TAPE_ATLAS_BODIES
						and (_reference_plane == null or not _reference_plane.lock_to_ecliptic)
						and _ballistic_solver.is_tape_sample_gpu_active()
				);
				sampled_in_bake = _schedule_and_bake_trajectory_tapes(
						dt,
						prev_local_positions,
						want_fuse
				);
			if not sampled_in_bake:
				var t_samp0: int = Time.get_ticks_usec();
				_sample_all_ballistics_from_tapes();
				_telemetry.add_phase_usec(
						JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_POST,
						Time.get_ticks_usec() - t_samp0
				);
			if dt > 0.0 and not _burn_scheduled_ballistics.is_empty():
				_emit_scheduled_burn_events(_last_stepped_time, time);

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
			_ballistic_solver.is_forecast_gpu_active()
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
	if Engine.is_editor_hint():
		ballistic._obeying_shared_tape_clock = false;
		ballistic._state_time = 0.0;
	else:
		ballistic._obeying_shared_tape_clock = false;
		ballistic._state_time = _last_stepped_time if _runtime_initialized else time;
		ballistic.invalidate_trajectory();
		_queue_dirty_trajectory_ballistic(ballistic);


## Synchronizes transforms on all non-root-fast [JrBallisticOrbiter] nodes in `O(K)` and
## graduates root-parent/root-anchor bodies to lazy [member JrBallisticOrbiter._fast_root_sync_ready].
func _sync_all_ballistics_transforms(_playing: bool) -> void:
	var nr_count: int = _non_root_sync_ballistics.size();
	if nr_count == 0:
		return;
	_flush_pending_gpu_tape_sample();

	var self_elem: JrOrreryElement = self;
	var cur_pass: float = _sampled_time_cell[1];
	var pos_vecs: PackedVector3Array = _sampled_pos_holder[0] as PackedVector3Array;
	var write_idx: int = 0;

	for r: int in range(nr_count):
		var ballistic: JrBallisticOrbiter = _non_root_sync_ballistics[r];
		if not is_instance_valid(ballistic) or ballistic._owning_driver != self_elem:
			if is_instance_valid(ballistic):
				ballistic._in_non_root_sync_list = false;
			continue;

		var sync_res: int = ballistic.sync_non_root_fast(self_elem, cur_pass, pos_vecs);
		if sync_res == 2:
			continue;
		if sync_res == 0:
			_sync_single_ballistic_transforms(ballistic);
			ballistic._in_non_root_sync_list = true;
		_non_root_sync_ballistics[write_idx] = ballistic;
		write_idx += 1;

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
			ballistic.invalidate_trajectory();
			_queue_dirty_trajectory_ballistic(ballistic);

		var pinned: JrOrreryElement = ballistic.pinned_anchor;
		if pinned == null or pinned == curr_anchor:
			continue;
		if pinned._owning_driver == self_elem:
			var old_to_new_pos: Vector3 = _compute_lca_position_difference(pinned, curr_anchor);
			var old_to_new_vel: Vector3 = _compute_lca_velocity_difference(pinned, curr_anchor);
			ballistic.switch_anchor_element(pinned, old_to_new_pos, old_to_new_vel);
			ballistic.invalidate_trajectory();
			_queue_dirty_trajectory_ballistic(ballistic);
	_ballistic_anchors_dirty = false;


## Rebuilds [member _runtime_well_elements], [member _massive_child_orbiters_by_well], and
## [member _well_all_time_safe_radii] when gravitational topology or parameters change.
func _rebuild_runtime_well_cache() -> void:
	_recompute_gravity_strength_from_mass();
	_runtime_well_elements = _build_runtime_well_elements();
	var w_count: int = _runtime_well_elements.size();
	_massive_child_orbiters_by_well.resize(w_count);
	for w: int in range(w_count):
		_massive_child_orbiters_by_well[w] = [];

	for w: int in range(1, w_count):
		var orb := _runtime_well_elements[w] as JrRailOrbiter;
		if orb.get_gravity_strength() > 0.0:
			var p_elem: JrOrreryElement = _children.get(orb, self);
			var p_idx: int = (
					p_elem._runtime_well_index
					if (p_elem != null and p_elem._runtime_well_index >= 0)
					else 0
			);
			_massive_child_orbiters_by_well[p_idx].append(orb);

	_well_all_time_safe_radii.resize(w_count);
	for w: int in range(w_count):
		_well_all_time_safe_radii[w] = _compute_all_time_safe_radius_for_well(w);

	for b: JrBallisticOrbiter in _ballistic_sequence:
		var a_elem: JrOrreryElement = b._anchor_element;
		b._tape_anchor_idx = (
				a_elem._runtime_well_index
				if (a_elem != null and a_elem._runtime_well_index >= 0)
				else 0
		);
	_runtime_wells_dirty = false;


## Computes the worst-case all-time analytical tidal safe radius around well [param well_idx].
func _compute_all_time_safe_radius_for_well(well_idx: int) -> float:
	if tidal_cull_threshold <= 0.0 or well_idx < 0 or well_idx >= _runtime_well_elements.size():
		return 0.0;
	var well_a: JrOrreryElement = _runtime_well_elements[well_idx];
	var mu_a: float = _gravity_strength if well_idx == 0 else well_a.get_gravity_strength();
	if mu_a <= 0.0:
		return 0.0;

	var min_safe_r: float = INF;
	var w_count: int = _runtime_well_elements.size();
	for k: int in range(w_count):
		if k == well_idx:
			continue;
		var well_k: JrOrreryElement = _runtime_well_elements[k];
		var mu_k: float = _gravity_strength if k == 0 else well_k.get_gravity_strength();
		if mu_k <= 0.0:
			continue;
		var d_min: float = _compute_min_inter_well_periapsis(well_a, well_k);
		if d_min <= 1e-8:
			return 0.0;
		var scale_k: float = pow(tidal_cull_threshold / (2.0 * (mu_k / mu_a)), 1.0 / 3.0);
		min_safe_r = minf(min_safe_r, d_min * scale_k);
	return min_safe_r;


## Returns a guaranteed lower bound on `|r_a(t) - r_b(t)|` across all time `t` along the rail tree.
func _compute_min_inter_well_periapsis(
		well_a: JrOrreryElement,
		well_b: JrOrreryElement
) -> float:
	var lca: JrOrreryElement = _find_common_ancestor_well(well_a, well_b);
	var hops: Array[JrRailOrbiter] = [];

	var curr_a: JrOrreryElement = well_a;
	while curr_a != null and curr_a != lca and curr_a is JrRailOrbiter:
		var orb_a := curr_a as JrRailOrbiter;
		hops.append(orb_a);
		curr_a = _children.get(orb_a, self);

	var curr_b: JrOrreryElement = well_b;
	while curr_b != null and curr_b != lca and curr_b is JrRailOrbiter:
		var orb_b := curr_b as JrRailOrbiter;
		hops.append(orb_b);
		curr_b = _children.get(orb_b, self);

	if hops.is_empty():
		return 0.0;
	if hops.size() == 1:
		return hops[0].periapsis;

	var max_peri_idx: int = 0;
	var max_peri: float = hops[0].periapsis;
	for i: int in range(1, hops.size()):
		if hops[i].periapsis > max_peri:
			max_peri = hops[i].periapsis;
			max_peri_idx = i;

	var rest_apo_sum: float = 0.0;
	for i: int in range(hops.size()):
		if i == max_peri_idx:
			continue;
		if hops[i].is_unbound_orbit():
			return 0.0;
		rest_apo_sum += hops[i].apoapsis;
	return maxf(0.0, max_peri - rest_apo_sum);


## Returns the exact 64-bit orbital period when [param ballistic] is permanently unperturbed,
## bound, non-forecasted, and SOI-safe across all time (`0.0` otherwise).
func _try_compute_periodic_loop_period(ballistic: JrBallisticOrbiter) -> float:
	if (
			(enable_ballistic_forecasts and ballistic.runtime_forecast_enabled)
			or ballistic._has_external_thrust
			or ballistic.has_pending_or_active_burns_at(ballistic.get_effective_state_time())
	):
		return 0.0;
	var anchor_elem: JrOrreryElement = ballistic.get_anchor_element();
	if anchor_elem == null or not _is_tracked_or_self(anchor_elem):
		return 0.0;
	var anchor_idx: int = anchor_elem._runtime_well_index;
	if anchor_idx < 0 or anchor_idx >= _runtime_well_elements.size():
		return 0.0;

	var mu_anchor: float = _gravity_strength if anchor_idx == 0 else anchor_elem.get_gravity_strength();
	if mu_anchor <= 0.0:
		return 0.0;

	if ballistic.is_tape_position_pending():
		ballistic.ensure_tape_position();
	if ballistic.is_tape_velocity_pending():
		ballistic.ensure_tape_velocity();
	var r_vec: Vector3 = ballistic._anchor_relative_position;
	var v_vec: Vector3 = ballistic._anchor_relative_velocity;
	var r_len: float = r_vec.length();
	if r_len <= 1e-8:
		return 0.0;
	var alpha: float = (2.0 / r_len) - (v_vec.length_squared() / mu_anchor);
	if alpha <= 1e-9:
		return 0.0;

	var semi_a: float = 1.0 / alpha;
	var h_sq: float = r_vec.cross(v_vec).length_squared();
	var semi_p: float = h_sq / mu_anchor;
	var ecc: float = sqrt(maxf(0.0, 1.0 - (semi_p / semi_a)));
	if ecc >= 0.85:
		return 0.0;

	var r_peri: float = semi_a * (1.0 - ecc);
	var r_apo: float = semi_a * (1.0 + ecc);
	if r_peri <= 1e-6:
		return 0.0;
	if ballistic.enable_tidal_forces and r_apo >= 0.98 * _well_all_time_safe_radii[anchor_idx]:
		return 0.0;
	if ballistic.pinned_anchor == null and _does_orbit_overlap_well_soi(anchor_idx, r_peri, r_apo):
		return 0.0;

	var mean_n: float = sqrt(mu_anchor / (semi_a * semi_a * semi_a));
	return (TAU / mean_n) if mean_n > 0.0 else 0.0;


## Returns `true` if radial shell `[r_peri, r_apo]` around [param anchor_idx] can intersect any
## parent Hill exit sphere or direct child Hill entry band.
func _does_orbit_overlap_well_soi(anchor_idx: int, r_peri: float, r_apo: float) -> bool:
	if anchor_idx > 0:
		var anchor_orb := _runtime_well_elements[anchor_idx] as JrRailOrbiter;
		var p_mu: float = anchor_orb.get_effective_parent_mu();
		var mu_self: float = anchor_orb.get_gravity_strength();
		if p_mu > 0.0 and mu_self > 0.0:
			var min_exit_r: float = (
					HILL_EXIT_MULTIPLIER
					* anchor_orb.periapsis
					* pow(mu_self / (3.0 * p_mu), 1.0 / 3.0)
			);
			if r_apo >= 0.95 * min_exit_r:
				return true;

	var children: Array = _massive_child_orbiters_by_well[anchor_idx];
	for i: int in range(children.size()):
		var c_orb := children[i] as JrRailOrbiter;
		if c_orb.is_unbound_orbit():
			return true;
		var c_mu: float = c_orb.get_gravity_strength();
		var p_eff_mu: float = c_orb.get_effective_parent_mu();
		if c_mu <= 0.0 or p_eff_mu <= 0.0:
			continue;
		var entry_ratio: float = HILL_ENTRY_MULTIPLIER * pow(c_mu / (3.0 * p_eff_mu), 1.0 / 3.0);
		var band_min: float = c_orb.periapsis * (1.0 - 1.05 * entry_ratio);
		var band_max: float = c_orb.apoapsis * (1.0 + 1.05 * entry_ratio);
		if r_apo >= band_min and r_peri <= band_max:
			return true;
	return false;


## Schedules and executes batched GPU/CPU trajectory-horizon bakes for dirty, expiring,
## scrubbed, or reverse-stepping [JrBallisticOrbiter] cohorts. Returns `true` when all bodies
## were also sampled at [member time] via a fused single-submit GPU pass.
func _schedule_and_bake_trajectory_tapes(
		dt: float,
		prev_local_positions: Dictionary[JrRailOrbiter, Vector3],
		fuse_gpu_sample: bool = false
) -> bool:
	if dt < 0.0:
		_handle_reverse_time_tapes(dt, prev_local_positions);

	var abs_dt: float = absf(dt);
	if dt > 0.1:
		var bridge_bodies: Array[JrBallisticOrbiter] = [];
		for b: JrBallisticOrbiter in _ballistic_sequence:
			if b._trajectory_dirty or not b._trajectory_tape.is_valid or b._trajectory_tape.is_periodic_loop:
				continue;
			var end_t: float = b._trajectory_tape.get_end_time();
			var gap: float = time - end_t;
			if gap > 64.0 * b._trajectory_tape.knot_dt and gap > 0.1:
				var tail_w: int = clampi(
						b._trajectory_tape.get_tail_well_index(),
						0,
						maxi(0, _runtime_well_elements.size() - 1)
				);
				b.adopt_tape_anchor_element(_runtime_well_elements[tail_w], tail_w);
				b._anchor_relative_position = b._trajectory_tape.get_tail_position();
				b._anchor_relative_velocity = b._trajectory_tape.get_tail_velocity();
				b._obeying_shared_tape_clock = false;
				b._tape_vel_pending = false;
				b._state_time = end_t;
				bridge_bodies.append(b);
		if not bridge_bodies.is_empty():
			_bridge_integrate_bodies_to_now(bridge_bodies);

	var candidates: Array[JrBallisticOrbiter] = _collect_scheduled_tape_candidates(dt, abs_dt);
	var did_fuse_sample: bool = false;
	if enable_ballistic_forecasts and not candidates.is_empty():
		var fc_candidates: Array[JrBallisticOrbiter] = [];
		var active_candidates: Array[JrBallisticOrbiter] = [];
		for i: int in range(candidates.size()):
			var c_body: JrBallisticOrbiter = candidates[i];
			if c_body.runtime_forecast_enabled:
				fc_candidates.append(c_body);
			else:
				active_candidates.append(c_body);
		if not fc_candidates.is_empty() and not active_candidates.is_empty():
			_bake_scheduled_candidate_batch(fc_candidates, dt, false);
			did_fuse_sample = _bake_scheduled_candidate_batch(
					active_candidates,
					dt,
					fuse_gpu_sample
			);
		else:
			did_fuse_sample = _bake_scheduled_candidate_batch(candidates, dt, fuse_gpu_sample);
	elif not candidates.is_empty():
		did_fuse_sample = _bake_scheduled_candidate_batch(candidates, dt, fuse_gpu_sample);

	if not _ballistic_solver.has_pending_gpu_sample_sync():
		if not _dirty_trajectory_ballistics.is_empty():
			_earliest_tape_refill_time = -INF;
		else:
			_earliest_tape_refill_time = _min_across_cohort_refills();
	return did_fuse_sample;


## Builds and executes a trajectory-tape bake (or fused GPU bake+sample) for [param candidates].
func _bake_scheduled_candidate_batch(
		candidates: Array[JrBallisticOrbiter],
		dt: float,
		fuse_gpu_sample: bool
) -> bool:
	var abs_dt: float = absf(dt);
	var use_gpu: bool = _ballistic_solver.is_forecast_gpu_active();
	var ordered: Array[JrBallisticOrbiter] = (
			_partition_candidates_for_simt_coherence(candidates)
			if (use_gpu and candidates.size() >= 64)
			else candidates
	);

	var cand_n: int = ordered.size();
	_batch_jobs.resize(cand_n);
	_batch_start_times.resize(cand_n);
	_batch_knot_dts.resize(cand_n);
	_batch_step_counts.resize(cand_n);
	_batch_substeps.resize(cand_n);
	_batch_modes.resize(cand_n);
	_batch_loop_periods.resize(cand_n);

	var out_idx: int = 0;
	var batch_t0: float = INF;
	var batch_t1: float = -INF;
	var dt_is_zero: bool = is_zero_approx(dt);

	for i: int in range(cand_n):
		var b: JrBallisticOrbiter = ordered[i];
		b._in_dirty_trajectory_queue = false;
		var is_fc: bool = enable_ballistic_forecasts and b.runtime_forecast_enabled;
		var needs_reset: bool = b._trajectory_dirty or not b._trajectory_tape.is_valid;
		var is_reschedule: bool = false;

		if needs_reset and dt_is_zero and not is_fc:
			b._in_dirty_trajectory_queue = true;
			_dirty_trajectory_ballistics.append(b);
			continue;

		if not needs_reset and not is_fc and abs_dt > 0.0:
			var desired_dt: float = _compute_tail_desired_knot_dt(b._trajectory_tape, abs_dt);
			var end_t: float = b._trajectory_tape.get_end_time();
			var tail_tol: float = maxf(1e-6, 1e-4 * b._trajectory_tape.knot_dt);
			var finished_scheduled_burns: bool = (
					b.has_scheduled_burns()
					and not b._has_external_thrust
					and not b.has_pending_or_active_burns_at(end_t)
					and time >= end_t - tail_tol
			);
			if (
					finished_scheduled_burns
					or desired_dt > 4.0 * b._trajectory_tape.knot_dt
					or desired_dt < 0.2 * b._trajectory_tape.knot_dt
			):
				if time >= end_t - tail_tol:
					var tw: int = b.apply_tape_tail_state(end_t);
					if tw != b._tape_anchor_idx and tw >= 0 and tw < _runtime_well_elements.size():
						b.adopt_tape_anchor_element(_runtime_well_elements[tw], tw);
					is_reschedule = true;
				else:
					var clamp_t: float = minf(time, end_t);
					var sw: int = b.sample_tape_same_soi(clamp_t);
					if sw == -2:
						_sample_ballistic_cross_soi_at(b, clamp_t);
					elif sw != b._tape_anchor_idx and sw >= 0 and sw < _runtime_well_elements.size():
						b.adopt_tape_anchor_element(_runtime_well_elements[sw], sw);
				needs_reset = true;

		if needs_reset:
			var period: float = _try_compute_periodic_loop_period(b);
			if period > 0.0:
				var l_steps: int = 128 if use_gpu else 96;
				var l_dt: float = period / float(l_steps);
				var j_t0_loop: float = b.get_effective_state_time();
				_batch_jobs[out_idx] = b;
				_batch_start_times[out_idx] = j_t0_loop;
				_batch_knot_dts[out_idx] = l_dt;
				_batch_step_counts[out_idx] = l_steps;
				_batch_substeps[out_idx] = 1;
				_batch_modes[out_idx] = JrBallisticForecastConfig.JOB_PERIODIC_LOOP;
				_batch_loop_periods[out_idx] = period;
				out_idx += 1;
				batch_t0 = minf(batch_t0, j_t0_loop);
				batch_t1 = maxf(batch_t1, j_t0_loop + period);
				continue;

			var j_t0: float = b.get_effective_state_time();
			var lead_in: float = maxf(0.0, time - j_t0);
			var k_dt: float;
			var steps: int;
			var subs: int;
			var cohort: int = b._tape_cohort;
			if is_fc:
				var p_steps: int = maxi(8, b.preview_steps);
				k_dt = maxf(1e-6, b.preview_duration) / float(p_steps);
				var full_chunk: int = _compute_forecast_chunk_steps(p_steps, k_dt, abs_dt, use_gpu);
				var extra: int = 8 + int(roundf(float(full_chunk * cohort) / float(TAPE_COHORT_COUNT)));
				steps = p_steps + int(ceili(lead_in / k_dt)) + extra;
				subs = maxi(JrBallisticComputeSolver.FORECAST_INTERNAL_SUBSTEPS, ballistic_substeps);
			else:
				k_dt = _compute_active_knot_dt(b, abs_dt);
				var full_chunk_r: int = _compute_active_chunk_steps(k_dt, abs_dt);
				if dt < 0.0:
					steps = maxi(2, int(ceili(lead_in / k_dt)) + 2);
				elif is_reschedule:
					var needed_r: float = maxf(0.0, (time - j_t0) - 0.05 * k_dt);
					steps = int(ceili(needed_r / k_dt)) + full_chunk_r;
				else:
					var one_dt: float = maxf(1e-9, abs_dt);
					var chunk_frames: int = clampi(
							int(roundf((float(full_chunk_r) * k_dt) / one_dt)),
							1,
							TAPE_COHORT_COUNT
					);
					var frame_steps: int = (
							int(ceili((lead_in - 1e-6 * abs_dt) / one_dt))
							+ (cohort % chunk_frames)
							+ 1
					);
					var total_span: float = float(frame_steps) * one_dt;
					steps = maxi(2, int(roundf(total_span / maxf(1e-9, k_dt))));
				if use_gpu:
					steps = mini(steps, JrBallisticComputeSolver.GPU_TAPE_SLOT_KNOTS - 1);
				subs = _compute_active_substeps_for_body(b, abs_dt, false);

			_batch_jobs[out_idx] = b;
			_batch_start_times[out_idx] = j_t0;
			_batch_knot_dts[out_idx] = k_dt;
			_batch_step_counts[out_idx] = steps;
			_batch_substeps[out_idx] = subs;
			_batch_modes[out_idx] = JrBallisticForecastConfig.JOB_RESET;
			_batch_loop_periods[out_idx] = 0.0;
			out_idx += 1;
			batch_t0 = minf(batch_t0, j_t0);
			batch_t1 = maxf(batch_t1, j_t0 + float(steps) * k_dt);
		else:
			var tail_t0: float = b._trajectory_tape.get_end_time();
			var k_dt: float = b._trajectory_tape.knot_dt;
			var steps: int;
			var subs: int;
			if is_fc:
				var p_steps: int = maxi(8, b.preview_steps);
				var full_chunk: int = _compute_forecast_chunk_steps(
						p_steps,
						k_dt,
						abs_dt,
						use_gpu and cand_n >= 4
				);
				var needed: float = maxf(0.0, (time + b.preview_duration) - tail_t0);
				steps = int(ceili(needed / k_dt)) + full_chunk;
				subs = maxi(JrBallisticComputeSolver.FORECAST_INTERNAL_SUBSTEPS, ballistic_substeps);
			else:
				var full_chunk: int = _compute_active_chunk_steps(k_dt, abs_dt);
				var needed: float = maxf(0.0, (time - tail_t0) - 0.05 * k_dt);
				steps = int(ceili(needed / k_dt)) + full_chunk;
				if use_gpu:
					steps = mini(steps, JrBallisticComputeSolver.GPU_TAPE_SLOT_KNOTS - 1);
				subs = _compute_active_substeps_for_body(b, abs_dt, true);

			_batch_jobs[out_idx] = b;
			_batch_start_times[out_idx] = tail_t0;
			_batch_knot_dts[out_idx] = k_dt;
			_batch_step_counts[out_idx] = steps;
			_batch_substeps[out_idx] = subs;
			_batch_modes[out_idx] = JrBallisticForecastConfig.JOB_APPEND_TAIL;
			_batch_loop_periods[out_idx] = 0.0;
			out_idx += 1;
			batch_t0 = minf(batch_t0, tail_t0);
			batch_t1 = maxf(batch_t1, tail_t0 + float(steps) * k_dt);

	if out_idx != cand_n:
		_batch_jobs.resize(out_idx);
		_batch_start_times.resize(out_idx);
		_batch_knot_dts.resize(out_idx);
		_batch_step_counts.resize(out_idx);
		_batch_substeps.resize(out_idx);
		_batch_modes.resize(out_idx);
		_batch_loop_periods.resize(out_idx);

	if _batch_jobs.is_empty():
		return false;

	var cfg: JrBallisticForecastConfig = _build_forecast_config(_batch_jobs, batch_t0);
	cfg.total_driver_bodies = _ballistic_sequence.size();
	cfg.enable_forecasts = enable_ballistic_forecasts;
	cfg.timeline_end_time = maxf(batch_t0, batch_t1);
	cfg.trim_before_time = time;
	cfg.timeline_segments = 0;
	cfg.job_start_times = _batch_start_times;
	cfg.job_knot_dts = _batch_knot_dts;
	cfg.job_step_counts = _batch_step_counts;
	cfg.job_substeps = _batch_substeps;
	cfg.job_append_mode = _batch_modes;
	cfg.job_loop_periods = _batch_loop_periods;
	if fuse_gpu_sample and not dt_is_zero:
		_ballistic_solver.bake_and_sample_tapes_gpu(
				cfg,
				_ballistic_sequence,
				time
		);
		if _ballistic_solver.did_last_bake_fuse_sample():
			_sampled_time_cell[0] = time;
			_sampled_time_cell[1] += 1.0;
			_pending_gpu_finalize_jobs = _batch_jobs;
			return true;
		_finalize_baked_tape_jobs(_batch_jobs, false);
		return false;

	_ballistic_solver.bake_trajectory_tapes(cfg);
	_finalize_baked_tape_jobs(_batch_jobs, dt_is_zero);
	return false;


## Partitions [param candidates] so root-anchored coasting bodies precede child-well or thrusting
## bodies, packing perturbed KDK threads into contiguous GPU SIMT waves.
func _partition_candidates_for_simt_coherence(
		candidates: Array[JrBallisticOrbiter]
) -> Array[JrBallisticOrbiter]:
	var self_elem: JrOrreryElement = self;
	_batch_coasting_buf.clear();
	_batch_perturbed_buf.clear();
	for i: int in range(candidates.size()):
		var b: JrBallisticOrbiter = candidates[i];
		if b._anchor_element == self_elem and not b._has_external_thrust and not b.has_scheduled_burns():
			_batch_coasting_buf.append(b);
		else:
			_batch_perturbed_buf.append(b);
	if _batch_coasting_buf.is_empty() or _batch_perturbed_buf.is_empty():
		return candidates;
	_batch_coasting_buf.append_array(_batch_perturbed_buf);
	return _batch_coasting_buf;


## Collects dirty and refill-ready [JrBallisticOrbiter] bodies, scanning only expiring cohorts
## (`~1/16` of the swarm) during steady-state forward playback.
func _collect_scheduled_tape_candidates(
		dt: float,
		abs_dt: float
) -> Array[JrBallisticOrbiter]:
	var fast_cohort_pass: bool = (
			dt > 0.0
			and dt <= 0.1
			and _last_schedule_abs_dt > 0.0
			and abs_dt <= 1.8 * _last_schedule_abs_dt
			and abs_dt >= 0.55 * _last_schedule_abs_dt
	);
	var candidates: Array[JrBallisticOrbiter] = [];

	if fast_cohort_pass:
		_last_schedule_abs_dt = abs_dt;
		var dirty_snapshot: Array[JrBallisticOrbiter] = _dirty_trajectory_ballistics.duplicate();
		_dirty_trajectory_ballistics.clear();
		for c: int in range(TAPE_COHORT_COUNT):
			if time < _cohort_min_refill[c]:
				continue;
			_cohort_min_refill[c] = INF;
			var c_list: Array = _tape_cohorts[c];
			for k: int in range(c_list.size()):
				var b := c_list[k] as JrBallisticOrbiter;
				if not b._trajectory_dirty and b._trajectory_tape.is_valid:
					if b._trajectory_tape.is_periodic_loop:
						continue;
					if time < b._next_refill_time:
						_cohort_min_refill[c] = minf(_cohort_min_refill[c], b._next_refill_time);
						continue;
				b._in_dirty_trajectory_queue = false;
				candidates.append(b);
		for b: JrBallisticOrbiter in dirty_snapshot:
			if b._in_dirty_trajectory_queue and b._owning_driver == self:
				b._in_dirty_trajectory_queue = false;
				candidates.append(b);
		return candidates;

	_dirty_trajectory_ballistics.clear();
	_reset_cohort_min_refill(INF);
	_latest_tape_start_time = -INF;
	_last_schedule_abs_dt = abs_dt if (dt > 0.0 and dt <= 0.1) else 0.0;

	for b: JrBallisticOrbiter in _ballistic_sequence:
		if not b._trajectory_dirty and b._trajectory_tape.is_valid:
			if b._trajectory_tape.is_periodic_loop:
				continue;
			if time < b._next_refill_time:
				var c_idx: int = b._tape_cohort;
				_cohort_min_refill[c_idx] = minf(_cohort_min_refill[c_idx], b._next_refill_time);
				_latest_tape_start_time = maxf(_latest_tape_start_time, b._trajectory_tape.start_time);
				continue;
		b._in_dirty_trajectory_queue = false;
		candidates.append(b);
	return candidates;


## Marks newly baked [param jobs] clean, updates per-cohort refill bounds, and binds bodies to
## [member _sampled_time_cell].
func _finalize_baked_tape_jobs(jobs: Array[JrBallisticOrbiter], dt_is_zero: bool) -> void:
	if dt_is_zero:
		_sampled_time_cell[0] = time;
	var cur_pass: float = _sampled_time_cell[1];
	var max_start: float = _latest_tape_start_time;

	for i: int in range(jobs.size()):
		var b: JrBallisticOrbiter = jobs[i];
		var tape: JrBallisticTrajectoryTape = b._trajectory_tape;
		if tape.is_periodic_loop:
			if tape.was_completely_unperturbed():
				b.mark_trajectory_clean(INF);
			else:
				tape.is_periodic_loop = false;
				tape.loop_period = 0.0;
				_ballistic_solver.clear_body_periodic_loop_in_atlas(b._ballistic_seq_index);
				b.mark_trajectory_clean(tape.get_end_time() - 1e-4 * tape.knot_dt);
		elif enable_ballistic_forecasts and b.runtime_forecast_enabled:
			b.mark_trajectory_clean(tape.get_end_time() - b.preview_duration);
		else:
			b.mark_trajectory_clean(tape.get_end_time() - 1e-4 * tape.knot_dt);

		var c_idx: int = b._tape_cohort;
		_cohort_min_refill[c_idx] = minf(_cohort_min_refill[c_idx], b._next_refill_time);
		if not tape.is_periodic_loop:
			max_start = maxf(max_start, tape.start_time);
		if tape.is_valid:
			b._obeying_shared_tape_clock = true;
			if dt_is_zero:
				b._pos_materialized_pass_id = cur_pass;
				b._vel_materialized_pass_id = cur_pass;

	_latest_tape_start_time = max_start;


## Steps non-periodic bodies backward when [member time] moves earlier than their tape start.
func _handle_reverse_time_tapes(
		dt: float,
		prev_local_positions: Dictionary[JrRailOrbiter, Vector3]
) -> void:
	var backward_bodies: Array[JrBallisticOrbiter] = [];
	for b: JrBallisticOrbiter in _ballistic_sequence:
		if b._trajectory_tape.is_valid and b._trajectory_tape.is_periodic_loop and not b._trajectory_dirty:
			continue;
		if (
				not b._trajectory_dirty
				and b._trajectory_tape.is_valid
				and time >= b._trajectory_tape.start_time
				and time <= b._trajectory_tape.get_end_time()
		):
			continue;
		b.exit_on_rails_epoch();
		if (
				b._trajectory_tape.is_valid
				and not is_equal_approx(b.get_effective_state_time(), _last_stepped_time)
				and b._trajectory_tape.contains_time(_last_stepped_time)
		):
			var sw: int = b.sample_tape_same_soi(_last_stepped_time);
			if sw == -2:
				_sample_ballistic_cross_soi_at(b, _last_stepped_time);
			elif sw >= 0 and sw < _runtime_well_elements.size():
				b.adopt_tape_anchor_element(_runtime_well_elements[sw], sw);
		backward_bodies.append(b);

	if backward_bodies.is_empty():
		return;

	var abs_dt: float = absf(dt);
	var target_back_t: float = (time - 15.0 * abs_dt) if abs_dt <= 0.25 else time;
	_step_bodies_over_interval(backward_bodies, _last_stepped_time, target_back_t, prev_local_positions);
	for b: JrBallisticOrbiter in backward_bodies:
		b._obeying_shared_tape_clock = false;
		b._state_time = target_back_t;
		b.invalidate_trajectory();
		_queue_dirty_trajectory_ballistic(b);


## Integrates [param bodies] from [param t_from] to [param t_to] using adaptive KDK sub-intervals.
func _step_bodies_over_interval(
		bodies: Array[JrBallisticOrbiter],
		t_from: float,
		t_to: float,
		prev_local_positions: Dictionary[JrRailOrbiter, Vector3]
) -> void:
	var total_dt: float = t_to - t_from;
	if bodies.is_empty() or is_zero_approx(total_dt):
		return;

	var max_mean_n: float = 0.0;
	for orb: JrRailOrbiter in _orbit_sequence:
		if orb.is_orbit_valid() and orb.get_gravity_strength() > 0.0:
			max_mean_n = maxf(max_mean_n, orb.mean_motion);
	var max_chunk_dt: float = 0.15 / maxf(0.1, max_mean_n);
	var n_chunks: int = clampi(int(ceili(absf(total_dt) / max_chunk_dt)), 1, 128);
	var subs: int = maxi(2, ballistic_substeps * 2);
	_ballistic_solver.invalidate_resident_step_buffer();

	if n_chunks == 1 and is_equal_approx(t_to, time) and not prev_local_positions.is_empty():
		var packed_single: PackedFloat32Array = _pack_runtime_gravity_wells(
				_runtime_well_elements,
				prev_local_positions
		);
		_ballistic_solver.configure_step_context(t_from, tidal_cull_threshold, _runtime_well_elements);
		_ballistic_solver.step_bodies(bodies, packed_single, total_dt, subs);
		_update_ballistic_anchors_after_step(bodies, _runtime_well_elements);
		_ballistic_solver.invalidate_resident_step_buffer();
		return;

	var playing: bool = _should_step_time();
	var start_map: Dictionary[JrRailOrbiter, Vector3] = prev_local_positions;
	if start_map.is_empty():
		_evaluate_all_orbiters_at(t_from, playing);
		start_map = _capture_orbiter_local_positions();

	for c: int in range(n_chunks):
		var t_c0: float = lerpf(t_from, t_to, float(c) / float(n_chunks));
		var t_c1: float = lerpf(t_from, t_to, float(c + 1) / float(n_chunks));
		_evaluate_all_orbiters_at(t_c1, playing);
		var packed_c: PackedFloat32Array = _pack_runtime_gravity_wells(
				_runtime_well_elements,
				start_map
		);
		_ballistic_solver.configure_step_context(t_c0, tidal_cull_threshold, _runtime_well_elements);
		_ballistic_solver.step_bodies(bodies, packed_c, t_c1 - t_c0, subs);
		_update_ballistic_anchors_after_step(bodies, _runtime_well_elements);
		if c + 1 < n_chunks:
			start_map = _capture_orbiter_local_positions();

	_ballistic_solver.invalidate_resident_step_buffer();
	_evaluate_all_orbiters_at(time, playing);


## Advances [param bodies] across a large forward time scrub up to [member time] using a coarse
## timeline-sampled GPU/CPU trajectory bridge pass, then marks them dirty for fine horizon baking.
func _bridge_integrate_bodies_to_now(bodies: Array[JrBallisticOrbiter]) -> void:
	var start_times := PackedFloat64Array();
	var knot_dts := PackedFloat32Array();
	var step_counts := PackedInt32Array();
	var job_subs := PackedInt32Array();
	var modes := PackedByteArray();
	var batch_t0: float = INF;

	for b: JrBallisticOrbiter in bodies:
		var j_t0: float = b.get_effective_state_time();
		var span: float = maxf(1e-6, time - j_t0);
		var steps: int = clampi(int(ceili(span / 0.02)), 16, 256);
		start_times.append(j_t0);
		knot_dts.append(span / float(steps));
		step_counts.append(steps);
		job_subs.append(maxi(4, ballistic_substeps * 2));
		modes.append(JrBallisticForecastConfig.JOB_RESET);
		batch_t0 = minf(batch_t0, j_t0);

	var cfg: JrBallisticForecastConfig = _build_forecast_config(bodies, batch_t0);
	cfg.timeline_end_time = time;
	cfg.trim_before_time = time;
	cfg.timeline_segments = 0;
	cfg.job_start_times = start_times;
	cfg.job_knot_dts = knot_dts;
	cfg.job_step_counts = step_counts;
	cfg.job_substeps = job_subs;
	cfg.job_append_mode = modes;
	_ballistic_solver.bake_trajectory_tapes(cfg);

	var well_limit: int = maxi(0, _runtime_well_elements.size() - 1);
	for b: JrBallisticOrbiter in bodies:
		var tape: JrBallisticTrajectoryTape = b._trajectory_tape;
		if tape.is_valid:
			var tw: int = clampi(tape.get_tail_well_index(), 0, well_limit);
			b.adopt_tape_anchor_element(_runtime_well_elements[tw], tw);
			b._anchor_relative_position = tape.get_tail_position();
			b._anchor_relative_velocity = tape.get_tail_velocity();
			b._tape_vel_pending = false;
		b._obeying_shared_tape_clock = false;
		b._state_time = time;
		b.invalidate_trajectory();


## Chooses frame-aligned knot spacing (`abs_dt`) for a non-forecasted active body so all cohorts
## share uniform knot counts (`steps = TAPE_COHORT_COUNT`) and land on exact knots each tick.
func _compute_active_knot_dt(_ballistic: JrBallisticOrbiter, abs_dt: float) -> float:
	if abs_dt <= 1e-9:
		return 1e-3;
	return abs_dt;


## Evaluates desired knot spacing (`abs_dt`) for an active tail extension.
func _compute_tail_desired_knot_dt(_tape: JrBallisticTrajectoryTape, abs_dt: float) -> float:
	if abs_dt <= 1e-9:
		return 1e-3;
	return abs_dt;


## Chooses tail-chunk step count spanning `TAPE_COHORT_COUNT * abs_dt` (`64` frames).
func _compute_active_chunk_steps(knot_dt: float, abs_dt: float) -> int:
	var target_t: float = float(TAPE_COHORT_COUNT) * abs_dt;
	return clampi(int(roundf(target_t / maxf(1e-9, knot_dt))), 2, 128);


## Computes the adaptive KDK substep count (`{1, 2, 4}`) for [param ballistic] over `abs_dt`
## from its `O(1)` tail summary (when [param from_tail] is `true`) or current state.
func _compute_active_substeps_for_body(
		ballistic: JrBallisticOrbiter,
		abs_dt: float,
		from_tail: bool
) -> int:
	if ballistic.has_scheduled_burns():
		return maxi(JrBallisticComputeSolver.FORECAST_INTERNAL_SUBSTEPS, ballistic_substeps);
	var base_subs: int = maxi(1, ballistic_substeps);
	if abs_dt <= 1e-9:
		return base_subs;
	var r_sq: float;
	var v_sq: float;
	if from_tail and ballistic._trajectory_tape.is_valid:
		var tp: Vector4 = ballistic._trajectory_tape.tail_pos_anchor;
		var tv: Vector4 = ballistic._trajectory_tape.tail_vel_flags;
		r_sq = maxf(1e-12, tp.x * tp.x + tp.y * tp.y + tp.z * tp.z);
		v_sq = tv.x * tv.x + tv.y * tv.y + tv.z * tv.z;
	else:
		if ballistic.is_tape_position_pending():
			ballistic.ensure_tape_position();
		if ballistic.is_tape_velocity_pending():
			ballistic.ensure_tape_velocity();
		r_sq = maxf(1e-12, ballistic._anchor_relative_position.length_squared());
		v_sq = ballistic._anchor_relative_velocity.length_squared();
	if v_sq * (abs_dt * abs_dt) <= 0.16 * r_sq:
		return base_subs;
	return maxi(2, base_subs);


## Chooses extra headroom knot count beyond `preview_steps` for a forecasted body.
func _compute_forecast_chunk_steps(
		preview_steps: int,
		knot_dt: float,
		abs_dt: float,
		use_gpu: bool
) -> int:
	@warning_ignore("integer_division")
	var base_chunk: int = preview_steps / (2 if use_gpu else 4);
	var frame_target: float = (24.0 if use_gpu else 10.0) * abs_dt;
	var from_dt: int = int(ceili(frame_target / maxf(1e-7, knot_dt))) if abs_dt > 0.0 else 0;
	return clampi(maxi(base_chunk, from_dt), 16, 128 if use_gpu else 64);


## Recomputes [member _earliest_tape_refill_time], [member _cohort_min_refill], and
## [member _latest_tape_start_time] across [member _ballistic_sequence].
func _recompute_tape_schedule_bounds() -> void:
	_reset_cohort_min_refill(INF);
	var min_refill: float = INF;
	var max_start: float = -INF;
	for b: JrBallisticOrbiter in _ballistic_sequence:
		if b._trajectory_dirty or not b._trajectory_tape.is_valid:
			min_refill = -INF;
			_cohort_min_refill[b._tape_cohort] = -INF;
			continue;
		var c_idx: int = b._tape_cohort;
		_cohort_min_refill[c_idx] = minf(_cohort_min_refill[c_idx], b._next_refill_time);
		min_refill = minf(min_refill, b._next_refill_time);
		if not b._trajectory_tape.is_periodic_loop:
			max_start = maxf(max_start, b._trajectory_tape.start_time);
	_earliest_tape_refill_time = min_refill;
	_latest_tape_start_time = max_start;


## Samples all tracked [JrBallisticOrbiter] bodies at [member time] in `O(1)` per body via
## parallel GPU compute (`ballistic_tape_sample.glsl`) or CPU Cubic Hermite interpolation.
func _sample_all_ballistics_from_tapes() -> void:
	var b_seq: Array[JrBallisticOrbiter] = _ballistic_sequence;
	var count: int = b_seq.size();
	if count == 0:
		return;

	var eval_t: float = time;
	_sampled_time_cell[0] = eval_t;
	_sampled_time_cell[1] += 1.0;
	var cur_pass: float = _sampled_time_cell[1];

	var well_count: int = _runtime_well_elements.size();
	var ref_plane: JrReferenceDirection = _reference_plane;
	var lock_ecliptic: bool = ref_plane != null and ref_plane.lock_to_ecliptic;

	if (
			count >= JrBallisticComputeSolver.MIN_GPU_TAPE_SAMPLE_BODIES
			and not lock_ecliptic
			and _ballistic_solver.is_tape_sample_gpu_active()
	):
		_ballistic_solver.sample_tapes_gpu(b_seq, eval_t);
		return;

	_sampled_pos_holder[0] = PackedVector3Array();
	var tapes: Array[JrBallisticTrajectoryTape] = _ballistic_tapes;
	for i: int in range(count):
		var tape: JrBallisticTrajectoryTape = tapes[i];
		if not tape.is_valid:
			continue;
		if not tape.cpu_knots_resident:
			tape.ensure_cpu_knots();

		var rel_t: float = eval_t - tape.start_time;
		if tape.is_periodic_loop:
			rel_t = fposmod(rel_t, tape.loop_period);
		var max_idx: float = float(tape.knot_count - 1);
		var scaled: float = clampf(rel_t * tape.inv_knot_dt, 0.0, max_idx);
		var idx_a: int = mini(int(scaled), tape.knot_count - 2);
		var u: float = scaled - float(idx_a);

		var pa: Vector4 = tape.pos_anchor[idx_a];
		var r_rel: Vector3;
		var w_idx: int;
		if u <= 1e-5:
			r_rel = Vector3(pa.x, pa.y, pa.z);
			w_idx = int(pa.w);
		else:
			var pb: Vector4 = tape.pos_anchor[idx_a + 1];
			if pa.w != pb.w:
				_sample_ballistic_cross_soi_at(b_seq[i], eval_t);
				continue;
			if u >= 0.99999:
				r_rel = Vector3(pb.x, pb.y, pb.z);
				w_idx = int(pb.w);
			else:
				var dt_k: float = tape.knot_dt;
				var u2: float = u * u;
				var u3: float = u2 * u;
				var h00: float = 2.0 * u3 - 3.0 * u2 + 1.0;
				var h10: float = (u3 - 2.0 * u2 + u) * dt_k;
				var h01: float = -2.0 * u3 + 3.0 * u2;
				var h11: float = (u3 - u2) * dt_k;
				var p4: Vector4 = (
						pa * h00
						+ tape.vel_flags[idx_a] * h10
						+ pb * h01
						+ tape.vel_flags[idx_a + 1] * h11
				);
				r_rel = Vector3(p4.x, p4.y, p4.z);
				w_idx = int(pa.w);

		var ballistic: JrBallisticOrbiter = b_seq[i];
		if lock_ecliptic:
			r_rel = ref_plane.project_vector_to_ecliptic(r_rel);
		ballistic._anchor_relative_position = r_rel;
		ballistic._pos_materialized_pass_id = cur_pass;
		ballistic._obeying_shared_tape_clock = true;
		if w_idx != ballistic._tape_anchor_idx:
			var new_anchor: JrOrreryElement = (
					_runtime_well_elements[w_idx]
					if w_idx < well_count
					else self
			);
			ballistic.adopt_tape_anchor_element(new_anchor, w_idx);


## Waits for an in-flight GPU tape sample or fused bake+sample pass, finalizes baked tape jobs,
## and dispatches sparse SOI handoff events.
func _flush_pending_gpu_tape_sample() -> void:
	if not _ballistic_solver.has_pending_gpu_sample_sync():
		return;
	var ev_vec4s: PackedVector4Array = _ballistic_solver.sync_and_readback_pending_gpu_sample(
			_ballistic_sequence
	);
	_sampled_pos_holder[0] = _ballistic_solver.get_last_sampled_pos_vecs();
	if not _pending_gpu_finalize_jobs.is_empty():
		var jobs_to_fin: Array[JrBallisticOrbiter] = _pending_gpu_finalize_jobs;
		_pending_gpu_finalize_jobs = [];
		_finalize_baked_tape_jobs(jobs_to_fin, false);
		if not _dirty_trajectory_ballistics.is_empty():
			_earliest_tape_refill_time = -INF;
		else:
			_earliest_tape_refill_time = _min_across_cohort_refills();
	_apply_gpu_tape_sample_events(ev_vec4s, time);


## Dispatches sparse SOI-handoff, inline cross-SOI, and CPU-fallback events returned by `ballistic_tape_sample.glsl`.
func _apply_gpu_tape_sample_events(ev_vec4s: PackedVector4Array, eval_t: float) -> void:
	if ev_vec4s.is_empty():
		return;
	var b_seq: Array[JrBallisticOrbiter] = _ballistic_sequence;
	var count: int = b_seq.size();
	var well_count: int = _runtime_well_elements.size();
	var stride: int = JrBallisticComputeSolver.TAPE_SAMPLE_EVENT_VEC4S;
	@warning_ignore("integer_division")
	var ev_count: int = ev_vec4s.size() / stride;
	for e: int in range(ev_count):
		var base: int = e * stride;
		var hdr: Vector4 = ev_vec4s[base];
		var b_idx: int = int(roundf(hdr.x));
		if b_idx < 0 or b_idx >= count:
			continue;
		var kind: int = int(roundf(hdr.y));
		var b_ev: JrBallisticOrbiter = b_seq[b_idx];
		if kind == JrBallisticComputeSolver.TAPE_EVENT_WELL_SWITCH:
			var w_idx: int = int(roundf(hdr.z));
			if w_idx >= 0 and w_idx != b_ev._tape_anchor_idx:
				var next_anchor: JrOrreryElement = (
						_runtime_well_elements[w_idx]
						if w_idx < well_count
						else self
				);
				b_ev.adopt_tape_anchor_element(next_anchor, w_idx);
				_ballistic_solver.update_body_expected_well_in_atlas(b_idx, w_idx);
		elif kind == JrBallisticComputeSolver.TAPE_EVENT_CROSS_SOI:
			_apply_inline_gpu_cross_soi_event(b_ev, eval_t, ev_vec4s, base);
		elif kind == JrBallisticComputeSolver.TAPE_EVENT_CPU_FALLBACK:
			_sample_single_ballistic_from_tape(b_ev, eval_t);
			_ballistic_solver.update_body_expected_well_in_atlas(b_idx, b_ev._tape_anchor_idx);


## Evaluates Cubic Hermite `(r_rel, v_rel)` across a cross-SOI segment directly from inline GPU
## event knots `(pa, pb, va, vb)` without downloading the body's 4,352-byte VRAM atlas slot.
func _apply_inline_gpu_cross_soi_event(
		ballistic: JrBallisticOrbiter,
		eval_t: float,
		ev_vec4s: PackedVector4Array,
		ev_base: int
) -> void:
	var tape: JrBallisticTrajectoryTape = ballistic._trajectory_tape;
	if not tape.is_valid:
		return;
	var hdr: Vector4 = ev_vec4s[ev_base];
	var b_idx: int = int(roundf(hdr.x));
	var idx_a: int = int(roundf(hdr.z));
	var u: float = hdr.w;
	var pa: Vector4 = ev_vec4s[ev_base + 1];
	var pb: Vector4 = ev_vec4s[ev_base + 2];
	var va: Vector4 = ev_vec4s[ev_base + 3];
	var vb: Vector4 = ev_vec4s[ev_base + 4];

	var well_limit: int = maxi(0, _runtime_well_elements.size() - 1);
	var well_a_idx: int = clampi(int(roundf(pa.w)), 0, well_limit);
	var well_b_idx: int = clampi(int(roundf(pb.w)), 0, well_limit);
	var well_a: JrOrreryElement = _runtime_well_elements[well_a_idx];
	var well_b: JrOrreryElement = _runtime_well_elements[well_b_idx];

	var t_a: float = tape.start_time + float(idx_a) * tape.knot_dt;
	var pa_aligned := Vector3(pa.x, pa.y, pa.z) + _compute_lca_well_position_difference_at(well_a, well_b, t_a);
	var va_aligned := Vector3(va.x, va.y, va.z) + _compute_lca_well_velocity_difference_at(well_a, well_b, t_a);
	var pb3 := Vector3(pb.x, pb.y, pb.z);
	var vb3 := Vector3(vb.x, vb.y, vb.z);

	var u2: float = u * u;
	var u3: float = u2 * u;
	var h: float = tape.knot_dt;
	var r_rel: Vector3 = (
			pa_aligned * (2.0 * u3 - 3.0 * u2 + 1.0)
			+ va_aligned * ((u3 - 2.0 * u2 + u) * h)
			+ pb3 * (-2.0 * u3 + 3.0 * u2)
			+ vb3 * ((u3 - u2) * h)
	);
	var dh00: float = (6.0 * u2 - 6.0 * u) * tape.inv_knot_dt;
	var v_rel: Vector3 = (
			(pa_aligned - pb3) * dh00
			+ va_aligned * (3.0 * u2 - 4.0 * u + 1.0)
			+ vb3 * (3.0 * u2 - 2.0 * u)
	);

	ballistic.adopt_tape_anchor_element(well_b, well_b_idx);
	ballistic._state_time = eval_t;
	if not is_equal_approx(eval_t, time):
		ballistic._obeying_shared_tape_clock = false;
	ballistic.apply_sampled_tape_state(r_rel, v_rel);
	_ballistic_solver.update_body_expected_well_in_atlas(b_idx, well_b_idx);


## Evaluates CPU Cubic Hermite sampling for a single [param ballistic] delegated by GPU fallback
## (e.g., cross-SOI segment or long-horizon forecast tape exceeding 264 knots).
func _sample_single_ballistic_from_tape(ballistic: JrBallisticOrbiter, eval_t: float) -> void:
	var tape: JrBallisticTrajectoryTape = ballistic._trajectory_tape;
	if not tape.is_valid:
		return;
	if not tape.cpu_knots_resident:
		tape.ensure_cpu_knots();

	var rel_t: float = eval_t - tape.start_time;
	if tape.is_periodic_loop:
		rel_t = fposmod(rel_t, tape.loop_period);
	var max_idx: float = float(tape.knot_count - 1);
	var scaled: float = clampf(rel_t * tape.inv_knot_dt, 0.0, max_idx);
	var idx_a: int = mini(int(scaled), tape.knot_count - 2);
	var u: float = scaled - float(idx_a);

	var pa: Vector4 = tape.pos_anchor[idx_a];
	var r_rel: Vector3;
	var w_idx: int;
	if u <= 1e-5:
		r_rel = Vector3(pa.x, pa.y, pa.z);
		w_idx = int(pa.w);
	else:
		var pb: Vector4 = tape.pos_anchor[idx_a + 1];
		if pa.w != pb.w:
			_sample_ballistic_cross_soi_at(ballistic, eval_t);
			return;
		if u >= 0.99999:
			r_rel = Vector3(pb.x, pb.y, pb.z);
			w_idx = int(pb.w);
		else:
			var dt_k: float = tape.knot_dt;
			var u2: float = u * u;
			var u3: float = u2 * u;
			var h00: float = 2.0 * u3 - 3.0 * u2 + 1.0;
			var h10: float = (u3 - 2.0 * u2 + u) * dt_k;
			var h01: float = -2.0 * u3 + 3.0 * u2;
			var h11: float = (u3 - u2) * dt_k;
			var p4: Vector4 = (
					pa * h00
					+ tape.vel_flags[idx_a] * h10
					+ pb * h01
					+ tape.vel_flags[idx_a + 1] * h11
			);
			r_rel = Vector3(p4.x, p4.y, p4.z);
			w_idx = int(pa.w);

	if _reference_plane != null and _reference_plane.lock_to_ecliptic:
		r_rel = _reference_plane.project_vector_to_ecliptic(r_rel);
	ballistic._anchor_relative_position = r_rel;
	ballistic._pos_materialized_pass_id = _sampled_time_cell[1];
	if is_equal_approx(eval_t, time):
		ballistic._obeying_shared_tape_clock = true;
	if w_idx != ballistic._tape_anchor_idx:
		var well_count: int = _runtime_well_elements.size();
		var new_anchor: JrOrreryElement = (
				_runtime_well_elements[w_idx]
				if w_idx < well_count
				else self
		);
		ballistic.adopt_tape_anchor_element(new_anchor, w_idx);


## Evaluates Cubic Hermite `(r_rel, v_rel)` across a Sphere-of-Influence handoff segment by
## shifting knot `idx_a` into knot `idx_a + 1`'s gravity-well frame at `t_a`.
func _sample_ballistic_cross_soi_at(ballistic: JrBallisticOrbiter, eval_time: float) -> void:
	var tape: JrBallisticTrajectoryTape = ballistic._trajectory_tape;
	if not tape.cpu_knots_resident:
		tape.ensure_cpu_knots();
	var rel_t: float = eval_time - tape.start_time;
	if tape.is_periodic_loop:
		rel_t = fposmod(rel_t, tape.loop_period);
	var max_idx: float = float(tape.knot_count - 1);
	var scaled: float = clampf(rel_t * tape.inv_knot_dt, 0.0, max_idx);
	var idx_a: int = mini(int(scaled), tape.knot_count - 2);
	var u: float = scaled - float(idx_a);

	var pa: Vector4 = tape.pos_anchor[idx_a];
	var pb: Vector4 = tape.pos_anchor[idx_a + 1];
	var va: Vector4 = tape.vel_flags[idx_a];
	var well_limit: int = maxi(0, _runtime_well_elements.size() - 1);
	var well_a_idx: int = clampi(int(roundf(pa.w)), 0, well_limit);
	var well_b_idx: int = clampi(int(roundf(pb.w)), 0, well_limit);
	var well_a: JrOrreryElement = _runtime_well_elements[well_a_idx];
	var well_b: JrOrreryElement = _runtime_well_elements[well_b_idx];

	var t_a: float = tape.start_time + float(idx_a) * tape.knot_dt;
	var pos_a_to_b: Vector3 = _compute_lca_well_position_difference_at(well_a, well_b, t_a);
	var vel_a_to_b: Vector3 = _compute_lca_well_velocity_difference_at(well_a, well_b, t_a);
	var pa_aligned := Vector3(pa.x, pa.y, pa.z) + pos_a_to_b;
	var va_aligned := Vector3(va.x, va.y, va.z) + vel_a_to_b;

	ballistic.adopt_tape_anchor_element(well_b, well_b_idx);
	ballistic._state_time = eval_time;
	if not is_equal_approx(eval_time, time):
		ballistic._obeying_shared_tape_clock = false;
	ballistic.apply_sampled_cross_soi_state(idx_a, u, pa_aligned, va_aligned);


## Evaluates Hill-sphere hysteresis (`0.98x` entry / `1.05x` exit) after stepping
## for all unpinned [JrBallisticOrbiter] nodes in [param bodies].
func _update_ballistic_anchors_after_step(
		bodies: Array[JrBallisticOrbiter],
		well_elements: Array[JrOrreryElement]
) -> void:
	if _orbit_sequence.is_empty() or bodies.is_empty():
		return;
	if (
			_ballistic_solver.has_gpu_step_handoffs()
			and well_elements.size() == _orbit_sequence.size() + 1
	):
		_apply_gpu_step_handoffs(bodies, well_elements);
		return;

	var tables_built: bool = false;
	for ballistic: JrBallisticOrbiter in bodies:
		if ballistic.pinned_anchor != null:
			continue;
		if not tables_built:
			_build_step_hill_handoff_tables();
			tables_built = true;
		_evaluate_hill_sphere_handoff_fast(ballistic);


## Applies only the sparse Hill-sphere anchor transitions reported by the GPU stepping shader.
func _apply_gpu_step_handoffs(
		bodies: Array[JrBallisticOrbiter],
		well_elements: Array[JrOrreryElement]
) -> void:
	var body_indices: PackedInt32Array = _ballistic_solver.get_gpu_handoff_body_indices();
	if body_indices.is_empty():
		return;
	var target_wells: PackedInt32Array = _ballistic_solver.get_gpu_handoff_target_wells();
	var well_count: int = well_elements.size();
	var b_count: int = bodies.size();
	var self_elem: JrOrreryElement = self;
	var any_switched: bool = false;

	for h: int in range(body_indices.size()):
		var b_idx: int = body_indices[h];
		var w_idx: int = target_wells[h];
		if b_idx < 0 or b_idx >= b_count or w_idx < 0 or w_idx >= well_count:
			continue;
		var ballistic: JrBallisticOrbiter = bodies[b_idx];
		if ballistic.pinned_anchor != null:
			continue;
		var curr_anchor: JrOrreryElement = ballistic.get_anchor_element();
		var target_anchor: JrOrreryElement = well_elements[w_idx];
		if curr_anchor == null or target_anchor == null or curr_anchor == target_anchor:
			continue;

		any_switched = true;
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

	if any_switched:
		_ballistic_solver.invalidate_resident_step_buffer();


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


## Computes the analytical velocity difference [code]v_a(eval_time) - v_b(eval_time)[/code]
## between two gravity-well elements at [param eval_time] using LCA traversal.
func _compute_lca_well_velocity_difference_at(
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
				accum_b += orb_b.get_relative_velocity(eval_time, _reference_plane);
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
				accum_a += orb_a.get_relative_velocity(eval_time, _reference_plane);
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
	var cfg := JrBallisticForecastConfig.new();
	cfg.bodies = bodies;
	cfg.start_time = start_time;
	cfg.driver_gravity_strength = _gravity_strength;
	cfg.tidal_cull_threshold = tidal_cull_threshold;
	cfg.driver_element = self;
	cfg.reference_plane = _reference_plane;
	cfg.orbit_sequence = _orbit_sequence;
	cfg.orbiter_parents = _children;
	if not _runtime_wells_dirty and not _runtime_well_elements.is_empty() and not Engine.is_editor_hint():
		cfg.prebuilt_wells = _runtime_well_elements;
	return cfg;


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


## Projects forward-looking forecast polylines from each target's rolling
## [JrBallisticTrajectoryTape] into the active [member perspective_node] frame.
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

	_flush_pending_gpu_tape_sample();
	_refresh_perspective_cache_epoch();
	_begin_forecast_projection_pass(system_basis);
	for target: JrBallisticOrbiter in targets:
		_project_tape_forecast_to_perspective(target, system_basis);


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

	_flush_pending_gpu_tape_sample();
	var t_proj0: int = Time.get_ticks_usec();
	_refresh_perspective_cache_epoch();
	_begin_forecast_projection_pass(system_basis);
	for target: JrBallisticOrbiter in targets:
		_project_tape_forecast_to_perspective(target, system_basis);
	_telemetry.add_phase_usec(
			JrOrreryTelemetry.PHASE_FORECAST_PERSPECTIVE_PROJECT,
			Time.get_ticks_usec() - t_proj0
	);


## Collects all [JrBallisticOrbiter] nodes requiring forecast evaluation in `O(K_forecast)`.
func _collect_forecast_targets() -> Array[JrBallisticOrbiter]:
	return _forecast_enabled_ballistics;


## Increments [member _persp_cache_epoch] whenever the active perspective node, its anchor well,
## or the effective perspective rotation mode changes.
func _refresh_perspective_cache_epoch() -> void:
	var is_helio: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	var target_id: int = 0 if is_helio else int(perspective_node.get_instance_id());
	var p_well: JrOrreryElement = self if is_helio else _get_perspective_gravity_well();
	var well_id: int = int(p_well.get_instance_id());
	var eff_mode: int = (
			int(PerspectiveRotationMode.INERTIAL)
			if is_helio
			else int(perspective_rotation_mode)
	);
	if (
			target_id != _last_persp_target_id
			or well_id != _last_persp_well_id
			or eff_mode != _last_persp_rot_mode
			or (
					not is_helio
					and perspective_node is JrBallisticOrbiter
					and perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
			)
	):
		_last_persp_target_id = target_id;
		_last_persp_well_id = well_id;
		_last_persp_rot_mode = eff_mode;
		_persp_cache_epoch += 1;


## Initializes per-frame common-ancestor metadata for all runtime gravity wells so
## [method _project_tape_forecast_to_perspective] can share knot evaluations across bodies.
func _begin_forecast_projection_pass(system_basis: Basis) -> void:
	if _runtime_wells_dirty or _runtime_well_elements.is_empty():
		_rebuild_runtime_well_cache();

	_fc_grid_start_time = NAN;
	_fc_grid_knot_dt = -1.0;
	_fc_proj_synodic_bases.clear();

	var well_count: int = maxi(1, _runtime_well_elements.size());
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


## Ensures shared synodic inverse bases are populated for `(sub_t0, step_dt)` up to [param count],
## growing incrementally across cohorts sharing the same start timestamp and stride.
func _ensure_forecast_projection_grid(
		sub_t0: float,
		step_dt: float,
		count: int,
		need_synodic: bool
) -> void:
	if (
			is_nan(_fc_grid_start_time)
			or not is_equal_approx(_fc_grid_start_time, sub_t0)
			or not is_equal_approx(_fc_grid_knot_dt, step_dt)
	):
		_fc_grid_start_time = sub_t0;
		_fc_grid_knot_dt = step_dt;
		_fc_proj_synodic_bases.clear();
		_fc_proj_well_origins_ready.fill(0);
		_fc_proj_well_offsets_ready.fill(0);

	if need_synodic and _fc_proj_synodic_bases.size() < count:
		var prev_len: int = _fc_proj_synodic_bases.size();
		_fc_proj_synodic_bases.resize(count);
		for k: int in range(prev_len, count):
			var eval_t: float = sub_t0 + float(k) * step_dt;
			_fc_proj_synodic_bases[k] = _compute_synodic_perspective_basis_at(eval_t);


## Lazily evaluates and grows inertial origins and LCA offsets across `[0 .. count - 1]` from
## [param sub_t0] at stride [param step_dt] for [param well_idx].
func _ensure_forecast_well_arrays(
		well_idx: int,
		sub_t0: float,
		step_dt: float,
		count: int
) -> void:
	var anchor_elem: JrOrreryElement = _get_forecast_well_element(well_idx);
	if _fc_proj_well_origins_ready[well_idx] == 0:
		var origins := PackedVector3Array();
		origins.resize(count);
		if anchor_elem != null and anchor_elem != self:
			for k: int in range(count):
				origins[k] = evaluate_element_driver_origin_at(
						anchor_elem,
						sub_t0 + float(k) * step_dt
				);
		_fc_proj_well_origins[well_idx] = origins;
		_fc_proj_well_origins_ready[well_idx] = 1;
	elif _fc_proj_well_origins[well_idx].size() < count:
		var origins: PackedVector3Array = _fc_proj_well_origins[well_idx];
		var prev_n: int = origins.size();
		origins.resize(count);
		if anchor_elem != null and anchor_elem != self:
			for k: int in range(prev_n, count):
				origins[k] = evaluate_element_driver_origin_at(
						anchor_elem,
						sub_t0 + float(k) * step_dt
				);
		_fc_proj_well_origins[well_idx] = origins;

	var common_well: JrOrreryElement = _fc_proj_well_common[well_idx];
	if _fc_proj_well_offsets_ready[well_idx] == 0:
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
				offsets[k] = _compute_lca_well_position_difference_at(
						anchor_elem,
						common_well,
						sub_t0 + float(k) * step_dt
				);
			_fc_proj_well_offsets[well_idx] = offsets;
		_fc_proj_well_offsets_ready[well_idx] = 1;
	elif _fc_proj_well_offsets[well_idx].size() < count:
		if anchor_elem == common_well:
			var zero_offsets: PackedVector3Array = _fc_proj_well_offsets[well_idx];
			zero_offsets.resize(count);
			_fc_proj_well_offsets[well_idx] = zero_offsets;
		elif common_well == self:
			_fc_proj_well_offsets[well_idx] = _fc_proj_well_origins[well_idx];
		else:
			var offsets: PackedVector3Array = _fc_proj_well_offsets[well_idx];
			var prev_o: int = offsets.size();
			offsets.resize(count);
			for k: int in range(prev_o, count):
				offsets[k] = _compute_lca_well_position_difference_at(
						anchor_elem,
						common_well,
						sub_t0 + float(k) * step_dt
				);
			_fc_proj_well_offsets[well_idx] = offsets;


## Populates any unbuilt world-inertial knot positions `[inertial_built_count .. knot_count - 1]`
## on [param tape].
func _ensure_tape_inertial_knots(tape: JrBallisticTrajectoryTape) -> void:
	if tape.inertial_built_count >= tape.knot_count:
		return;
	if not tape.cpu_knots_resident:
		tape.ensure_cpu_knots();
	var from_k: int = tape.inertial_built_count;
	var sub_count: int = tape.knot_count - from_k;
	var sub_t0: float = tape.start_time + float(from_k) * tape.knot_dt;
	_ensure_forecast_projection_grid(sub_t0, tape.knot_dt, sub_count, false);

	var well_limit: int = maxi(0, _fc_proj_well_common.size() - 1);
	var pts: PackedVector3Array = tape.inertial_points;
	pts.resize(tape.knot_count);
	var last_w: int = -1;
	var origins := PackedVector3Array();

	for local_k: int in range(sub_count):
		var k: int = from_k + local_k;
		var p4: Vector4 = tape.pos_anchor[k];
		var w_idx: int = clampi(int(roundf(p4.w)), 0, well_limit);
		if w_idx != last_w:
			last_w = w_idx;
			if w_idx > 0:
				_ensure_forecast_well_arrays(w_idx, sub_t0, tape.knot_dt, sub_count);
				origins = _fc_proj_well_origins[w_idx];
		if last_w == 0:
			pts[k] = Vector3(p4.x, p4.y, p4.z);
		else:
			pts[k] = origins[local_k] + Vector3(p4.x, p4.y, p4.z);

	tape.inertial_points = pts;
	tape.inertial_built_count = tape.knot_count;


## Populates any unbuilt time-invariant perspective vectors `[persp_built_count .. knot_count - 1]`
## on [param tape] for the active [member _persp_cache_epoch].
func _ensure_tape_perspective_knots(tape: JrBallisticTrajectoryTape) -> void:
	if tape.persp_epoch != _persp_cache_epoch:
		tape.persp_epoch = _persp_cache_epoch;
		tape.persp_points.clear();
		tape.persp_built_count = 0;
	if tape.persp_built_count >= tape.knot_count:
		return;
	if not tape.cpu_knots_resident:
		tape.ensure_cpu_knots();

	var from_k: int = tape.persp_built_count;
	var sub_count: int = tape.knot_count - from_k;
	var sub_t0: float = tape.start_time + float(from_k) * tape.knot_dt;
	var need_syn: bool = (perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL);
	_ensure_forecast_projection_grid(sub_t0, tape.knot_dt, sub_count, need_syn);

	var well_limit: int = maxi(0, _fc_proj_well_common.size() - 1);
	var p_pts: PackedVector3Array = tape.persp_points;
	p_pts.resize(tape.knot_count);
	var last_w: int = -1;
	var offsets := PackedVector3Array();
	var curr_synodic: bool = false;

	for local_k: int in range(sub_count):
		var k: int = from_k + local_k;
		var p4: Vector4 = tape.pos_anchor[k];
		var w_idx: int = clampi(int(roundf(p4.w)), 0, well_limit);
		if w_idx != last_w:
			last_w = w_idx;
			_ensure_forecast_well_arrays(w_idx, sub_t0, tape.knot_dt, sub_count);
			offsets = _fc_proj_well_offsets[w_idx];
			curr_synodic = (_fc_proj_well_is_synodic[w_idx] != 0);
		var u_vec: Vector3 = offsets[local_k] + Vector3(p4.x, p4.y, p4.z);
		if curr_synodic:
			p_pts[k] = _fc_proj_synodic_bases[local_k] * u_vec;
		else:
			p_pts[k] = u_vec;

	tape.persp_points = p_pts;
	tape.persp_built_count = tape.knot_count;


## Slices `[time, time + preview_duration]` out of [param ballistic]'s [JrBallisticTrajectoryTape]
## and transforms contiguous SOI spans into the active perspective frame using native C++ SIMD
## array operations (`Transform3D * PackedVector3Array`), pinning vertex `0` to the body's live
## transform and the final vertex to the exact Cubic Hermite tip at `time + preview_duration`.
func _project_tape_forecast_to_perspective(
		ballistic: JrBallisticOrbiter,
		system_basis: Basis
) -> void:
	var tape: JrBallisticTrajectoryTape = ballistic._trajectory_tape;
	if not tape.is_valid or tape.knot_count < 2:
		if ballistic.has_runtime_forecast_points():
			ballistic.set_runtime_forecast_points(
					PackedVector3Array(),
					PackedVector3Array(),
					PackedVector3Array()
			);
		return;

	_ensure_tape_inertial_knots(tape);
	tape.ensure_soi_spans();

	var is_heliocentric: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	var identity_basis: bool = (
			system_basis == Basis.IDENTITY or system_basis.is_equal_approx(Basis.IDENTITY)
	);
	var well_limit: int = maxi(0, _fc_proj_well_common.size() - 1);
	var tip_time: float = time + ballistic.preview_duration;
	_evaluate_tape_tip_points(ballistic, tip_time, system_basis);

	var first_inner: int = maxi(
			0,
			int(floorf((time - tape.start_time) * tape.inv_knot_dt)) + 1
	);
	var last_inner: int = mini(
			tape.knot_count - 1,
			int(ceilf((tip_time - tape.start_time) * tape.inv_knot_dt)) - 1
	);

	var head_inertial: Vector3 = ballistic.get_driver_position();
	var head_well_idx: int = clampi(ballistic.get_anchor_well_index(), 0, well_limit);
	var head_id: int = _fc_proj_well_common_id[head_well_idx];

	var inertial_pts := PackedVector3Array();
	inertial_pts.push_back(head_inertial);
	if first_inner <= last_inner:
		inertial_pts.append_array(tape.inertial_points.slice(first_inner, last_inner + 1));
	inertial_pts.push_back(_tip_inertial_pt);

	var ref_ids := PackedInt64Array();
	ref_ids.push_back(head_id);

	if is_heliocentric and identity_basis:
		if first_inner <= last_inner:
			var span_count: int = tape.span_wells.size();
			for s: int in range(span_count):
				var k0: int = maxi(first_inner, tape.span_starts[s]);
				var k1: int = mini(last_inner, tape.span_ends[s]);
				if k0 > k1:
					continue;
				var w_idx: int = clampi(tape.span_wells[s], 0, well_limit);
				var seg_ids := PackedInt64Array();
				seg_ids.resize(k1 - k0 + 1);
				seg_ids.fill(_fc_proj_well_common_id[w_idx]);
				ref_ids.append_array(seg_ids);
		ref_ids.push_back(_tip_common_id);
		ballistic.set_runtime_forecast_points(inertial_pts, inertial_pts, inertial_pts, ref_ids);
		_any_ballistic_has_forecast_points = true;
		return;

	_ensure_tape_perspective_knots(tape);
	var is_synodic: bool = (
			not is_heliocentric
			and perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
	);
	var synodic_now_basis: Basis = system_basis.transposed() if is_synodic else Basis.IDENTITY;
	var head_unrot: Vector3 = _lookup_ballistic_perspective_offset(ballistic);
	var head_sys: Vector3 = head_unrot if identity_basis else (system_basis * head_unrot);

	var unrotated_pts := PackedVector3Array();
	var system_pts := PackedVector3Array();
	unrotated_pts.push_back(head_unrot);
	system_pts.push_back(head_sys);

	if first_inner <= last_inner:
		var span_count: int = tape.span_wells.size();
		for s: int in range(span_count):
			var k0: int = maxi(first_inner, tape.span_starts[s]);
			var k1: int = mini(last_inner, tape.span_ends[s]);
			if k0 > k1:
				continue;
			var w_idx: int = clampi(tape.span_wells[s], 0, well_limit);
			var seg_ids := PackedInt64Array();
			seg_ids.resize(k1 - k0 + 1);
			seg_ids.fill(_fc_proj_well_common_id[w_idx]);
			ref_ids.append_array(seg_ids);

			var persp_slice: PackedVector3Array = tape.persp_points.slice(k0, k1 + 1);
			if _fc_proj_well_is_synodic[w_idx] != 0:
				var sys_c: Vector3 = _fc_proj_well_sys_common_now[w_idx];
				var sys_seg: PackedVector3Array = (
						persp_slice
						if sys_c == Vector3.ZERO
						else (Transform3D(Basis.IDENTITY, sys_c) * persp_slice)
				);
				var unrot_seg: PackedVector3Array = (
						Transform3D(synodic_now_basis, _fc_proj_well_common_now[w_idx]) * persp_slice
				);
				system_pts.append_array(sys_seg);
				unrotated_pts.append_array(unrot_seg);
			else:
				var com_c: Vector3 = _fc_proj_well_common_now[w_idx];
				var unrot_seg: PackedVector3Array = (
						persp_slice
						if com_c == Vector3.ZERO
						else (Transform3D(Basis.IDENTITY, com_c) * persp_slice)
				);
				var sys_seg: PackedVector3Array = (
						unrot_seg
						if identity_basis
						else (Transform3D(system_basis, _fc_proj_well_sys_common_now[w_idx]) * persp_slice)
				);
				unrotated_pts.append_array(unrot_seg);
				system_pts.append_array(sys_seg);

	ref_ids.push_back(_tip_common_id);
	unrotated_pts.push_back(_tip_unrot_pt);
	system_pts.push_back(_tip_sys_pt);

	ballistic.set_runtime_forecast_points(inertial_pts, unrotated_pts, system_pts, ref_ids);
	_any_ballistic_has_forecast_points = true;


## Evaluates the exact Cubic Hermite forecast tip at [param tip_time] into [member _tip_inertial_pt],
## [member _tip_unrot_pt], [member _tip_sys_pt], and [member _tip_common_id].
func _evaluate_tape_tip_points(
		ballistic: JrBallisticOrbiter,
		tip_time: float,
		system_basis: Basis
) -> void:
	var well_limit: int = maxi(0, _fc_proj_well_common.size() - 1);
	var tip4: Vector4 = ballistic.sample_tape_pos_and_well_at(tip_time);
	var r_tip: Vector3;
	var tip_well_idx: int;

	if tip4.w >= -0.5:
		r_tip = Vector3(tip4.x, tip4.y, tip4.z);
		tip_well_idx = clampi(int(roundf(tip4.w)), 0, well_limit);
	else:
		var tape: JrBallisticTrajectoryTape = ballistic._trajectory_tape;
		if not tape.cpu_knots_resident:
			tape.ensure_cpu_knots();
		var rel_t: float = tip_time - tape.start_time;
		var max_idx: float = float(tape.knot_count - 1);
		var scaled: float = clampf(rel_t * tape.inv_knot_dt, 0.0, max_idx);
		var idx_a: int = mini(int(scaled), tape.knot_count - 2);
		var u: float = scaled - float(idx_a);
		var pa: Vector4 = tape.pos_anchor[idx_a];
		var pb: Vector4 = tape.pos_anchor[idx_a + 1];
		var va: Vector4 = tape.vel_flags[idx_a];
		var wa_idx: int = clampi(int(roundf(pa.w)), 0, well_limit);
		tip_well_idx = clampi(int(roundf(pb.w)), 0, well_limit);
		var wa_elem: JrOrreryElement = _get_forecast_well_element(wa_idx);
		var wb_elem: JrOrreryElement = _get_forecast_well_element(tip_well_idx);
		var t_a: float = tape.start_time + float(idx_a) * tape.knot_dt;
		var pos_ab: Vector3 = _compute_lca_well_position_difference_at(wa_elem, wb_elem, t_a);
		var vel_ab: Vector3 = _compute_lca_well_velocity_difference_at(wa_elem, wb_elem, t_a);
		r_tip = ballistic.evaluate_tape_pos_aligned(
				idx_a,
				u,
				Vector3(pa.x, pa.y, pa.z) + pos_ab,
				Vector3(va.x, va.y, va.z) + vel_ab
		);

	var anchor_elem: JrOrreryElement = _get_forecast_well_element(tip_well_idx);
	_tip_common_id = _fc_proj_well_common_id[tip_well_idx];
	_tip_inertial_pt = evaluate_element_driver_origin_at(anchor_elem, tip_time) + r_tip;

	var is_heliocentric: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	);
	var identity_basis: bool = (
			system_basis == Basis.IDENTITY or system_basis.is_equal_approx(Basis.IDENTITY)
	);
	if is_heliocentric and identity_basis:
		_tip_unrot_pt = _tip_inertial_pt;
		_tip_sys_pt = _tip_inertial_pt;
		return;

	var common_well: JrOrreryElement = _fc_proj_well_common[tip_well_idx];
	var offset_from_common: Vector3 = (
			_compute_lca_well_position_difference_at(anchor_elem, common_well, tip_time) + r_tip
	);
	if _fc_proj_well_is_synodic[tip_well_idx] != 0:
		var syn_inv: Basis = _compute_synodic_perspective_basis_at(tip_time);
		_tip_sys_pt = _fc_proj_well_sys_common_now[tip_well_idx] + (syn_inv * offset_from_common);
		_tip_unrot_pt = system_basis.transposed() * _tip_sys_pt;
	else:
		_tip_unrot_pt = _fc_proj_well_common_now[tip_well_idx] + offset_from_common;
		_tip_sys_pt = _tip_unrot_pt if identity_basis else (system_basis * _tip_unrot_pt);


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


## Returns the [JrOrreryElement] at [param well_idx] in [member _runtime_well_elements].
func _get_forecast_well_element(well_idx: int) -> JrOrreryElement:
	if _runtime_wells_dirty or _runtime_well_elements.is_empty():
		if _orbit_sequence.is_empty():
			return self;
		_rebuild_runtime_well_cache();
	var clamped: int = clampi(well_idx, 0, _runtime_well_elements.size() - 1);
	return _runtime_well_elements[clamped];


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
	);
	invalidate_all_ballistic_trajectories();
	if is_inside_tree():
		update_orbits();
	orbit_visuals_changed.emit();


## Resolves the original spawn parent [JrOrreryElement] for [param ballistic] (preserving
## its orbital focus even after [method collect_ballistic_nodes] reparents it to this driver).
func _resolve_ballistic_spawn_parent(ballistic: JrBallisticOrbiter) -> JrOrreryElement:
	var spawn_p: JrOrreryElement = ballistic.get_spawn_parent();
	if spawn_p != null and _is_tracked_or_self(spawn_p):
		return spawn_p;
	return _element_parents.get(ballistic, self);


## Returns the heliocentric driver-relative [Transform3D] of [param element], or
## [constant Transform3D.IDENTITY] when [param element] is this driver.
func _get_element_driver_transform(element: JrOrreryElement) -> Transform3D:
	if element != null and element != self:
		return element.get_driver_relative_transform();
	return Transform3D.IDENTITY;


## Returns the global inertial velocity of [param element], or [constant Vector3.ZERO]
## when [param element] is this driver.
func _get_element_global_velocity(element: JrOrreryElement) -> Vector3:
	if element != null and element != self:
		return element.get_global_velocity();
	return Vector3.ZERO;


## Returns the heliocentric driver-relative [Transform3D] of [param element]'s
## immediate parent, or [constant Transform3D.IDENTITY] when the parent is this driver.
func _get_parent_driver_transform_for(element: JrOrreryElement) -> Transform3D:
	var parent_elem: JrOrreryElement = _element_parents.get(element, null);
	return _get_element_driver_transform(parent_elem);


## Returns the accumulated global inertial velocity vector of [param element]'s
## immediate parent in this driver's reference frame, or [constant Vector3.ZERO]
## when the parent is this driver.
func _get_parent_global_velocity_for(element: JrOrreryElement) -> Vector3:
	var parent_elem: JrOrreryElement = _element_parents.get(element, null);
	return _get_element_global_velocity(parent_elem);


## Computes the active perspective rotation [Basis] at the current simulation [member time].
func _compute_perspective_basis() -> Basis:
	if Engine.is_editor_hint():
		return Basis.IDENTITY;
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return Basis.IDENTITY;

	if perspective_rotation_mode == PerspectiveRotationMode.INERTIAL:
		return Basis.IDENTITY;
	if perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL:
		return _compute_synodic_perspective_basis_at(time);

	var target_transform: Transform3D = perspective_node.get_driver_relative_transform();
	return target_transform.basis.transposed().orthonormalized();


## Computes the inverse synodic orbital [Basis] at [param eval_time] that locks the
## primary-to-[member perspective_node] radius vector along the positive X-axis and the
## orbital angular momentum along the ecliptic north pole (+Y).
func _compute_synodic_perspective_basis_at(eval_time: float) -> Basis:
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return Basis.IDENTITY;

	var r_orb := Vector3.ZERO;
	var v_orb := Vector3.ZERO;

	if perspective_node is JrRailOrbiter:
		var orb := perspective_node as JrRailOrbiter;
		if is_equal_approx(eval_time, time):
			r_orb = orb.get_focus_relative_position();
			v_orb = orb.get_relative_velocity();
		else:
			r_orb = orb.get_relative_position(eval_time, _reference_plane);
			v_orb = orb.get_relative_velocity(eval_time, _reference_plane);
	elif perspective_node is JrBallisticOrbiter:
		var b_node := perspective_node as JrBallisticOrbiter;
		var b_tape: JrBallisticTrajectoryTape = b_node._trajectory_tape;
		if not is_equal_approx(eval_time, time) and b_tape.contains_time(eval_time):
			if not b_tape.cpu_knots_resident:
				b_tape.ensure_cpu_knots();
			var rel_t: float = eval_time - b_tape.start_time;
			if b_tape.is_periodic_loop:
				rel_t = fposmod(rel_t, b_tape.loop_period);
			var scaled: float = clampf(rel_t * b_tape.inv_knot_dt, 0.0, float(b_tape.knot_count - 1));
			var idx_a: int = mini(int(scaled), b_tape.knot_count - 2);
			var u: float = scaled - float(idx_a);
			var pa4: Vector4 = b_tape.pos_anchor[idx_a];
			var pb4: Vector4 = b_tape.pos_anchor[idx_a + 1];
			var va4: Vector4 = b_tape.vel_flags[idx_a];
			var vb4: Vector4 = b_tape.vel_flags[idx_a + 1];
			if is_equal_approx(pa4.w, pb4.w):
				var p0 := Vector3(pa4.x, pa4.y, pa4.z);
				var m0: Vector3 = Vector3(va4.x, va4.y, va4.z) * b_tape.knot_dt;
				var p1 := Vector3(pb4.x, pb4.y, pb4.z);
				var m1: Vector3 = Vector3(vb4.x, vb4.y, vb4.z) * b_tape.knot_dt;
				var u2: float = u * u;
				var u3: float = u2 * u;
				r_orb = (
						(2.0 * u3 - 3.0 * u2 + 1.0) * p0
						+ (u3 - 2.0 * u2 + u) * m0
						+ (-2.0 * u3 + 3.0 * u2) * p1
						+ (u3 - u2) * m1
				);
				v_orb = (
						(6.0 * u2 - 6.0 * u) * (p0 - p1)
						+ (3.0 * u2 - 4.0 * u + 1.0) * m0
						+ (3.0 * u2 - 2.0 * u) * m1
				) * b_tape.inv_knot_dt;
			else:
				r_orb = b_node.get_anchor_relative_position();
				v_orb = b_node.get_anchor_relative_velocity();
		else:
			r_orb = b_node.get_anchor_relative_position();
			v_orb = b_node.get_anchor_relative_velocity();

	var r_len: float = r_orb.length();
	if r_len <= 1e-8:
		return Basis.IDENTITY;

	var z_back: Vector3 = -r_orb / r_len;
	var ref_up: Vector3 = (
			_reference_plane.ecliptic_north_pole
			if _reference_plane != null
			else Vector3.UP
	);
	var h_vec: Vector3 = r_orb.cross(v_orb);
	var y_normal: Vector3 = h_vec.normalized() if h_vec.length_squared() > 1e-12 else ref_up;
	if y_normal.dot(ref_up) < 0.0:
		y_normal = -y_normal;
	if absf(z_back.dot(y_normal)) > 0.999:
		y_normal = Vector3.RIGHT if absf(z_back.y) > 0.9 else Vector3.UP;
	var x_right: Vector3 = y_normal.cross(z_back).normalized();
	y_normal = z_back.cross(x_right).normalized();
	var synodic_frame := Basis(x_right, y_normal, z_back);
	return synodic_frame.transposed();


## Returns [code]true[/code] when automatic runtime time progression is active.
func _should_step_time() -> bool:
	return auto_advance and not Engine.is_editor_hint();


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
		if not orbiter.orbit_visuals_changed.is_connected(_on_rail_orbit_visuals_changed):
			orbiter.orbit_visuals_changed.connect(_on_rail_orbit_visuals_changed);
		_queue_dirty_rail_spline(orbiter);
		invalidate_all_ballistic_trajectories();
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter;
		ballistic._fast_root_sync_ready = false;
		ballistic._direct_system_xf = _last_direct_system_xf_state;
		ballistic.bind_driver_sample_buffers(_sampled_time_cell, _sampled_pos_holder);
		if ballistic._trajectory_tape.is_valid and not ballistic._trajectory_dirty:
			ballistic._obeying_shared_tape_clock = true;
		_queue_non_root_sync_ballistic(ballistic);
		if not already_tracked:
			var seq_idx: int = _ballistic_sequence.size();
			var cohort_idx: int = ((seq_idx * 29) + (seq_idx >> 6)) & (TAPE_COHORT_COUNT - 1);
			ballistic._ballistic_seq_index = seq_idx;
			ballistic._tape_cohort = cohort_idx;
			_ballistic_sequence.append(ballistic);
			_ballistic_tapes.append(ballistic._trajectory_tape);
			_tape_cohorts[cohort_idx].append(ballistic);
			_cohort_min_refill[cohort_idx] = -INF;
			_invalidate_solver_tape_atlas_safely();
			if ballistic.runtime_forecast_enabled:
				_forecast_enabled_ballistics.append(ballistic);
			if not ballistic.scheduled_burns.is_empty():
				_burn_scheduled_ballistics.append(ballistic);
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
		if not ballistic.ballistic_trajectory_dirtied.is_connected(
				_on_ballistic_trajectory_dirtied
		):
			ballistic.ballistic_trajectory_dirtied.connect(_on_ballistic_trajectory_dirtied);
		if not ballistic.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
			ballistic.orbit_visuals_changed.connect(_on_descendant_orbit_visuals_changed);
		if ballistic.is_trajectory_dirty():
			_queue_dirty_trajectory_ballistic(ballistic);
	else:
		if not child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
			child.orbit_visuals_changed.connect(_on_descendant_orbit_visuals_changed);

	_units_and_mu_dirty = true;
	_has_synced_playing = false;

	if is_inside_tree() and not _updating_orbits:
		update_orbits();


## Removes an exiting descendant simulation [param child] from the lookup tables
## and flat traversal sequences.
func _on_subtree_element_removed(_parent: JrOrreryElement, child: JrOrreryElement) -> void:
	if child == null:
		return;

	if child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
		child.orbit_visuals_changed.disconnect(_on_descendant_orbit_visuals_changed);
	if child.orbit_visuals_changed.is_connected(_on_rail_orbit_visuals_changed):
		child.orbit_visuals_changed.disconnect(_on_rail_orbit_visuals_changed);

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
		invalidate_all_ballistic_trajectories();
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter;
		var old_cohort: int = ballistic._tape_cohort;
		if ballistic.is_tape_position_pending():
			ballistic.ensure_tape_position();
		if ballistic.is_tape_velocity_pending():
			ballistic.ensure_tape_velocity();
		ballistic.set_driver_sampled_time_cell(PackedFloat64Array());
		ballistic._fast_root_sync_ready = false;
		ballistic._in_non_root_sync_list = false;
		ballistic._in_dirty_trajectory_queue = false;
		ballistic._ballistic_seq_index = -1;
		_invalidate_solver_tape_atlas_safely();
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
		if ballistic.ballistic_trajectory_dirtied.is_connected(_on_ballistic_trajectory_dirtied):
			ballistic.ballistic_trajectory_dirtied.disconnect(_on_ballistic_trajectory_dirtied);
		_remove_ballistic_from_sparse_list(_non_root_sync_ballistics, ballistic);
		_remove_ballistic_from_sparse_list(_forecast_enabled_ballistics, ballistic);
		_remove_ballistic_from_sparse_list(_dirty_trajectory_ballistics, ballistic);
		_remove_ballistic_from_sparse_list(_burn_scheduled_ballistics, ballistic);
		if (
				not _ballistic_sequence.is_empty()
				and _ballistic_sequence[_ballistic_sequence.size() - 1] == ballistic
		):
			_ballistic_sequence.pop_back();
			if not _ballistic_tapes.is_empty():
				_ballistic_tapes.pop_back();
			var c_list: Array = _tape_cohorts[old_cohort];
			if not c_list.is_empty() and c_list[c_list.size() - 1] == ballistic:
				c_list.pop_back();
			else:
				c_list.erase(ballistic);
		else:
			_ballistic_sequence.erase(ballistic);
			if _batch_update_depth == 0:
				_rebuild_ballistic_tape_cohorts();
		if _batch_update_depth == 0:
			_recompute_tape_schedule_bounds();
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


## Queues [param ballistic] for trajectory-tape recomputation when [signal JrBallisticOrbiter.ballistic_trajectory_dirtied] fires.
func _on_ballistic_trajectory_dirtied(ballistic: JrBallisticOrbiter) -> void:
	if not ballistic.scheduled_burns.is_empty():
		if not _burn_scheduled_ballistics.has(ballistic):
			_burn_scheduled_ballistics.append(ballistic);
	else:
		_remove_ballistic_from_sparse_list(_burn_scheduled_ballistics, ballistic);
	_queue_dirty_trajectory_ballistic(ballistic);
	if (
			not _should_step_time()
			and enable_ballistic_forecasts
			and ballistic.runtime_forecast_enabled
			and not _updating_orbits
	):
		_schedule_deferred_orbit_update();


## Emits scheduled burn start, stage jettison, and completion signals across `(t_prev, t_curr]`.
func _emit_scheduled_burn_events(t_prev: float, t_curr: float) -> void:
	for i: int in range(_burn_scheduled_ballistics.size()):
		var b: JrBallisticOrbiter = _burn_scheduled_ballistics[i];
		if is_instance_valid(b) and b._owning_driver == self:
			b.emit_scheduled_burn_events_between(t_prev, t_curr);


## Appends [param ballistic] to [member _dirty_trajectory_ballistics] if not already queued.
func _queue_dirty_trajectory_ballistic(ballistic: JrBallisticOrbiter) -> void:
	_earliest_tape_refill_time = -INF;
	if not ballistic._in_dirty_trajectory_queue:
		ballistic._in_dirty_trajectory_queue = true;
		_dirty_trajectory_ballistics.append(ballistic);


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
		ballistic.invalidate_trajectory();
		_queue_dirty_trajectory_ballistic(ballistic);
	else:
		_remove_ballistic_from_sparse_list(_forecast_enabled_ballistics, ballistic);
		if ballistic.has_runtime_forecast_points():
			ballistic.set_runtime_forecast_points(
					PackedVector3Array(),
					PackedVector3Array(),
					PackedVector3Array()
			);


## Removes [param target] from [param list] with a fast `pop_back()` path during LIFO teardown.
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


## Rebuilds [member _ballistic_tapes], sequence indices, and 32-bucket [member _tape_cohorts]
## after a non-LIFO subtree removal.
func _rebuild_ballistic_tape_cohorts() -> void:
	var count: int = _ballistic_sequence.size();
	_ballistic_tapes.resize(count);
	for c: int in range(TAPE_COHORT_COUNT):
		_tape_cohorts[c].clear();
	if count <= 0:
		return;
	var use_contiguous_slices: bool = count >= 1024;
	for seq_idx: int in range(count):
		var b: JrBallisticOrbiter = _ballistic_sequence[seq_idx];
		var cohort_idx: int;
		if use_contiguous_slices:
			@warning_ignore("integer_division")
			cohort_idx = mini(
					TAPE_COHORT_COUNT - 1,
					(seq_idx * TAPE_COHORT_COUNT) / count
			);
		else:
			cohort_idx = ((seq_idx * 29) + (seq_idx >> 6)) & (TAPE_COHORT_COUNT - 1);
		b._ballistic_seq_index = seq_idx;
		b._tape_cohort = cohort_idx;
		_ballistic_tapes[seq_idx] = b._trajectory_tape;
		_tape_cohorts[cohort_idx].append(b);


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


## Invalidates all ballistic trajectories and re-evaluates system state when a gravitational
## [JrRailOrbiter] mutates mass or orbital parameters.
func _on_rail_orbit_visuals_changed() -> void:
	_units_and_mu_dirty = true;
	invalidate_all_ballistic_trajectories();
	if is_inside_tree() and not _updating_orbits:
		update_orbits();


## Re-evaluates system state when a non-rail descendant element's parameters change in
## the editor or at runtime.
func _on_descendant_orbit_visuals_changed() -> void:
	_units_and_mu_dirty = true;
	if is_inside_tree() and not _updating_orbits:
		update_orbits();
