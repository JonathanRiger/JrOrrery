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
			distance_unit = value
			_propagate_units_and_parent_mu()
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree():
				update_orbits()
			orbit_visuals_changed.emit()

## Physical time unit represented by [code]1.0[/code] unit of [member time] across this driver's
## hierarchy.
@export var time_unit := JrOrreryUnits.TimeUnit.SECONDS:
	set(value):
		if time_unit != value:
			time_unit = value
			_propagate_units_and_parent_mu()
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree():
				update_orbits()
			orbit_visuals_changed.emit()

@export_group("Gravity")
## Physical mass of the root focus at the origin of this [JrOrbitDriver] (for example,
## a central star, or [code]0.0[/code] for a massless binary barycenter), expressed in
## [member mass_unit].
@export_range(0.0, 1_000_000.0, 0.001, "or_greater")
var mass: float = 0.0:
	set(value):
		var clamped: float = maxf(0.0, value)
		if mass != clamped:
			mass = clamped
			_propagate_units_and_parent_mu()
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree() and Engine.is_editor_hint():
				update_orbits()
			orbit_visuals_changed.emit()

## Unit in which [member mass] is specified. Combined with [member distance_unit] and
## [member time_unit] to compute the root gravitational parameter [code]mu = G * M[/code].
@export var mass_unit := JrOrreryUnits.MassUnit.DRIVER_GM:
	set(value):
		if mass_unit != value:
			mass_unit = value
			_propagate_units_and_parent_mu()
			_invalidate_all_ballistic_forecasts()
			if is_inside_tree() and Engine.is_editor_hint():
				update_orbits()
			orbit_visuals_changed.emit()

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
var _ballistic_sequence: Array[JrBallisticOrbiter] = []

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
	_recompute_gravity_strength_from_mass()
	var own_mu: float = _gravity_strength
	if own_mu > 0.0:
		return own_mu
	var binary_sum: float = 0.0
	for child_node: Node in get_children():
		if child_node is JrRailOrbiter:
			var child_orb := child_node as JrRailOrbiter
			if child_orb.binary_partner != null and child_orb.binary_partner.get_parent() == self:
				binary_sum += child_orb.get_gravity_strength()
	return binary_sum


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
	_batch_update_depth = maxi(0, _batch_update_depth - 1)
	if _batch_update_depth == 0 and is_inside_tree():
		_propagate_units_and_parent_mu()
		update_orbits()


## Reparents all descendant [JrBallisticOrbiter] nodes in this hierarchy to be direct
## children of this [JrOrbitDriver] without resetting their initialized orbital state
## or losing their original spawn focus reference.
func collect_ballistic_nodes() -> void:
	if _orbit_sequence.is_empty() and _ballistic_sequence.is_empty() and get_child_count() > 0:
		_initialize_driver_hierarchy()
	begin_batch_updates()
	_propagate_units_and_parent_mu()
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
		return

	if not child_element_added.is_connected(_on_subtree_element_added) or (
			_orbit_sequence.is_empty() and _ballistic_sequence.is_empty() and get_child_count() > 0
	):
		_reference_plane.configure(
				reference_rotation,
				euler_order,
				positive_is_clockwise,
				lock_to_ecliptic
		)
		if not child_element_added.is_connected(_on_subtree_element_added):
			child_element_added.connect(_on_subtree_element_added)
		if not child_element_removed.is_connected(_on_subtree_element_removed):
			child_element_removed.connect(_on_subtree_element_removed)
		refresh_subtree_tracking()

	_updating_orbits = true
	_propagate_units_and_parent_mu()
	var playing: bool = _should_step_time()
	set_simulation_playing(playing)

	_relative_velocity = Vector3.ZERO
	_global_velocity = Vector3.ZERO

	if Engine.is_editor_hint():
		_evaluate_editor_system_state(playing)
	else:
		_evaluate_runtime_system_state(playing)

	var persp_basis: Basis = _compute_perspective_basis()
	var driver_origin: Vector3 = persp_basis * _compute_lca_position_difference(
			self,
			perspective_node
	)
	var driver_system_transform := Transform3D(persp_basis, driver_origin)
	_local_orbital_transform = driver_system_transform
	_driver_relative_transform = Transform3D.IDENTITY
	set_system_transform(driver_system_transform)

	for orbiter: JrRailOrbiter in _orbit_sequence:
		var rel_pos: Vector3 = _compute_lca_position_difference(orbiter, perspective_node)
		var sys_origin: Vector3 = persp_basis * rel_pos
		var sys_basis: Basis = persp_basis * orbiter.get_driver_relative_transform().basis
		orbiter.set_system_transform(Transform3D(sys_basis, sys_origin))

	if not Engine.is_editor_hint():
		_update_runtime_ballistic_forecasts(persp_basis)

	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var rel_pos: Vector3 = _compute_lca_position_difference(ballistic, perspective_node)
		var sys_origin: Vector3 = persp_basis * rel_pos
		var sys_basis: Basis = persp_basis * ballistic.get_driver_relative_transform().basis
		ballistic.set_system_transform(Transform3D(sys_basis, sys_origin))

	if Engine.is_editor_hint():
		_update_editor_ghosts()

	_updating_orbits = false


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


## Connects subtree signals, configures [member _reference_plane], scans all
## descendant elements, and evaluates initial orbital and ballistic transforms.
func _initialize_driver_hierarchy() -> void:
	_reference_plane.configure(
			reference_rotation,
			euler_order,
			positive_is_clockwise,
			lock_to_ecliptic
	)
	if not child_element_added.is_connected(_on_subtree_element_added):
		child_element_added.connect(_on_subtree_element_added)
	if not child_element_removed.is_connected(_on_subtree_element_removed):
		child_element_removed.connect(_on_subtree_element_removed)
	refresh_subtree_tracking()
	_propagate_units_and_parent_mu()
	update_orbits()


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
	set_driver_units(distance_unit, time_unit)
	for orbiter: JrRailOrbiter in _orbit_sequence:
		orbiter.set_driver_units(distance_unit, time_unit)
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_elem: JrOrreryElement = _children.get(orbiter, self)
		var eff_mu: float = parent_elem.get_effective_parent_mu() if parent_elem != null else 0.0
		orbiter.set_parent_effective_mu(eff_mu)
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_driver_units(distance_unit, time_unit)
		var spawn_parent: JrOrreryElement = _resolve_ballistic_spawn_parent(ballistic)
		var eff_mu: float = spawn_parent.get_effective_parent_mu() if spawn_parent != null else 0.0
		ballistic.set_parent_effective_mu(eff_mu)


## Marks all tracked [JrBallisticOrbiter] forecasts as dirty when system gravity or units change.
func _invalidate_all_ballistic_forecasts() -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.mark_forecast_dirty()


## Evaluates epoch Keplerian states and forward-integrated ballistic trajectory
## previews when running inside the Godot Editor.
func _evaluate_editor_system_state(playing: bool) -> void:
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_xf: Transform3D = _get_parent_driver_transform_for(orbiter)
		var parent_vel: Vector3 = _get_parent_global_velocity_for(orbiter)
		orbiter.set_simulation_playing(playing)
		orbiter.evaluate_orbital_state(0.0, _reference_plane, parent_xf, parent_vel)

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
	var dt: float = time - _last_stepped_time if _runtime_initialized else 0.0
	var prev_local_positions: Dictionary[JrRailOrbiter, Vector3] = {}

	if not _runtime_initialized:
		_evaluate_all_orbiters_at(time, playing)
		_initialize_pending_ballistics(playing)
		_sync_all_ballistics_transforms(playing)
		_runtime_initialized = true
		_last_stepped_time = time
		return

	_initialize_pending_ballistics(playing)
	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		prev_local_positions = _capture_orbiter_local_positions()

	_evaluate_all_orbiters_at(time, playing)

	if not _ballistic_sequence.is_empty() and not is_zero_approx(dt):
		_update_ballistic_anchors_before_step()
		var well_elements: Array[JrOrreryElement] = _build_runtime_well_elements()
		var packed_wells: PackedFloat32Array = _pack_runtime_gravity_wells(
				well_elements,
				prev_local_positions
		)
		_ballistic_solver.configure_step_context(time, tidal_cull_threshold, well_elements)
		_ballistic_solver.step_bodies(_ballistic_sequence, packed_wells, dt, ballistic_substeps)
		_update_ballistic_anchors_after_step()

	_last_stepped_time = time
	_sync_all_ballistics_transforms(playing)


## Evaluates all tracked [JrRailOrbiter] nodes in [member _orbit_sequence] at [param eval_time].
func _evaluate_all_orbiters_at(eval_time: float, playing: bool) -> void:
	for orbiter: JrRailOrbiter in _orbit_sequence:
		var parent_xf: Transform3D = _get_parent_driver_transform_for(orbiter)
		var parent_vel: Vector3 = _get_parent_global_velocity_for(orbiter)
		orbiter.set_simulation_playing(playing)
		orbiter.evaluate_orbital_state(eval_time, _reference_plane, parent_xf, parent_vel)


## Initializes any [JrBallisticOrbiter] nodes that were newly added or requested a state reset.
func _initialize_pending_ballistics(playing: bool) -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_simulation_playing(playing)
		if not ballistic.is_runtime_state_ready():
			_initialize_single_ballistic(ballistic, playing)


## Initializes a single [param ballistic] body relative to its spawn parent and initial anchor.
func _initialize_single_ballistic(ballistic: JrBallisticOrbiter, playing: bool) -> void:
	ballistic.set_simulation_playing(playing)
	var spawn_parent: JrOrreryElement = _resolve_ballistic_spawn_parent(ballistic)
	var spawn_xf: Transform3D = _get_element_driver_transform(spawn_parent)
	var spawn_vel: Vector3 = _get_element_global_velocity(spawn_parent)
	var init_anchor: JrOrreryElement = spawn_parent
	if ballistic.pinned_anchor != null and _is_tracked_or_self(ballistic.pinned_anchor):
		init_anchor = ballistic.pinned_anchor
	ballistic.initialize_from_parent(spawn_xf, spawn_vel, _reference_plane, init_anchor)
	if init_anchor != spawn_parent:
		var spawn_to_anchor_pos: Vector3 = _compute_lca_position_difference(
				init_anchor,
				spawn_parent
		)
		var spawn_to_anchor_vel: Vector3 = _compute_lca_velocity_difference(
				init_anchor,
				spawn_parent
		)
		ballistic.apply_integrated_anchor_state(
				ballistic.get_anchor_relative_position() - spawn_to_anchor_pos,
				ballistic.get_anchor_relative_velocity() - spawn_to_anchor_vel
		)


## Synchronizes transforms on all tracked [JrBallisticOrbiter] nodes using LCA offsets.
func _sync_all_ballistics_transforms(playing: bool) -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_simulation_playing(playing)
		_sync_single_ballistic_transforms(ballistic)


## Synchronizes a single [param ballistic] node's transforms with its anchor and scene parent.
func _sync_single_ballistic_transforms(ballistic: JrBallisticOrbiter) -> void:
	var scene_parent: JrOrreryElement = _element_parents.get(ballistic, self)
	var parent_xf: Transform3D = _get_element_driver_transform(scene_parent)
	var parent_vel: Vector3 = _get_element_global_velocity(scene_parent)
	var anchor: JrOrreryElement = ballistic.get_anchor_element()
	if anchor == null or not _is_tracked_or_self(anchor):
		anchor = scene_parent
	var anchor_to_parent_pos: Vector3 = _compute_lca_position_difference(anchor, scene_parent)
	var anchor_to_parent_vel: Vector3 = _compute_lca_velocity_difference(anchor, scene_parent)
	ballistic.sync_transforms_with_anchor_and_parent(
			parent_xf,
			parent_vel,
			anchor_to_parent_pos,
			anchor_to_parent_vel
	)


## Records the current parent-local orbital offset of each [JrRailOrbiter] prior to stepping.
func _capture_orbiter_local_positions() -> Dictionary[JrRailOrbiter, Vector3]:
	var positions: Dictionary[JrRailOrbiter, Vector3] = {}
	for orbiter: JrRailOrbiter in _orbit_sequence:
		if orbiter.is_orbit_valid():
			positions[orbiter] = orbiter.get_focus_relative_position()
	return positions


## Builds the ordered list of gravity-well [JrOrreryElement] nodes (index 0 = driver,
## followed by all valid [JrRailOrbiter] nodes in topological parent-before-child order).
func _build_runtime_well_elements() -> Array[JrOrreryElement]:
	var wells: Array[JrOrreryElement] = [self]
	for orbiter: JrRailOrbiter in _orbit_sequence:
		if orbiter.is_orbit_valid():
			wells.append(orbiter)
	return wells


## Packs start-of-step and end-of-step parent-local positions, gravitational parameters,
## and parent well indices for all elements in [param well_elements].
func _pack_runtime_gravity_wells(
		well_elements: Array[JrOrreryElement],
		prev_local_positions: Dictionary[JrRailOrbiter, Vector3]
) -> PackedFloat32Array:
	var packed: PackedFloat32Array = []
	var well_index_map: Dictionary[JrOrreryElement, int] = {}
	for i: int in range(well_elements.size()):
		well_index_map[well_elements[i]] = i

	_append_well_floats(packed, Vector3.ZERO, Vector3.ZERO, Vector2(_gravity_strength, -1.0))

	for i: int in range(1, well_elements.size()):
		var orbiter := well_elements[i] as JrRailOrbiter
		var mu: float = orbiter.get_gravity_strength()
		var end_local: Vector3 = orbiter.get_focus_relative_position()
		var start_local: Vector3 = prev_local_positions.get(orbiter, end_local)
		var parent_elem: JrOrreryElement = _children.get(orbiter, self)
		var parent_idx: int = well_index_map.get(parent_elem, 0)
		_append_well_floats(packed, start_local, end_local, Vector2(mu, float(parent_idx)))

	return packed


## Appends a single 8-float tree-relative [code]GravityWell[/code] record into [param packed].
func _append_well_floats(
		packed: PackedFloat32Array,
		start_local: Vector3,
		end_local: Vector3,
		mu_and_parent: Vector2
) -> void:
	packed.append(start_local.x)
	packed.append(start_local.y)
	packed.append(start_local.z)
	packed.append(mu_and_parent.x)
	packed.append(end_local.x)
	packed.append(end_local.y)
	packed.append(end_local.z)
	packed.append(mu_and_parent.y)


## Synchronizes any pinned anchor assignments before stepping [member _ballistic_sequence].
func _update_ballistic_anchors_before_step() -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		var curr_anchor: JrOrreryElement = ballistic.get_anchor_element()
		if curr_anchor == null or not _is_tracked_or_self(curr_anchor):
			curr_anchor = _resolve_ballistic_spawn_parent(ballistic)
			var spawn_p: JrOrreryElement = _element_parents.get(ballistic, self)
			var diff_pos: Vector3 = _compute_lca_position_difference(curr_anchor, spawn_p)
			var diff_vel: Vector3 = _compute_lca_velocity_difference(curr_anchor, spawn_p)
			ballistic.switch_anchor_element(curr_anchor, diff_pos, diff_vel)

		if ballistic.pinned_anchor != null and _is_tracked_or_self(ballistic.pinned_anchor):
			if ballistic.get_anchor_element() != ballistic.pinned_anchor:
				var old_anchor: JrOrreryElement = ballistic.get_anchor_element()
				var old_to_new_pos: Vector3 = _compute_lca_position_difference(
						ballistic.pinned_anchor,
						old_anchor
				)
				var old_to_new_vel: Vector3 = _compute_lca_velocity_difference(
						ballistic.pinned_anchor,
						old_anchor
				)
				ballistic.switch_anchor_element(
						ballistic.pinned_anchor,
						old_to_new_pos,
						old_to_new_vel
				)


## Evaluates Hill-sphere hysteresis (`2.4x` entry / `2.6x` exit) after stepping
## for all unpinned [JrBallisticOrbiter] nodes.
func _update_ballistic_anchors_after_step() -> void:
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if ballistic.pinned_anchor != null:
			continue
		_evaluate_hill_sphere_handoff_for(ballistic)


## Evaluates whether [param ballistic] should enter a direct child well of its current
## anchor (`< 2.4 * r_Hill`) or exit to its current anchor's parent well (`> 2.6 * r_Hill`).
func _evaluate_hill_sphere_handoff_for(ballistic: JrBallisticOrbiter) -> void:
	var curr_anchor: JrOrreryElement = ballistic.get_anchor_element()
	if curr_anchor == null or not _is_tracked_or_self(curr_anchor):
		return

	var r_rel: Vector3 = ballistic.get_anchor_relative_position()
	var mu_curr_effective: float = curr_anchor.get_effective_parent_mu()

	if mu_curr_effective > 0.0:
		var best_child: JrRailOrbiter = null
		var best_dist: float = INF
		for orbiter: JrRailOrbiter in _orbit_sequence:
			if _children.get(orbiter, self) != curr_anchor:
				continue
			var mu_child: float = orbiter.get_gravity_strength()
			if mu_child <= 0.0 or not orbiter.is_orbit_valid():
				continue
			var child_offset: Vector3 = orbiter.get_focus_relative_position()
			var d_len: float = child_offset.length()
			if d_len <= 1e-6:
				continue
			var r_hill: float = d_len * pow(mu_child / (3.0 * mu_curr_effective), 1.0 / 3.0)
			var dist_to_child: float = (r_rel - child_offset).length()
			if dist_to_child < HILL_ENTRY_MULTIPLIER * r_hill and dist_to_child < best_dist:
				best_dist = dist_to_child
				best_child = orbiter

		if best_child != null:
			var old_to_new_pos: Vector3 = best_child.get_focus_relative_position()
			var old_to_new_vel: Vector3 = best_child.get_relative_velocity()
			ballistic.switch_anchor_element(best_child, old_to_new_pos, old_to_new_vel)
			return

	if curr_anchor is JrRailOrbiter:
		var anchor_orb := curr_anchor as JrRailOrbiter
		var parent_anchor: JrOrreryElement = _children.get(anchor_orb, self)
		if parent_anchor == null:
			return
		var mu_parent: float = parent_anchor.get_effective_parent_mu()
		var mu_curr: float = anchor_orb.get_gravity_strength()
		if mu_parent <= 0.0 or mu_curr <= 0.0:
			return
		var anchor_offset: Vector3 = anchor_orb.get_focus_relative_position()
		var d_len: float = anchor_offset.length()
		var r_hill: float = d_len * pow(mu_curr / (3.0 * mu_parent), 1.0 / 3.0)
		if r_rel.length() > HILL_EXIT_MULTIPLIER * r_hill:
			var old_to_new_pos: Vector3 = -anchor_offset
			var old_to_new_vel: Vector3 = -anchor_orb.get_relative_velocity()
			ballistic.switch_anchor_element(parent_anchor, old_to_new_pos, old_to_new_vel)


## Computes the exact position difference [code]r_a - r_b[/code] between [param elem_a]
## and [param elem_b] by summing tree-relative offsets only up to their Lowest Common Ancestor,
## avoiding cancellation of shared ancestor distances (such as 1 AU).
func _compute_lca_position_difference(elem_a: JrOrreryElement, elem_b: JrOrreryElement) -> Vector3:
	var target_a: JrOrreryElement = (
			elem_a
			if (elem_a != null and _is_tracked_or_self(elem_a))
			else self
	)
	var target_b: JrOrreryElement = (
			elem_b
			if (elem_b != null and _is_tracked_or_self(elem_b))
			else self
	)
	if target_a == target_b:
		return Vector3.ZERO

	var b_ancestors: Dictionary[JrOrreryElement, Vector3] = {}
	var curr_b: JrOrreryElement = target_b
	var accum_b := Vector3.ZERO
	while curr_b != null:
		b_ancestors[curr_b] = accum_b
		if curr_b == self:
			break
		accum_b += _get_element_kinematic_local_offset(curr_b)
		curr_b = _get_element_kinematic_parent(curr_b)

	var curr_a: JrOrreryElement = target_a
	var accum_a := Vector3.ZERO
	while curr_a != null:
		if b_ancestors.has(curr_a):
			return accum_a - b_ancestors[curr_a]
		if curr_a == self:
			break
		accum_a += _get_element_kinematic_local_offset(curr_a)
		curr_a = _get_element_kinematic_parent(curr_a)

	return accum_a - accum_b


## Computes the exact velocity difference [code]v_a - v_b[/code] between [param elem_a]
## and [param elem_b] by summing tree-relative velocities only up to their Lowest Common Ancestor.
func _compute_lca_velocity_difference(elem_a: JrOrreryElement, elem_b: JrOrreryElement) -> Vector3:
	var target_a: JrOrreryElement = (
			elem_a
			if (elem_a != null and _is_tracked_or_self(elem_a))
			else self
	)
	var target_b: JrOrreryElement = (
			elem_b
			if (elem_b != null and _is_tracked_or_self(elem_b))
			else self
	)
	if target_a == target_b:
		return Vector3.ZERO

	var b_ancestors: Dictionary[JrOrreryElement, Vector3] = {}
	var curr_b: JrOrreryElement = target_b
	var accum_b := Vector3.ZERO
	while curr_b != null:
		b_ancestors[curr_b] = accum_b
		if curr_b == self:
			break
		accum_b += _get_element_kinematic_local_velocity(curr_b)
		curr_b = _get_element_kinematic_parent(curr_b)

	var curr_a: JrOrreryElement = target_a
	var accum_a := Vector3.ZERO
	while curr_a != null:
		if b_ancestors.has(curr_a):
			return accum_a - b_ancestors[curr_a]
		if curr_a == self:
			break
		accum_a += _get_element_kinematic_local_velocity(curr_a)
		curr_a = _get_element_kinematic_parent(curr_a)

	return accum_a - accum_b


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
	)
	var target_b: JrOrreryElement = (
			well_b
			if (well_b != null and _is_tracked_or_self(well_b))
			else self
	)
	if target_a == target_b:
		return Vector3.ZERO

	var b_ancestors: Dictionary[JrOrreryElement, Vector3] = {}
	var curr_b: JrOrreryElement = target_b
	var accum_b := Vector3.ZERO
	while curr_b != null:
		b_ancestors[curr_b] = accum_b
		if curr_b == self:
			break
		if curr_b is JrRailOrbiter:
			var orb_b := curr_b as JrRailOrbiter
			if orb_b.is_orbit_valid():
				accum_b += orb_b.get_relative_position(eval_time, _reference_plane)
			curr_b = _children.get(orb_b, self)
		else:
			curr_b = _element_parents.get(curr_b, self)

	var curr_a: JrOrreryElement = target_a
	var accum_a := Vector3.ZERO
	while curr_a != null:
		if b_ancestors.has(curr_a):
			return accum_a - b_ancestors[curr_a]
		if curr_a == self:
			break
		if curr_a is JrRailOrbiter:
			var orb_a := curr_a as JrRailOrbiter
			if orb_a.is_orbit_valid():
				accum_a += orb_a.get_relative_position(eval_time, _reference_plane)
			curr_a = _children.get(orb_a, self)
		else:
			curr_a = _element_parents.get(curr_a, self)

	return accum_a - accum_b


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
		return

	if not enable_ballistic_forecasts:
		for ballistic: JrBallisticOrbiter in _ballistic_sequence:
			ballistic.set_runtime_forecast_points([], [], [])
		return

	var targets: Array[JrBallisticOrbiter] = _collect_forecast_targets()
	if targets.is_empty():
		for ballistic: JrBallisticOrbiter in _ballistic_sequence:
			ballistic.set_runtime_forecast_points([], [], [])
		return

	var anchor_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			_resolve_anchor_relative_forecasts(targets)
	)
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if not ballistic.runtime_forecast_enabled:
			ballistic.set_runtime_forecast_points([], [], [])
			continue
		var pts4: PackedVector4Array = anchor_map.get(ballistic, PackedVector4Array())
		_project_forecast_to_perspective(ballistic, pts4, system_basis)


## Collects all [JrBallisticOrbiter] nodes requiring forecast evaluation.
func _collect_forecast_targets() -> Array[JrBallisticOrbiter]:
	var targets: Array[JrBallisticOrbiter] = []
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		if ballistic.runtime_forecast_enabled:
			targets.append(ballistic)
	return targets


## Re-integrates anchor-relative forecasts on the GPU/CPU when any target is dirty, or
## reuses cached anchor-relative forecasts when only perspective settings changed.
func _resolve_anchor_relative_forecasts(
		targets: Array[JrBallisticOrbiter]
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var any_dirty: bool = _last_forecast_well_list.is_empty()
	if not any_dirty:
		for target: JrBallisticOrbiter in targets:
			if target.is_forecast_dirty() or not _last_forecast_anchor_points.has(target):
				any_dirty = true
				break

	if any_dirty:
		var cfg: JrBallisticForecastConfig = _build_forecast_config(targets, time)
		_last_forecast_well_list = _ballistic_solver.get_forecast_well_elements(cfg)
		_last_forecast_anchor_points = _ballistic_solver.compute_forecasts_anchor_relative(cfg)

	return _last_forecast_anchor_points


## Projects [param pts4] at each forecast step into the reference frame of the
## earliest (lowest) common ancestor of the step's SOI anchor and [member perspective_node].
func _project_forecast_to_perspective(
		ballistic: JrBallisticOrbiter,
		pts4: PackedVector4Array,
		system_basis: Basis
) -> void:
	var count: int = pts4.size()
	if count == 0:
		ballistic.set_runtime_forecast_points([], [], [])
		return

	var step_dt: float = ballistic.preview_duration / float(maxi(1, count - 1))
	var selected_well: JrOrreryElement = _get_perspective_gravity_well()
	var is_heliocentric: bool = (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	)
	var identity_basis: bool = system_basis.is_equal_approx(Basis.IDENTITY)
	var is_synodic: bool = (
			not is_heliocentric
			and perspective_rotation_mode == PerspectiveRotationMode.SYNODIC_ORBITAL
	)
	var synodic_now_basis: Basis = system_basis.transposed() if is_synodic else Basis.IDENTITY

	var inertial_pts: PackedVector3Array = []
	var unrotated_pts: PackedVector3Array = []
	var system_pts: PackedVector3Array = []
	var ref_ids: PackedInt64Array = []
	inertial_pts.resize(count)
	unrotated_pts.resize(count)
	system_pts.resize(count)
	ref_ids.resize(count)

	var last_common_well: JrOrreryElement = null
	var last_common_from_persp_now := Vector3.ZERO
	var last_common_synodic: bool = false
	var last_common_id: int = 0

	for k: int in range(count):
		var pt4: Vector4 = pts4[k]
		var anchor_elem: JrOrreryElement = _get_forecast_well_element(int(roundf(pt4.w)))
		var eval_t: float = time + float(k) * step_dt
		var r_rel := Vector3(pt4.x, pt4.y, pt4.z)
		var anchor_origin: Vector3 = evaluate_element_driver_origin_at(anchor_elem, eval_t)
		inertial_pts[k] = anchor_origin + r_rel

		var common_well: JrOrreryElement = _find_common_ancestor_well(anchor_elem, selected_well)
		if common_well != last_common_well:
			last_common_well = common_well
			last_common_from_persp_now = _compute_lca_position_difference(
					common_well,
					perspective_node
			)
			last_common_synodic = is_synodic and _should_apply_synodic_rotation_to_well(
					common_well
			)
			last_common_id = int(common_well.get_instance_id())
		ref_ids[k] = last_common_id

		var offset_from_common_at_eval: Vector3 = (
				_compute_lca_well_position_difference_at(anchor_elem, common_well, eval_t)
				+ r_rel
		)

		if last_common_synodic:
			var synodic_step_inv: Basis = _compute_synodic_perspective_basis_at(eval_t)
			var synodic_pt: Vector3 = (
					(system_basis * last_common_from_persp_now)
					+ (synodic_step_inv * offset_from_common_at_eval)
			)
			system_pts[k] = synodic_pt
			unrotated_pts[k] = synodic_now_basis * synodic_pt
		else:
			var unrot_pt: Vector3 = last_common_from_persp_now + offset_from_common_at_eval
			unrotated_pts[k] = unrot_pt
			system_pts[k] = unrot_pt if identity_basis else (system_basis * unrot_pt)

	ballistic.set_runtime_forecast_points(inertial_pts, unrotated_pts, system_pts, ref_ids)


## Returns the gravity-well element ([JrOrbitDriver] or [JrRailOrbiter]) associated
## with [member perspective_node].
func _get_perspective_gravity_well() -> JrOrreryElement:
	if (
			perspective_node == null
			or perspective_node == self
			or not _element_parents.has(perspective_node)
	):
		return self
	if perspective_node is JrRailOrbiter:
		return perspective_node
	if perspective_node is JrBallisticOrbiter:
		var b_node := perspective_node as JrBallisticOrbiter
		var b_anchor: JrOrreryElement = b_node.get_anchor_element()
		if b_anchor != null and _is_tracked_or_self(b_anchor):
			return b_anchor
		return _resolve_ballistic_spawn_parent(b_node)
	return self


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
	)
	var target_b: JrOrreryElement = (
			well_b
			if (well_b != null and _is_tracked_or_self(well_b))
			else self
	)
	if target_a == self or target_b == self:
		return self
	if target_a == target_b:
		return target_a

	var b_ancestors: Array[JrOrreryElement] = []
	var curr_b: JrOrreryElement = target_b
	while curr_b != null:
		b_ancestors.append(curr_b)
		if curr_b == self:
			break
		curr_b = _get_element_kinematic_parent(curr_b)

	var curr_a: JrOrreryElement = target_a
	while curr_a != null:
		if b_ancestors.has(curr_a):
			return curr_a
		if curr_a == self:
			break
		curr_a = _get_element_kinematic_parent(curr_a)

	return self


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
	return element == self or _element_parents.has(element)


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
	var playing: bool = _should_step_time()
	set_simulation_playing(playing)
	for orbiter: JrRailOrbiter in _orbit_sequence:
		orbiter.set_simulation_playing(playing)
	for ballistic: JrBallisticOrbiter in _ballistic_sequence:
		ballistic.set_simulation_playing(playing)


## Registers a newly added descendant simulation [param child] ([JrRailOrbiter] or
## [JrBallisticOrbiter])
## and its [param parent] into the appropriate flat traversal sequence.
func _on_subtree_element_added(parent: JrOrreryElement, child: JrOrreryElement) -> void:
	if child == null or parent == null or child == self:
		return

	_element_parents[child] = parent
	if child is JrRailOrbiter:
		var orbiter := child as JrRailOrbiter
		_children[orbiter] = parent
		if not _orbit_sequence.has(orbiter):
			_orbit_sequence.append(orbiter)
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter
		if not _ballistic_sequence.has(ballistic):
			_ballistic_sequence.append(ballistic)

	if not child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
		child.orbit_visuals_changed.connect(_on_descendant_orbit_visuals_changed)

	if is_inside_tree() and not _updating_orbits:
		update_orbits()


## Removes an exiting descendant simulation [param child] from the lookup tables
## and flat traversal sequences.
func _on_subtree_element_removed(_parent: JrOrreryElement, child: JrOrreryElement) -> void:
	if child == null:
		return

	if child.orbit_visuals_changed.is_connected(_on_descendant_orbit_visuals_changed):
		child.orbit_visuals_changed.disconnect(_on_descendant_orbit_visuals_changed)

	_element_parents.erase(child)
	if child is JrRailOrbiter:
		var orbiter := child as JrRailOrbiter
		_children.erase(orbiter)
		_orbit_sequence.erase(orbiter)
	elif child is JrBallisticOrbiter:
		var ballistic := child as JrBallisticOrbiter
		_ballistic_sequence.erase(ballistic)
		_last_forecast_anchor_points.erase(ballistic)

	if is_inside_tree() and not _updating_orbits:
		update_orbits()


## Re-evaluates system state when a descendant element's parameters change in
## the editor or at runtime.
func _on_descendant_orbit_visuals_changed() -> void:
	if is_inside_tree() and not _updating_orbits:
		update_orbits()
