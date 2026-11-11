@tool
@abstract
@icon("res://addons/jr_orrery/icons/orrery_element.svg")
class_name JrOrreryElement
extends Node
## Abstract base class for simulation nodes ([JrOrbitDriver], [JrRailOrbiter], and
## [JrBallisticOrbiter]).
##
## Manages direct child simulation nodes, bubbles subtree addition/removal
## signals upward following a strict "Call Down / Signal Up" hierarchy, and
## stores the calculated 3D orbital/ballistic transforms, physical mass and
## gravitational strength, and reactive signals consumed by [JrOrbiter2D] and
## [JrOrbiter3D] visual follower nodes.

## Emitted when a [param child] rail orbiter is added under [param parent] within this
## element's subtree.
signal child_orbiter_added(parent: JrOrreryElement, child: JrRailOrbiter)

## Emitted when a [param child] rail orbiter is removed from [param parent] within this
## element's subtree.
signal child_orbiter_removed(parent: JrOrreryElement, child: JrRailOrbiter)

## Emitted when any simulation [param child] element ([JrRailOrbiter] or [JrBallisticOrbiter])
## is added under [param parent] within this element's subtree.
signal child_element_added(parent: JrOrreryElement, child: JrOrreryElement)

## Emitted when any simulation [param child] element ([JrRailOrbiter] or [JrBallisticOrbiter])
## is removed from [param parent] within this element's subtree.
signal child_element_removed(parent: JrOrreryElement, child: JrOrreryElement)

## Emitted each tick (or when parameters change) after this element's calculated
## orbital transforms have been updated.
signal transform_updated

## Emitted in the Godot Editor when the animated preview ghost timestamp is stepped.
signal editor_ghost_updated(preview_time: float)

## Emitted when orbital parameters, mass/gravity strength, reference direction, or
## simulation playback state change.
signal orbit_visuals_changed

## Tracks direct child [JrRailOrbiter] nodes parented to this element.
var _child_orbiters: Dictionary[JrRailOrbiter, bool] = {}

## Tracks direct child [JrOrreryElement] simulation nodes ([JrRailOrbiter] and
## [JrBallisticOrbiter])
## parented to this element.
var _child_elements: Dictionary[JrOrreryElement, bool] = {}

## Active 3D reference coordinate frame supplied by the owning [JrOrbitDriver].
var _reference_plane := JrReferenceDirection.new()

## Whether the owning [JrOrbitDriver] is actively stepping simulation time.
var _simulation_playing: bool = false

## Physical mass value assigned to this element in [member _mass_unit].
var _mass: float = 0.0

## Physical mass unit used to convert [member _mass] into [member _gravity_strength].
var _mass_unit := JrOrreryUnits.MassUnit.DRIVER_GM

## Active distance unit supplied downward by the owning [JrOrbitDriver].
var _driver_distance_unit := JrOrreryUnits.DistanceUnit.METERS

## Active time unit supplied downward by the owning [JrOrbitDriver].
var _driver_time_unit := JrOrreryUnits.TimeUnit.SECONDS

## Standard gravitational parameter (mu = G * M) in driver units (dist_unit^3 / time_unit^2)
## exerted by this element on [JrBallisticOrbiter] and child [JrRailOrbiter] bodies.
var _gravity_strength: float = 0.0

## Cached position of this element relative to its parent focus in the driver's
## unrotated reference coordinate space.
var _focus_relative_position := Vector3.ZERO

## Cached orbital velocity vector of this element relative to its parent focus in
## the driver's unrotated reference coordinate space.
var _relative_velocity := Vector3.ZERO

## Cached accumulated global orbital velocity vector of this element relative to
## the root [JrOrbitDriver] in the driver's unrotated reference coordinate space.
var _global_velocity := Vector3.ZERO

## Cached orientation [Basis] of this element in the driver's unrotated reference
## coordinate space.
var _driver_relative_basis := Basis.IDENTITY

## Cached driver-relative [Basis] of this element's immediate parent [JrOrreryElement].
var _parent_driver_basis := Basis.IDENTITY

## Calculated local [Transform3D] relative to the immediate parent [JrOrreryElement],
## with [member _parent_driver_basis] canceled so child orbits remain fixed in the
## system's [JrReferenceDirection] frame.
var _local_orbital_transform := Transform3D.IDENTITY

## Calculated heliocentric [Transform3D] relative to the root [JrOrbitDriver]
## prior to perspective shifting.
var _driver_relative_transform := Transform3D.IDENTITY

## Calculated system-space [Transform3D] incorporating the active
## [member JrOrbitDriver.perspective_node] offset and rotation lock.
var _system_transform := Transform3D.IDENTITY

## Transient runtime gravity-well index assigned during [JrOrbitDriver] stepping passes (`-1` when unassigned).
var _runtime_well_index: int = -1;

## Owning [JrOrbitDriver] currently tracking this element (`null` when untracked).
var _owning_driver: JrOrreryElement = null;

## Immediate scene-tree parent [JrOrreryElement] recorded by [member _owning_driver].
var _tracked_scene_parent: JrOrreryElement = null;


## Connects tree-monitoring signals and populates the initial set of direct child elements.
func _ready() -> void:
	refresh_subtree_tracking()


## Ensures child-monitoring signals are connected and rescans direct and descendant
## [JrOrreryElement] nodes (recovering automatically after editor script reloads).
func refresh_subtree_tracking() -> void:
	if not child_entered_tree.is_connected(_on_child_entered_tree):
		child_entered_tree.connect(_on_child_entered_tree)
	if not child_exiting_tree.is_connected(_on_child_exiting_tree):
		child_exiting_tree.connect(_on_child_exiting_tree)
	for child: Node in get_children():
		if child is JrOrreryElement:
			(child as JrOrreryElement).refresh_subtree_tracking()
	_refresh_tracked_nodes()


## Returns an array of all direct child [JrRailOrbiter] nodes attached to this element.
func get_child_orbiters() -> Array[JrRailOrbiter]:
	return _child_orbiters.keys()


## Returns an array of all direct child [JrOrreryElement] simulation nodes attached
## to this element.
func get_child_elements() -> Array[JrOrreryElement]:
	return _child_elements.keys()


## Returns the gravitational parameter (mu = G * M) in driver units exerted by this element.
func get_gravity_strength() -> float:
	return _gravity_strength


## Returns the effective central gravitational parameter (mu) felt by child orbiters
## parented to this element. If this element is a zero-mass barycenter hosting a linked
## binary pair of [JrRailOrbiter] children, returns the combined mu of the binary pair.
func get_effective_parent_mu() -> float:
	if _gravity_strength > 0.0:
		return _gravity_strength
	for child_orbiter: JrRailOrbiter in _child_orbiters.keys():
		if child_orbiter == null:
			continue
		var partner: JrRailOrbiter = child_orbiter.binary_partner
		if partner != null and partner != child_orbiter and _child_orbiters.has(partner):
			return child_orbiter.get_gravity_strength() + partner.get_gravity_strength()
	return 0.0


## Called downward by [JrOrbitDriver] to synchronize the system's active distance
## and time units on this element and recompute its scaled [member _gravity_strength].
func set_driver_units(
		dist_unit: JrOrreryUnits.DistanceUnit,
		time_unit: JrOrreryUnits.TimeUnit
) -> void:
	var units_changed: bool = _driver_distance_unit != dist_unit or _driver_time_unit != time_unit
	_driver_distance_unit = dist_unit
	_driver_time_unit = time_unit
	_recompute_gravity_strength_from_mass()
	if units_changed:
		_on_driver_units_updated()


## Returns the active [enum JrOrreryUnits.DistanceUnit] supplied by the owning [JrOrbitDriver].
func get_driver_distance_unit() -> JrOrreryUnits.DistanceUnit:
	return _driver_distance_unit


## Returns the active [enum JrOrreryUnits.TimeUnit] supplied by the owning [JrOrbitDriver].
func get_driver_time_unit() -> JrOrreryUnits.TimeUnit:
	return _driver_time_unit


## Returns the active [JrReferenceDirection] used to orient this element's path.
func get_reference_plane() -> JrReferenceDirection:
	return _reference_plane


## Updates the active [JrReferenceDirection] used to orient this element's path.
func set_reference_plane(new_reference_plane: JrReferenceDirection) -> void:
	if new_reference_plane != null:
		_reference_plane = new_reference_plane


## Returns [code]true[/code] if the active [JrReferenceDirection] constrains all orbits,
## ballistic trajectories, and axial tilts to the ecliptic plane.
func is_locked_to_ecliptic() -> bool:
	return _reference_plane != null and _reference_plane.lock_to_ecliptic


## Returns [code]true[/code] if the owning [JrOrbitDriver] is actively stepping simulation time.
func is_simulation_playing() -> bool:
	return _simulation_playing


## Updates whether the owning [JrOrbitDriver] is actively stepping simulation time.
func set_simulation_playing(playing: bool) -> void:
	if _simulation_playing != playing:
		_simulation_playing = playing
		orbit_visuals_changed.emit()


## Returns [code]true[/code] if this element represents a body with a valid trajectory
## ([JrRailOrbiter] with valid conic elements or [JrBallisticOrbiter]). Returns [code]false[/code]
## on root [JrOrbitDriver] nodes.
func is_trajectory_valid() -> bool:
	return false


## Returns [code]true[/code] if this element exposes interactive Keplerian conic gizmo
## handles in the 2D and 3D editor viewports.
func has_keplerian_gizmo() -> bool:
	return false


## Returns [code]true[/code] if this element's Keplerian conic parameters are valid.
func is_orbit_valid() -> bool:
	return false


## Returns [code]true[/code] if this element's Keplerian orbit is gravitationally bound ([code]e <
## 1[/code]).
func is_bound_orbit() -> bool:
	return false


## Returns [code]true[/code] if this element's Keplerian orbit is unbound ([code]e >= 1[/code]).
func is_unbound_orbit() -> bool:
	return false


## Returns [code]true[/code] if this element currently holds a valid animated editor
## preview ghost position.
func has_ghost_position() -> bool:
	return false


## Returns the animated editor preview ghost's position relative to the parent focus
## in the coordinate space of [member _reference_plane].
func get_ghost_relative_position() -> Vector3:
	return Vector3.ZERO


## Returns the animated editor preview ghost's 3D orientation [Basis] in the
## coordinate space of [member _reference_plane].
func get_ghost_body_basis() -> Basis:
	return Basis.IDENTITY


## Returns 3D trajectory path points relative to the parent focus for editor and
## runtime visualization. Overridden by [JrRailOrbiter] and [JrBallisticOrbiter].
func get_trajectory_path_points(_point_count: int = 64) -> PackedVector3Array:
	return []


## Projects [method get_trajectory_path_points] onto top-down 2D X-Z coordinates
## scaled by [param scale_factor]. Overridden by [JrRailOrbiter] for unbound scaling.
func get_trajectory_path_points_2d(
		point_count: int = 64,
		scale_factor: float = 1.0
) -> PackedVector2Array:
	var points_3d: PackedVector3Array = get_trajectory_path_points(point_count)
	var points_2d: PackedVector2Array = []
	points_2d.resize(points_3d.size())
	for i: int in range(points_3d.size()):
		var pt: Vector3 = points_3d[i]
		points_2d[i] = Vector2(pt.x, pt.z) * scale_factor
	return points_2d


## Returns this element's cached position vector relative to its parent focus
## in the owning [JrOrbitDriver]'s coordinate space.
func get_focus_relative_position() -> Vector3:
	return _focus_relative_position


## Alias for [method get_focus_relative_position] preserving compatibility with
## driver-relative position queries.
func get_driver_relative_position() -> Vector3:
	return _focus_relative_position


## Returns this element's 3D position relative to the parent focus at [param _time]
## in the coordinate space of [param _reference]. Overridden by [JrRailOrbiter] and
## [JrBallisticOrbiter].
func get_relative_position(_time: float, _reference: JrReferenceDirection = null) -> Vector3:
	return _focus_relative_position


## Returns this element's cached orbital velocity vector relative to its immediate
## parent focus in the owning [JrOrbitDriver]'s reference coordinate space.
## Subclasses ([JrRailOrbiter]) may also evaluate velocity at a specific [param _time]
## and [param _reference] direction.
func get_relative_velocity(_time: float = NAN, _reference: JrReferenceDirection = null) -> Vector3:
	return _relative_velocity


## Returns this element's cached orbital velocity vector relative to its immediate
## parent focus in the owning [JrOrbitDriver]'s reference coordinate space.
func get_focus_relative_velocity() -> Vector3:
	return _relative_velocity


## Returns this element's cached global orbital velocity vector (accumulated across
## all parent orbiters) relative to the root [JrOrbitDriver] in the driver's
## reference coordinate space.
func get_global_velocity() -> Vector3:
	return _global_velocity


## Alias for [method get_global_velocity] returning the accumulated orbital velocity
## vector relative to the root [JrOrbitDriver].
func get_driver_relative_velocity() -> Vector3:
	return _global_velocity


## Returns this element's parent-relative orbital velocity vector expressed in the
## immediate parent [JrOrreryElement]'s local coordinate frame (with parent spin
## and axial tilt canceled).
func get_local_orbital_velocity() -> Vector3:
	return _parent_driver_basis.transposed() * _relative_velocity


## Returns this element's cached orientation [Basis] in the owning
## [JrOrbitDriver]'s coordinate space.
func get_driver_relative_basis() -> Basis:
	return _driver_relative_basis


## Returns the parent element's driver-relative [Basis] most recently supplied
## by the owning [JrOrbitDriver].
func get_parent_driver_basis() -> Basis:
	return _parent_driver_basis


## Returns the calculated local [Transform3D] relative to this element's parent
## [JrOrreryElement] (with parent spin and axial tilt canceled).
func get_local_orbital_transform() -> Transform3D:
	return _local_orbital_transform


## Returns the calculated heliocentric [Transform3D] in the owning
## [JrOrbitDriver]'s unshifted frame.
func get_driver_relative_transform() -> Transform3D:
	return _driver_relative_transform


## Returns the calculated system [Transform3D] after applying any active
## perspective offset from the owning [JrOrbitDriver].
func get_system_transform() -> Transform3D:
	return _system_transform


## Updates [member _system_transform] and emits [signal transform_updated].
func set_system_transform(new_system_transform: Transform3D) -> void:
	_system_transform = new_system_transform
	transform_updated.emit()


## Returns the transient runtime gravity-well index assigned by [JrOrbitDriver].
func get_runtime_well_index() -> int:
	return _runtime_well_index;


## Sets the transient runtime gravity-well index assigned by [JrOrbitDriver].
func set_runtime_well_index(well_index: int) -> void:
	_runtime_well_index = well_index;


## Returns the 3D periapsis position relative to the central focus. Overridden by Keplerian
## elements.
func get_periapsis_relative_position(_reference: JrReferenceDirection = null) -> Vector3:
	return Vector3.ZERO


## Returns the 3D apoapsis position relative to the central focus. Overridden by Keplerian
## elements.
func get_apoapsis_relative_position(_reference: JrReferenceDirection = null) -> Vector3:
	return Vector3.INF


## Returns the 3D semi-parameter position relative to the central focus. Overridden by Keplerian
## elements.
func get_semi_parameter_relative_position(_reference: JrReferenceDirection = null) -> Vector3:
	return Vector3.ZERO


## Returns the radial distance used for the ascending node and inclination gizmo handles.
func get_orbital_plane_gizmo_radius() -> float:
	return 1.0


## Returns the 3D position of the ascending node gizmo grabber relative to the focus.
func get_ascending_node_gizmo_position() -> Vector3:
	return Vector3.ZERO


## Returns the 3D position of the inclination gizmo grabber relative to the focus.
func get_inclination_gizmo_position() -> Vector3:
	return Vector3.ZERO


## Returns the 3D [Basis] oriented to the ascending node in the ecliptic plane.
func get_nodal_ecliptic_basis(_reference: JrReferenceDirection = null) -> Basis:
	return Basis.IDENTITY


## Returns the 3D [Basis] after applying both ascending node and inclination.
func get_inclined_nodal_basis(_reference: JrReferenceDirection = null) -> Basis:
	return Basis.IDENTITY


## Returns the 3D [Basis] of the orbital perifocal plane.
func get_orbital_plane_basis(_reference: JrReferenceDirection = null) -> Basis:
	return Basis.IDENTITY


## Returns the 3D [Basis] of the body after applying axial tilt.
func get_tilted_body_basis(_reference: JrReferenceDirection = null) -> Basis:
	return _driver_relative_basis


## Returns the 3D orientation [Basis] of the body at [param _time].
func get_body_basis(_time: float, _reference: JrReferenceDirection = null) -> Basis:
	return _driver_relative_basis


## Returns the unit vector in the orbital plane toward which the north pole tilts.
func get_axial_tilt_direction(_reference: JrReferenceDirection = null) -> Vector3:
	return Vector3.RIGHT


## Returns the unit hinge axis in the orbital plane around which axial tilt rotates.
func get_axial_tilt_hinge_axis(_reference: JrReferenceDirection = null) -> Vector3:
	return Vector3.FORWARD


## Converts [param _true_anomaly] into mean anomaly.
func get_mean_anomaly_from_true_anomaly(_true_anomaly: float) -> float:
	return 0.0


## Generates 3D points tracing the conic path in [param _reference] space.
func get_orbit_path_points(
		_reference: JrReferenceDirection = null,
		_point_count: int = 64,
		_close_loop: bool = true,
		_max_unbound_distance: float = 500.0
) -> PackedVector3Array:
	return get_trajectory_path_points(_point_count)


## Reads a named Keplerian or rotational float property [param prop] on this element
## (such as [code]&"periapsis"[/code] or [code]&"eccentricity"[/code]).
func get_keplerian_prop(prop: StringName) -> float:
	var val: Variant = get(prop)
	return float(val) if val != null else 0.0


## Writes a named Keplerian or rotational float property [param prop] on this element.
func set_keplerian_prop(prop: StringName, value: float) -> void:
	set(prop, value)


## Recomputes [member _gravity_strength] in driver units from [member _mass] and
## [member _mass_unit].
func _recompute_gravity_strength_from_mass() -> void:
	_gravity_strength = JrOrreryUnits.compute_mu_in_driver_units(
			_mass,
			_mass_unit,
			_driver_distance_unit,
			_driver_time_unit
	)


## Virtual hook invoked when [method set_driver_units] changes the active unit scale.
func _on_driver_units_updated() -> void:
	pass


## Rescans direct children to synchronize [member _child_orbiters] and
## [member _child_elements], then updates editor configuration warnings.
func _refresh_tracked_nodes() -> void:
	for existing_orbiter: JrRailOrbiter in _child_orbiters.keys():
		_unregister_child_orbiter(existing_orbiter)
	_child_orbiters.clear()

	for existing_elem: JrOrreryElement in _child_elements.keys():
		_unregister_child_element(existing_elem)
	_child_elements.clear()

	for child: Node in get_children():
		if child is JrRailOrbiter:
			_register_child_orbiter(child as JrRailOrbiter)
		if child is JrOrreryElement:
			_register_child_element(child as JrOrreryElement)

	if Engine.is_editor_hint():
		update_configuration_warnings()


## Registers a direct child [param element], subscribes to its subtree signals,
## and emits [signal child_element_added] for it and any existing descendants.
func _register_child_element(element: JrOrreryElement) -> void:
	if element == null or _child_elements.has(element):
		return

	_child_elements[element] = true
	if not element.child_element_added.is_connected(_on_descendant_element_added):
		element.child_element_added.connect(_on_descendant_element_added)
	if not element.child_element_removed.is_connected(_on_descendant_element_removed):
		element.child_element_removed.connect(_on_descendant_element_removed)

	child_element_added.emit(self, element)
	_emit_existing_descendant_elements_added(element)


## Unregisters a direct child [param element], disconnects its subtree signals,
## and emits [signal child_element_removed] for its descendants and itself.
func _unregister_child_element(element: JrOrreryElement) -> void:
	if element == null or not _child_elements.has(element):
		return

	_emit_existing_descendant_elements_removed(element)
	if element.child_element_added.is_connected(_on_descendant_element_added):
		element.child_element_added.disconnect(_on_descendant_element_added)
	if element.child_element_removed.is_connected(_on_descendant_element_removed):
		element.child_element_removed.disconnect(_on_descendant_element_removed)

	_child_elements.erase(element)
	child_element_removed.emit(self, element)


## Recursively emits [signal child_element_added] for any elements already present
## under [param root_element].
func _emit_existing_descendant_elements_added(root_element: JrOrreryElement) -> void:
	for grandchild: JrOrreryElement in root_element.get_child_elements():
		child_element_added.emit(root_element, grandchild)
		_emit_existing_descendant_elements_added(grandchild)


## Recursively emits [signal child_element_removed] for any elements present
## under [param root_element] prior to removal.
func _emit_existing_descendant_elements_removed(root_element: JrOrreryElement) -> void:
	for grandchild: JrOrreryElement in root_element.get_child_elements():
		_emit_existing_descendant_elements_removed(grandchild)
		child_element_removed.emit(root_element, grandchild)


## Registers a direct child [param orbiter], subscribes to its subtree signals,
## and emits [signal child_orbiter_added] for it and any existing descendants.
func _register_child_orbiter(orbiter: JrRailOrbiter) -> void:
	if orbiter == null or _child_orbiters.has(orbiter):
		return

	_child_orbiters[orbiter] = true
	if not orbiter.child_orbiter_added.is_connected(_on_descendant_orbiter_added):
		orbiter.child_orbiter_added.connect(_on_descendant_orbiter_added)
	if not orbiter.child_orbiter_removed.is_connected(_on_descendant_orbiter_removed):
		orbiter.child_orbiter_removed.connect(_on_descendant_orbiter_removed)

	child_orbiter_added.emit(self, orbiter)
	_emit_existing_descendants_added(orbiter)


## Unregisters a direct child [param orbiter], disconnects its subtree signals,
## and emits [signal child_orbiter_removed] for its descendants and itself.
func _unregister_child_orbiter(orbiter: JrRailOrbiter) -> void:
	if orbiter == null or not _child_orbiters.has(orbiter):
		return

	_emit_existing_descendants_removed(orbiter)
	if orbiter.child_orbiter_added.is_connected(_on_descendant_orbiter_added):
		orbiter.child_orbiter_added.disconnect(_on_descendant_orbiter_added)
	if orbiter.child_orbiter_removed.is_connected(_on_descendant_orbiter_removed):
		orbiter.child_orbiter_removed.disconnect(_on_descendant_orbiter_removed)

	_child_orbiters.erase(orbiter)
	child_orbiter_removed.emit(self, orbiter)


## Recursively emits [signal child_orbiter_added] for any orbiters already present
## under [param root_orbiter].
func _emit_existing_descendants_added(root_orbiter: JrRailOrbiter) -> void:
	for grandchild: JrRailOrbiter in root_orbiter.get_child_orbiters():
		child_orbiter_added.emit(root_orbiter, grandchild)
		_emit_existing_descendants_added(grandchild)


## Recursively emits [signal child_orbiter_removed] for any orbiters present
## under [param root_orbiter] prior to removal.
func _emit_existing_descendants_removed(root_orbiter: JrRailOrbiter) -> void:
	for grandchild: JrRailOrbiter in root_orbiter.get_child_orbiters():
		_emit_existing_descendants_removed(grandchild)
		child_orbiter_removed.emit(root_orbiter, grandchild)


## Handles direct child nodes entering the tree.
func _on_child_entered_tree(node: Node) -> void:
	if node is JrRailOrbiter:
		_register_child_orbiter(node as JrRailOrbiter)
	if node is JrOrreryElement:
		_register_child_element(node as JrOrreryElement)


## Handles direct child nodes exiting the tree.
func _on_child_exiting_tree(node: Node) -> void:
	if node is JrRailOrbiter:
		_unregister_child_orbiter(node as JrRailOrbiter)
	if node is JrOrreryElement:
		_unregister_child_element(node as JrOrreryElement)


## Relays [signal child_orbiter_added] upward from descendant orbiters.
func _on_descendant_orbiter_added(parent: JrOrreryElement, child: JrRailOrbiter) -> void:
	child_orbiter_added.emit(parent, child)


## Relays [signal child_orbiter_removed] upward from descendant orbiters.
func _on_descendant_orbiter_removed(parent: JrOrreryElement, child: JrRailOrbiter) -> void:
	child_orbiter_removed.emit(parent, child)


## Relays [signal child_element_added] upward from descendant elements.
func _on_descendant_element_added(parent: JrOrreryElement, child: JrOrreryElement) -> void:
	child_element_added.emit(parent, child)


## Relays [signal child_element_removed] upward from descendant elements.
func _on_descendant_element_removed(parent: JrOrreryElement, child: JrOrreryElement) -> void:
	child_element_removed.emit(parent, child)
