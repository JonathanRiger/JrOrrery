@tool
@icon("res://addons/jr_orrery/icons/orbiter_3d.svg")
class_name JrOrbiter3D
extends Node3D
## Visual 3D follower node ([Node3D]) driven by a [JrOrreryElement] ([JrRailOrbiter],
## [JrBallisticOrbiter], or [JrOrbitDriver]).
##
## Subscribes to [signal JrOrreryElement.transform_updated] on [member orbiter] and
## applies its calculated 3D orbital or ballistic position and rotational [Basis].
## Supports both nested parent-child visual hierarchies (canceling parent axial tilt and spin)
## and flat sibling hierarchies, and renders the 3D trajectory path and animated
## wireframe ghost in the Godot Editor.

## Emitted when the orbiter's parameters, scale, or cached editor path change.
signal orbit_visuals_changed

## Determines how this 3D node applies the transform calculated by [member orbiter].
enum TransformMode {
	## Automatically uses [constant LOCAL] when parented to another [JrOrbiter3D],
	## or [constant SYSTEM] otherwise.
	AUTO,
	## Applies the full system/perspective-adjusted transform relative to the root driver.
	SYSTEM,
	## Applies the parent-focus-relative local orbital transform (with parent spin/tilt canceled).
	LOCAL,
}

## Number of segments per great circle when drawing the 3D wireframe ghost sphere.
const GHOST_RING_SEGMENTS: int = 24

## Ratio of [member JrRailOrbiter.periapsis] used for the ascending node and inclination gizmo
## radius.
const ORBITAL_PLANE_GIZMO_RADIUS_RATIO: float = JrRailOrbiter.ORBITAL_PLANE_GIZMO_RADIUS_RATIO

## Multiplier applied to the body gizmo radius for the axial tilt north-pole grabber handle.
const AXIAL_TILT_GIZMO_RADIUS_SCALE: float = JrRailOrbiter.AXIAL_TILT_GIZMO_RADIUS_SCALE

## Multiplier applied to the body gizmo radius for the equatorial starting day angle grabber
## handle.
const STARTING_DAY_GIZMO_RADIUS_SCALE: float = JrRailOrbiter.STARTING_DAY_GIZMO_RADIUS_SCALE

## The simulation [JrOrreryElement] ([JrOrbiter] or [JrOrbitDriver]) that drives
## this 3D visual node's transform.
@export var orbiter: JrOrreryElement = null:
	set(value):
		if orbiter == value:
			return
		_disconnect_orbiter_signals()
		orbiter = value
		if is_node_ready():
			_connect_orbiter_signals()
			_on_orbiter_rebound()

## Controls whether this node applies the parent-relative local transform or the
## full system/perspective transform from [member orbiter].
@export var transform_mode := TransformMode.AUTO:
	set(value):
		if transform_mode != value:
			transform_mode = value
			_on_orbiter_transform_updated()
			_redraw_editor_visuals()

## Multiplier applied to 3D orbital translation coordinates while keeping the
## rotational [Basis] orthonormal.
@export_range(0.000001, 10000.0, 0.000001, "exp", "or_greater")
var position_scale: float = 1.0:
	set(value):
		var clamped: float = maxf(1e-9, value)
		if not is_equal_approx(position_scale, clamped):
			position_scale = clamped
			_on_orbiter_transform_updated()
			_rebuild_orbit_path_cache()

@export_group("Editor Visualization")
## If [code]true[/code], renders the conic orbital path in the 3D editor viewport.
@export var draw_orbit_in_editor: bool = true:
	set(value):
		if draw_orbit_in_editor != value:
			draw_orbit_in_editor = value
			_redraw_editor_visuals()

## Color of the orbital path line drawn in the 3D editor viewport.
@export var editor_orbit_color := Color(0.4, 0.7, 1.0, 0.85):
	set(value):
		if editor_orbit_color != value:
			editor_orbit_color = value
			_redraw_editor_visuals()
			orbit_visuals_changed.emit()

## Number of segments used to sample the orbital path in the 3D editor viewport.
@export_range(8, 512, 1)
var editor_orbit_segments: int = 64:
	set(value):
		var clamped: int = maxi(8, value)
		if editor_orbit_segments != clamped:
			editor_orbit_segments = clamped
			_rebuild_orbit_path_cache()

## If [code]true[/code], renders an animated 3D ghost marker along the orbital path
## in the 3D editor viewport to preview orbital motion and spin without moving the node.
@export var show_editor_ghost: bool = true:
	set(value):
		if show_editor_ghost != value:
			show_editor_ghost = value
			if show_editor_ghost and Engine.is_editor_hint():
				_refresh_ghost_state()
			_redraw_editor_visuals()

## Color of the animated ghost marker rendered in the 3D editor viewport.
@export var editor_ghost_color := Color(1.0, 1.0, 1.0, 0.65):
	set(value):
		if editor_ghost_color != value:
			editor_ghost_color = value
			_redraw_editor_visuals()

## Radius (in local 3D units) of the animated ghost marker rendered in the 3D editor viewport.
@export_range(0.05, 64.0, 0.05)
var editor_ghost_radius: float = 0.5:
	set(value):
		if not is_equal_approx(editor_ghost_radius, value):
			editor_ghost_radius = value
			_redraw_editor_visuals()

## Cached polyline vertices representing the 3D orbital path relative to the parent focus.
var _cached_path_points: PackedVector3Array = []

## Relative position of the animated editor preview ghost with respect to the parent focus.
var _ghost_relative_position := Vector3.ZERO

## Orientation [Basis] of the animated editor preview ghost at the preview timestamp.
var _ghost_body_basis := Basis.IDENTITY

## Whether [member _ghost_relative_position] currently holds a valid preview position.
var _has_ghost_position: bool = false

## Initial local position captured when [member orbiter] is a [JrOrbitDriver].
var _anchor_position := Vector3.ZERO

## Initial local orientation [Basis] captured when [member orbiter] is a [JrOrbitDriver].
var _anchor_basis := Basis.IDENTITY

## Guard flag set while this script is updating [member Node3D.transform] internally.
var _updating_transform_internally: bool = false

## Flag indicating that the local transform was modified externally in the editor.
var _transform_modified_externally: bool = false

## Internal editor-only [MeshInstance3D] used to render the 3D orbit path and ghost.
var _editor_visual_instance: MeshInstance3D = null

## Internal [ImmediateMesh] populated with the 3D orbit polyline and ghost wireframe.
var _editor_immediate_mesh: ImmediateMesh = null

## Unshaded vertex-color material assigned to [member _editor_visual_instance].
var _editor_visual_material: StandardMaterial3D = null


## Captures initial anchor state, creates editor visualization nodes when in the editor,
## connects to [member orbiter], and builds the orbit path cache.
func _ready() -> void:
	_anchor_position = position
	_anchor_basis = basis
	if Engine.is_editor_hint():
		_setup_editor_visual_nodes()
	_connect_orbiter_signals()
	set_notify_local_transform(Engine.is_editor_hint())
	_on_orbiter_transform_updated()
	_rebuild_orbit_path_cache()


## Handles hierarchy and local transform notifications to refresh editor warnings and visuals.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PARENTED or what == NOTIFICATION_UNPARENTED:
		update_configuration_warnings()
		if is_inside_tree():
			_on_orbiter_transform_updated()
			_redraw_editor_visuals()
	elif what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED and Engine.is_editor_hint():
		_on_editor_local_transform_changed()
		_update_editor_visual_transform()


## Validates that [member orbiter] is assigned and valid, and checks for external
## transform edits in the Godot Editor.
func _get_configuration_warnings() -> PackedStringArray:
	var warnings: PackedStringArray = []

	if orbiter == null:
		warnings.append(
				"An orbiter (JrRailOrbiter, JrBallisticOrbiter, or JrOrbitDriver) "
				+ "node must be assigned."
		)
	elif orbiter is JrRailOrbiter and not (orbiter as JrRailOrbiter).is_orbit_valid():
		warnings.append(
				"The assigned JrRailOrbiter's orbital parameters "
				+ "do not form a valid conic section."
		)

	if (
			get_target_trajectory_element() != null
			and (_transform_modified_externally or _is_transform_out_of_sync())
	):
		warnings.append(
				"Transform was modified outside of script. "
				+ "Local transform is controlled by the assigned simulation node."
		)

	return warnings


## Returns [member orbiter] cast to [JrRailOrbiter], or [code]null[/code] if unassigned
## or if [member orbiter] is another element type.
func get_target_orbiter() -> JrRailOrbiter:
	return orbiter as JrRailOrbiter


## Returns [member orbiter] when it supports Keplerian conic gizmo handles
## ([JrRailOrbiter] or [JrBallisticOrbiter] in
## [constant JrBallisticOrbiter.InitialStateMode.KEPLERIAN_ELEMENTS] mode).
func get_target_keplerian_element() -> JrOrreryElement:
	if orbiter != null and orbiter.has_keplerian_gizmo():
		return orbiter
	return null


## Returns [member orbiter] cast to [JrBallisticOrbiter], or [code]null[/code] if unassigned
## or if [member orbiter] is another element type.
func get_target_ballistic() -> JrBallisticOrbiter:
	return orbiter as JrBallisticOrbiter


## Returns [member orbiter] when it is a trajectory-bearing simulation element
## ([JrRailOrbiter] with valid conic parameters or [JrBallisticOrbiter]), or [code]null[/code].
func get_target_trajectory_element() -> JrOrreryElement:
	if orbiter == null or orbiter is JrOrbitDriver:
		return null
	if not orbiter.is_trajectory_valid():
		return null
	return orbiter


## Returns the active [JrReferenceDirection] used to orient this orbiter's path.
func get_reference_plane() -> JrReferenceDirection:
	if orbiter == null:
		return null
	return orbiter.get_reference_plane()


## Returns this orbiter's cached position vector relative to its parent focus
## in the owning [JrOrbitDriver]'s coordinate space, scaled by [member position_scale].
func get_driver_relative_position() -> Vector3:
	if orbiter == null:
		return Vector3.ZERO
	return orbiter.get_focus_relative_position() * position_scale


## Returns this orbiter's cached orientation [Basis] in the owning
## [JrOrbitDriver]'s coordinate space.
func get_driver_relative_basis() -> Basis:
	if orbiter == null:
		return Basis.IDENTITY
	return orbiter.get_driver_relative_basis()


## Returns the parent orbiter's driver-relative [Basis] most recently supplied
## by the owning [JrOrbitDriver].
func get_parent_driver_basis() -> Basis:
	if orbiter == null:
		return Basis.IDENTITY
	return orbiter.get_parent_driver_basis()


## Returns [code]true[/code] if the owning [JrOrbitDriver] is actively stepping simulation time.
func is_simulation_playing() -> bool:
	if orbiter == null:
		return false
	return orbiter.is_simulation_playing()


## Returns [code]true[/code] when this node is applying parent-relative local transforms
## rather than full system transforms.
func is_using_local_transform() -> bool:
	if transform_mode == TransformMode.LOCAL:
		return true
	if transform_mode == TransformMode.SYSTEM:
		return false
	return get_parent() is JrOrbiter3D


## Returns the [Transform3D] that maps scaled driver-relative coordinates centered at
## the parent focus into 3D global world space.
func get_focus_to_world_transform() -> Transform3D:
	return global_transform * _get_focus_to_local_transform()


## Returns the radial distance from the parent focus used for the ascending node
## and inclination gizmo grabbers in scaled driver-focus space.
func get_orbital_plane_gizmo_radius() -> float:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return 1.0
	return target_elem.get_orbital_plane_gizmo_radius() * position_scale


## Returns the 3D position of the ascending node (RAAN) grabber relative to the
## parent focus in scaled driver-focus space.
func get_ascending_node_gizmo_position() -> Vector3:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return Vector3.ZERO
	return target_elem.get_ascending_node_gizmo_position() * position_scale


## Returns the 3D position of the orbital inclination grabber relative to the
## parent focus in scaled driver-focus space.
func get_inclination_gizmo_position() -> Vector3:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return Vector3.ZERO
	return target_elem.get_inclination_gizmo_position() * position_scale


## Returns the 3D position of the axial tilt grabber (at the tip of the north spin pole)
## relative to the parent focus in scaled driver-focus space.
func get_axial_tilt_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target_elem.get_relative_position(0.0) * position_scale
	var north_pole: Vector3 = target_elem.get_tilted_body_basis().y
	return body_pos + north_pole * (body_gizmo_radius * AXIAL_TILT_GIZMO_RADIUS_SCALE)


## Returns the 3D position of the axial tilt longitude grabber (in the orbital plane
## around the body) relative to the parent focus in scaled driver-focus space.
func get_axial_tilt_longitude_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target_elem.get_relative_position(0.0) * position_scale
	var tilt_dir: Vector3 = target_elem.get_axial_tilt_direction()
	return body_pos + tilt_dir * body_gizmo_radius


## Returns the 3D position of the starting day angle ([member JrRailOrbiter.rotation_at_epoch])
## grabber on the body's equatorial ring relative to the parent focus in scaled driver-focus space.
func get_starting_day_angle_gizmo_position(body_gizmo_radius: float) -> Vector3:
	var target_elem: JrOrreryElement = get_target_keplerian_element()
	if target_elem == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target_elem.get_relative_position(0.0) * position_scale
	var prime_meridian: Vector3 = -target_elem.get_body_basis(0.0).z
	return body_pos + prime_meridian * (body_gizmo_radius * STARTING_DAY_GIZMO_RADIUS_SCALE)


## Connects reactive signals on [member orbiter].
func _connect_orbiter_signals() -> void:
	if orbiter == null:
		return
	if not orbiter.transform_updated.is_connected(_on_orbiter_transform_updated):
		orbiter.transform_updated.connect(_on_orbiter_transform_updated)
	if not orbiter.editor_ghost_updated.is_connected(_on_orbiter_ghost_updated):
		orbiter.editor_ghost_updated.connect(_on_orbiter_ghost_updated)
	if not orbiter.orbit_visuals_changed.is_connected(_on_orbiter_visuals_changed):
		orbiter.orbit_visuals_changed.connect(_on_orbiter_visuals_changed)


## Disconnects reactive signals from [member orbiter].
func _disconnect_orbiter_signals() -> void:
	if orbiter == null:
		return
	if orbiter.transform_updated.is_connected(_on_orbiter_transform_updated):
		orbiter.transform_updated.disconnect(_on_orbiter_transform_updated)
	if orbiter.editor_ghost_updated.is_connected(_on_orbiter_ghost_updated):
		orbiter.editor_ghost_updated.disconnect(_on_orbiter_ghost_updated)
	if orbiter.orbit_visuals_changed.is_connected(_on_orbiter_visuals_changed):
		orbiter.orbit_visuals_changed.disconnect(_on_orbiter_visuals_changed)


## Handles rebinding when [member orbiter] is reassigned.
func _on_orbiter_rebound() -> void:
	if is_inside_tree():
		_on_orbiter_transform_updated()
	update_configuration_warnings()
	_rebuild_orbit_path_cache()


## Updates this node's 3D transform from [member orbiter]'s calculated transform.
func _on_orbiter_transform_updated() -> void:
	if orbiter == null or not is_inside_tree():
		return

	if orbiter is JrOrbitDriver:
		_apply_driver_follower_transform()
		return

	var target_element: JrOrreryElement = get_target_trajectory_element()
	if target_element == null:
		return

	var target_xf: Transform3D = _compute_expected_3d_transform(target_element)
	_apply_orbital_transform(target_xf.origin, target_xf.basis)


## Applies the perspective-shifted root transform when [member orbiter] is a [JrOrbitDriver].
func _apply_driver_follower_transform() -> void:
	if Engine.is_editor_hint():
		_anchor_position = position
		_anchor_basis = basis
		return

	var sys_xf: Transform3D = orbiter.get_system_transform()
	_updating_transform_internally = true
	basis = (_anchor_basis * sys_xf.basis).orthonormalized()
	position = _anchor_position + (_anchor_basis * (sys_xf.origin * position_scale))
	_updating_transform_internally = false


## Computes the expected local [Transform3D] for [param target_element] according
## to [method is_using_local_transform].
func _compute_expected_3d_transform(target_element: JrOrreryElement) -> Transform3D:
	if is_using_local_transform():
		var local_xf: Transform3D = target_element.get_local_orbital_transform()
		return Transform3D(local_xf.basis, local_xf.origin * position_scale)
	var sys_xf: Transform3D = target_element.get_system_transform()
	return Transform3D(sys_xf.basis, sys_xf.origin * position_scale)


## Sets [member Node3D.position] and [member Node3D.basis] internally without
## triggering an external-edit warning.
func _apply_orbital_transform(new_position: Vector3, new_basis: Basis) -> void:
	_updating_transform_internally = true
	position = new_position
	basis = new_basis
	if _transform_modified_externally:
		_transform_modified_externally = false
		update_configuration_warnings()
	_updating_transform_internally = false
	if Engine.is_editor_hint():
		_update_editor_visual_transform()


## Returns the [Transform3D] that maps scaled driver-focus coordinates into
## this node's local coordinate space.
func _get_focus_to_local_transform() -> Transform3D:
	var target_element: JrOrreryElement = get_target_trajectory_element()
	if target_element == null:
		return Transform3D.IDENTITY
	var driver_to_local_basis: Basis = target_element.get_driver_relative_basis().transposed()
	var focus_offset: Vector3 = target_element.get_focus_relative_position() * position_scale
	return Transform3D(
			driver_to_local_basis,
			-(driver_to_local_basis * focus_offset)
	)


## Updates the 3D ghost state when [member orbiter] emits
## [signal JrOrreryElement.editor_ghost_updated].
func _on_orbiter_ghost_updated(_preview_time: float) -> void:
	if not Engine.is_editor_hint() or not show_editor_ghost:
		return
	if _cached_path_points.is_empty() and get_target_trajectory_element() != null:
		_rebuild_orbit_path_cache()
		return
	_refresh_ghost_state()
	_redraw_editor_visuals()


## Rebuilds the 3D orbit path cache when [member orbiter] emits
## [signal JrOrreryElement.orbit_visuals_changed].
func _on_orbiter_visuals_changed() -> void:
	_on_orbiter_transform_updated()
	update_configuration_warnings()
	_rebuild_orbit_path_cache()


## Updates [member _ghost_relative_position] and [member _ghost_body_basis] from [member orbiter].
func _refresh_ghost_state() -> void:
	var target_element: JrOrreryElement = get_target_trajectory_element()
	if target_element == null or not target_element.has_ghost_position():
		_has_ghost_position = false
		return
	_ghost_relative_position = target_element.get_ghost_relative_position() * position_scale
	_ghost_body_basis = target_element.get_ghost_body_basis()
	_has_ghost_position = true


## Recomputes [member _cached_path_points] for editor visualization and redraws
## the 3D immediate mesh.
func _rebuild_orbit_path_cache() -> void:
	if not Engine.is_editor_hint() and not draw_orbit_in_editor:
		_cached_path_points = []
		_has_ghost_position = false
		return
	var target_element: JrOrreryElement = get_target_trajectory_element()
	if target_element == null or target_element.get_reference_plane() == null:
		_cached_path_points = []
		_has_ghost_position = false
		_redraw_editor_visuals()
		orbit_visuals_changed.emit()
		return

	var unscaled_points: PackedVector3Array = target_element.get_trajectory_path_points(
			editor_orbit_segments
	)
	if is_equal_approx(position_scale, 1.0):
		_cached_path_points = unscaled_points
	else:
		_cached_path_points = []
		_cached_path_points.resize(unscaled_points.size())
		for i: int in range(unscaled_points.size()):
			_cached_path_points[i] = unscaled_points[i] * position_scale

	_refresh_ghost_state()
	_redraw_editor_visuals()
	orbit_visuals_changed.emit()


## Detects when the user modifies a simulation-driven node's transform directly
## in the editor, emits a warning, and restores the calculated transform.
func _on_editor_local_transform_changed() -> void:
	if _updating_transform_internally or not is_inside_tree():
		return
	if orbiter is JrOrbitDriver:
		_anchor_position = position
		_anchor_basis = basis
		return
	if _is_transform_out_of_sync():
		push_warning(
				(
						"JrOrbiter3D '%s': Transform was modified outside of script. "
						+ "Local transform is controlled by the assigned simulation node."
				) % name
		)
		_on_orbiter_transform_updated()


## Returns [code]true[/code] if the node's current local transform deviates from
## the script-driven orbital or ballistic position or basis.
func _is_transform_out_of_sync() -> bool:
	var target_element: JrOrreryElement = get_target_trajectory_element()
	if target_element == null:
		return false
	if not scale.is_equal_approx(Vector3.ONE):
		return true
	var expected_xf: Transform3D = _compute_expected_3d_transform(target_element)
	if position.distance_squared_to(expected_xf.origin) > 0.0001:
		return true
	if not basis.is_equal_approx(expected_xf.basis):
		return true
	return false


## Creates the internal non-saved [MeshInstance3D] and [ImmediateMesh] used for
## drawing the 3D orbit line and animated ghost in the editor viewport.
func _setup_editor_visual_nodes() -> void:
	if _editor_visual_instance != null and is_instance_valid(_editor_visual_instance):
		return
	for child: Node in get_children(true):
		if child.name == "_JrOrbiter3DEditorVisuals":
			remove_child(child)
			child.queue_free()
	_editor_immediate_mesh = ImmediateMesh.new()
	_editor_visual_material = StandardMaterial3D.new()
	_editor_visual_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_editor_visual_material.vertex_color_use_as_albedo = true
	_editor_visual_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_editor_visual_material.cull_mode = BaseMaterial3D.CULL_DISABLED

	_editor_visual_instance = MeshInstance3D.new()
	_editor_visual_instance.name = "_JrOrbiter3DEditorVisuals"
	_editor_visual_instance.mesh = _editor_immediate_mesh
	_editor_visual_instance.material_override = _editor_visual_material
	_editor_visual_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_editor_visual_instance, false, Node.INTERNAL_MODE_BACK)
	_update_editor_visual_transform()


## Updates [member _editor_visual_instance]'s local transform so its origin aligns
## with the parent focus and its basis aligns with the driver's coordinate space.
func _update_editor_visual_transform() -> void:
	if _editor_visual_instance == null or not is_instance_valid(_editor_visual_instance):
		return
	_editor_visual_instance.transform = _get_focus_to_local_transform()


## Rebuilds the [ImmediateMesh] surfaces for the 3D orbital path and editor ghost.
func _redraw_editor_visuals() -> void:
	if not Engine.is_editor_hint() or not is_inside_tree():
		return
	if (
			_editor_immediate_mesh == null
			or _editor_visual_instance == null
			or not is_instance_valid(_editor_visual_instance)
	):
		_setup_editor_visual_nodes()
	_update_editor_visual_transform()
	_editor_immediate_mesh.clear_surfaces()

	if get_target_trajectory_element() == null:
		return

	if draw_orbit_in_editor and _cached_path_points.size() >= 2:
		_draw_orbit_path_surface()

	if show_editor_ghost and _has_ghost_position:
		_draw_ghost_surface()


## Appends a line-strip surface to [member _editor_immediate_mesh] tracing
## [member _cached_path_points].
func _draw_orbit_path_surface() -> void:
	_editor_immediate_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	_editor_immediate_mesh.surface_set_color(editor_orbit_color)
	for point: Vector3 in _cached_path_points:
		_editor_immediate_mesh.surface_add_vertex(point)
	_editor_immediate_mesh.surface_end()


## Appends a line surface to [member _editor_immediate_mesh] rendering the radius vector,
## wireframe sphere, and spin axis of the animated editor ghost.
func _draw_ghost_surface() -> void:
	_editor_immediate_mesh.surface_begin(Mesh.PRIMITIVE_LINES)

	var line_color := Color(
			editor_ghost_color.r,
			editor_ghost_color.g,
			editor_ghost_color.b,
			editor_ghost_color.a * 0.4
	)
	_editor_immediate_mesh.surface_set_color(line_color)
	_editor_immediate_mesh.surface_add_vertex(Vector3.ZERO)
	_editor_immediate_mesh.surface_add_vertex(_ghost_relative_position)

	_editor_immediate_mesh.surface_set_color(editor_ghost_color)
	var x_axis: Vector3 = _ghost_body_basis.x * editor_ghost_radius
	var y_axis: Vector3 = _ghost_body_basis.y * editor_ghost_radius
	var z_axis: Vector3 = _ghost_body_basis.z * editor_ghost_radius

	_add_ghost_great_circle(x_axis, z_axis)
	_add_ghost_great_circle(x_axis, y_axis)
	_add_ghost_great_circle(y_axis, z_axis)

	# Draw a polar spin-axis indicator protruding above the ghost's north (+Y) pole.
	_editor_immediate_mesh.surface_add_vertex(_ghost_relative_position - y_axis * 1.25)
	_editor_immediate_mesh.surface_add_vertex(_ghost_relative_position + y_axis * 1.5)

	_editor_immediate_mesh.surface_end()


## Adds a segmented great circle in the plane spanned by [param axis_a] and [param axis_b]
## around [member _ghost_relative_position].
func _add_ghost_great_circle(axis_a: Vector3, axis_b: Vector3) -> void:
	var step: float = TAU / float(GHOST_RING_SEGMENTS)
	var prev_point: Vector3 = _ghost_relative_position + axis_a
	for i: int in range(1, GHOST_RING_SEGMENTS + 1):
		var angle: float = float(i) * step
		var next_point: Vector3 = _ghost_relative_position + axis_a * cos(angle) + axis_b * sin(
				angle
		)
		_editor_immediate_mesh.surface_add_vertex(prev_point)
		_editor_immediate_mesh.surface_add_vertex(next_point)
		prev_point = next_point
