@tool
@icon("res://addons/jr_orrery/icons/orbiter_2d.svg")

## Visual 2D follower node ([Node2D]) driven by a [JrOrreryElement] ([JrRailOrbiter], [JrBallisticOrbiter], or [JrOrbitDriver]).
##
## Subscribes to [signal JrOrreryElement.transform_updated] on [member orbiter] and
## projects its calculated 3D orbital or ballistic transform onto the top-down 2D X-Z plane
## ([code]Vector2(x, z) * position_scale[/code]). Supports both nested parent-child
## visual hierarchies and flat sibling hierarchies, and renders the 2D trajectory
## path and animated preview ghost in the Godot Editor.
extends Node2D
class_name JrOrbiter2D

## Emitted when the orbiter's parameters, scale, or cached editor path change.
signal orbit_visuals_changed();

## Determines how this 2D node applies the transform calculated by [member orbiter].
enum TransformMode {
	## Automatically uses [constant LOCAL] when parented to another [JrOrbiter2D],
	## or [constant SYSTEM] otherwise.
	AUTO,
	## Applies the full system/perspective-adjusted transform relative to the root driver.
	SYSTEM,
	## Applies the parent-focus-relative local orbital transform.
	LOCAL
}

#region Private Members
## Cached polyline vertices representing the 2D projected orbital path relative to the parent focus.
var _cached_path_points: PackedVector2Array = [];

## Projected 2D position of the animated editor preview ghost relative to the parent focus.
var _ghost_relative_position: Vector2 = Vector2.ZERO;

## Whether [member _ghost_relative_position] currently holds a valid preview position.
var _has_ghost_position: bool = false;

## Initial local position captured when [member orbiter] is a [JrOrbitDriver].
var _anchor_position: Vector2 = Vector2.ZERO;

## Initial local rotation captured when [member orbiter] is a [JrOrbitDriver].
var _anchor_rotation: float = 0.0;

## Guard flag set while this script is updating [member Node2D.position] or [member Node2D.rotation] internally.
var _updating_transform_internally: bool = false;

## Flag indicating that the local transform was modified externally in the editor.
var _transform_modified_externally: bool = false;
#endregion

#region Exports
## The simulation [JrOrreryElement] ([JrOrbiter] or [JrOrbitDriver]) that drives
## this 2D visual node's transform.
@export
var orbiter: JrOrreryElement = null:
	set(value):
		if orbiter == value:
			return;
		_disconnect_orbiter_signals();
		orbiter = value;
		if is_node_ready():
			_connect_orbiter_signals();
			_on_orbiter_rebound();

## Controls whether this node applies the parent-relative local transform or the
## full system/perspective transform from [member orbiter].
@export
var transform_mode: TransformMode = TransformMode.AUTO:
	set(value):
		if transform_mode != value:
			transform_mode = value;
			_on_orbiter_transform_updated();
			queue_redraw();

## Multiplier applied when projecting 3D orbital coordinates ([code](x, z)[/code])
## into 2D pixel space ([code](x, y)[/code]).
@export_range(0.000001, 10000.0, 0.000001, "exp", "or_greater")
var position_scale: float = 1.0:
	set(value):
		var clamped: float = maxf(1e-9, value);
		if not is_equal_approx(position_scale, clamped):
			position_scale = clamped;
			_on_orbiter_transform_updated();
			_rebuild_orbit_path_cache();

## If [code]true[/code], synchronizes [member Node2D.rotation] with the 3D body's
## horizontal yaw/spin around the +Y axis.
@export
var sync_rotation: bool = false:
	set(value):
		if sync_rotation != value:
			sync_rotation = value;
			_on_orbiter_transform_updated();

@export_group("Editor Visualization")
## If [code]true[/code], renders the projected conic orbital path in the 2D editor viewport.
@export
var draw_orbit_in_editor: bool = true:
	set(value):
		if draw_orbit_in_editor != value:
			draw_orbit_in_editor = value;
			queue_redraw();

## Color of the orbital path line drawn in the 2D editor viewport.
@export
var editor_orbit_color: Color = Color(0.4, 0.7, 1.0, 0.85):
	set(value):
		if editor_orbit_color != value:
			editor_orbit_color = value;
			queue_redraw();
			orbit_visuals_changed.emit();

## Line width (in pixels) of the orbital path drawn in the 2D editor viewport.
@export_range(0.5, 20.0, 0.5)
var editor_orbit_width: float = 1.5:
	set(value):
		if not is_equal_approx(editor_orbit_width, value):
			editor_orbit_width = value;
			queue_redraw();
			orbit_visuals_changed.emit();

## Number of segments used to sample the orbital path in the 2D editor viewport.
@export_range(8, 512, 1)
var editor_orbit_segments: int = 64:
	set(value):
		var clamped: int = maxi(8, value);
		if editor_orbit_segments != clamped:
			editor_orbit_segments = clamped;
			_rebuild_orbit_path_cache();

## If [code]true[/code], renders an animated ghost marker along the orbital path
## in the 2D editor viewport to preview orbital motion without moving the node.
@export
var show_editor_ghost: bool = true:
	set(value):
		if show_editor_ghost != value:
			show_editor_ghost = value;
			if show_editor_ghost and Engine.is_editor_hint():
				_refresh_ghost_position();
			queue_redraw();

## Color of the animated ghost marker rendered in the 2D editor viewport.
@export
var editor_ghost_color: Color = Color(1.0, 1.0, 1.0, 0.65):
	set(value):
		if editor_ghost_color != value:
			editor_ghost_color = value;
			queue_redraw();

## Radius (in local units) of the animated ghost marker rendered in the 2D editor viewport.
@export_range(1.0, 64.0, 0.5)
var editor_ghost_radius: float = 6.0:
	set(value):
		if not is_equal_approx(editor_ghost_radius, value):
			editor_ghost_radius = value;
			queue_redraw();
#endregion

#region Lifecycle & Notifications
## Captures initial anchor state, connects to [member orbiter], synchronizes
## the initial transform, and builds the editor orbit path cache.
func _ready() -> void:
	_anchor_position = position;
	_anchor_rotation = rotation;
	_connect_orbiter_signals();
	set_notify_local_transform(Engine.is_editor_hint());
	_on_orbiter_transform_updated();
	_rebuild_orbit_path_cache();

## Handles hierarchy and local transform notifications to refresh editor warnings and redraws.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PARENTED or what == NOTIFICATION_UNPARENTED:
		update_configuration_warnings();
		if is_inside_tree():
			_on_orbiter_transform_updated();
			queue_redraw();
	elif what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED and Engine.is_editor_hint():
		_on_editor_local_transform_changed();
		queue_redraw();

## Draws the cached trajectory path and animated editor ghost centered on the parent
## focus when running in the Godot Editor.
func _draw() -> void:
	if not Engine.is_editor_hint() or get_target_trajectory_element() == null:
		return;

	var focus_to_local: Transform2D = _get_focus_to_local_transform();
	draw_set_transform_matrix(focus_to_local);

	if draw_orbit_in_editor and _cached_path_points.size() >= 2:
		draw_polyline(_cached_path_points, editor_orbit_color, editor_orbit_width, true);

	if show_editor_ghost and _has_ghost_position:
		var line_color: Color = Color(
			editor_ghost_color.r,
			editor_ghost_color.g,
			editor_ghost_color.b,
			editor_ghost_color.a * 0.4
		);
		var fill_color: Color = Color(
			editor_ghost_color.r,
			editor_ghost_color.g,
			editor_ghost_color.b,
			editor_ghost_color.a * 0.45
		);
		draw_line(Vector2.ZERO, _ghost_relative_position, line_color, 1.0, true);
		draw_circle(_ghost_relative_position, editor_ghost_radius, fill_color);
		draw_arc(_ghost_relative_position, editor_ghost_radius, 0.0, TAU, 24, editor_ghost_color, 1.5, true);

	draw_set_transform_matrix(Transform2D.IDENTITY);

## Validates that [member orbiter] is assigned and valid, and checks for external
## transform edits in the Godot Editor.
func _get_configuration_warnings() -> PackedStringArray:
	var warnings: PackedStringArray = [];

	if orbiter == null:
		warnings.append("An orbiter (JrRailOrbiter, JrBallisticOrbiter, or JrOrbitDriver) node must be assigned.");
	elif orbiter is JrRailOrbiter and not (orbiter as JrRailOrbiter).is_orbit_valid():
		warnings.append("The assigned JrRailOrbiter's orbital parameters do not form a valid conic section.");

	if get_target_trajectory_element() != null and (_transform_modified_externally or _is_transform_out_of_sync()):
		warnings.append("Transform was modified outside of script. Local transform is controlled by the assigned simulation node.");

	return warnings;
#endregion

#region Public Helpers
## Returns [member orbiter] cast to [JrRailOrbiter], or [code]null[/code] if unassigned
## or if [member orbiter] is another element type.
func get_target_orbiter() -> JrRailOrbiter:
	return orbiter as JrRailOrbiter;

## Returns [member orbiter] when it supports Keplerian conic gizmo handles
## ([JrRailOrbiter] or [JrBallisticOrbiter] in [constant JrBallisticOrbiter.InitialStateMode.KEPLERIAN_ELEMENTS] mode).
func get_target_keplerian_element() -> JrOrreryElement:
	if orbiter != null and orbiter.has_keplerian_gizmo():
		return orbiter;
	return null;

## Returns [member orbiter] cast to [JrBallisticOrbiter], or [code]null[/code] if unassigned
## or if [member orbiter] is another element type.
func get_target_ballistic() -> JrBallisticOrbiter:
	return orbiter as JrBallisticOrbiter;

## Returns [member orbiter] when it is a trajectory-bearing simulation element
## ([JrRailOrbiter] with valid conic parameters or [JrBallisticOrbiter]), or [code]null[/code].
func get_target_trajectory_element() -> JrOrreryElement:
	if orbiter == null or orbiter is JrOrbitDriver:
		return null;
	if not orbiter.is_trajectory_valid():
		return null;
	return orbiter;

## Returns the active [JrReferenceDirection] from [member orbiter], or [code]null[/code].
func get_reference_plane() -> JrReferenceDirection:
	if orbiter == null:
		return null;
	return orbiter.get_reference_plane();

## Returns [code]true[/code] if [member orbiter] is actively stepping simulation time.
func is_simulation_playing() -> bool:
	if orbiter == null:
		return false;
	return orbiter.is_simulation_playing();

## Returns [code]true[/code] when this node is applying parent-relative local transforms
## rather than full system transforms.
func is_using_local_transform() -> bool:
	if transform_mode == TransformMode.LOCAL:
		return true;
	if transform_mode == TransformMode.SYSTEM:
		return false;
	return get_parent() is JrOrbiter2D;

## Projects a 3D orbital vector onto 2D top-down X-Z coordinates scaled by [member position_scale].
func project_3d_to_2d(vec_3d: Vector3) -> Vector2:
	return Vector2(vec_3d.x, vec_3d.z) * position_scale;

## Unprojects a 2D focus-relative vector back into 3D X-Z coordinates divided by [member position_scale].
func unproject_2d_to_3d(vec_2d: Vector2) -> Vector3:
	var inv_scale: float = 1.0 / maxf(1e-9, position_scale);
	return Vector3(vec_2d.x * inv_scale, 0.0, vec_2d.y * inv_scale);

## Returns the [Transform2D] that maps scaled 2D focus-relative coordinates into
## global 2D canvas coordinates.
func get_focus_to_world_transform() -> Transform2D:
	return global_transform * _get_focus_to_local_transform();
#endregion

#region Signal Connections & Transform Sync
## Connects reactive signals on [member orbiter].
func _connect_orbiter_signals() -> void:
	if orbiter == null:
		return;
	if not orbiter.transform_updated.is_connected(_on_orbiter_transform_updated):
		orbiter.transform_updated.connect(_on_orbiter_transform_updated);
	if not orbiter.editor_ghost_updated.is_connected(_on_orbiter_ghost_updated):
		orbiter.editor_ghost_updated.connect(_on_orbiter_ghost_updated);
	if not orbiter.orbit_visuals_changed.is_connected(_on_orbiter_visuals_changed):
		orbiter.orbit_visuals_changed.connect(_on_orbiter_visuals_changed);

## Disconnects reactive signals from [member orbiter].
func _disconnect_orbiter_signals() -> void:
	if orbiter == null:
		return;
	if orbiter.transform_updated.is_connected(_on_orbiter_transform_updated):
		orbiter.transform_updated.disconnect(_on_orbiter_transform_updated);
	if orbiter.editor_ghost_updated.is_connected(_on_orbiter_ghost_updated):
		orbiter.editor_ghost_updated.disconnect(_on_orbiter_ghost_updated);
	if orbiter.orbit_visuals_changed.is_connected(_on_orbiter_visuals_changed):
		orbiter.orbit_visuals_changed.disconnect(_on_orbiter_visuals_changed);

## Handles rebinding when [member orbiter] is reassigned.
func _on_orbiter_rebound() -> void:
	if is_inside_tree():
		_on_orbiter_transform_updated();
	update_configuration_warnings();
	_rebuild_orbit_path_cache();

## Updates this node's 2D transform from [member orbiter]'s calculated 3D transform.
func _on_orbiter_transform_updated() -> void:
	if orbiter == null or not is_inside_tree():
		return;

	if orbiter is JrOrbitDriver:
		_apply_driver_follower_transform();
		return;

	var target_element: JrOrreryElement = get_target_trajectory_element();
	if target_element == null:
		return;

	var target_pos: Vector2 = _compute_expected_2d_position(target_element);
	var target_rot: float = _compute_expected_2d_rotation(target_element);
	_apply_orbital_transform_2d(target_pos, target_rot);

## Applies the perspective-shifted root transform when [member orbiter] is a [JrOrbitDriver].
func _apply_driver_follower_transform() -> void:
	if Engine.is_editor_hint():
		_anchor_position = position;
		_anchor_rotation = rotation;
		return;

	var sys_xf: Transform3D = orbiter.get_system_transform();
	var offset_2d: Vector2 = project_3d_to_2d(sys_xf.origin).rotated(_anchor_rotation);
	var rot_2d: float = _extract_yaw_2d(sys_xf.basis);
	_updating_transform_internally = true;
	rotation = _anchor_rotation + rot_2d;
	position = _anchor_position + offset_2d;
	_updating_transform_internally = false;

## Computes the expected local 2D position for [param target_element] according
## to [method is_using_local_transform].
func _compute_expected_2d_position(target_element: JrOrreryElement) -> Vector2:
	if is_using_local_transform():
		var focus_pos_2d: Vector2 = project_3d_to_2d(target_element.get_focus_relative_position());
		if get_parent() is JrOrbiter2D:
			var parent_orb_2d: JrOrbiter2D = get_parent() as JrOrbiter2D;
			if parent_orb_2d.orbiter is not JrOrbitDriver and not is_zero_approx(parent_orb_2d.rotation):
				return focus_pos_2d.rotated(-parent_orb_2d.rotation);
		return focus_pos_2d;
	return project_3d_to_2d(target_element.get_system_transform().origin);

## Computes the expected local 2D rotation for [param target_element].
func _compute_expected_2d_rotation(target_element: JrOrreryElement) -> float:
	if not sync_rotation:
		return 0.0;
	if is_using_local_transform():
		var local_yaw: float = _extract_yaw_2d(target_element.get_driver_relative_basis());
		if get_parent() is Node2D:
			return local_yaw - (get_parent() as Node2D).rotation;
		return local_yaw;
	return _extract_yaw_2d(target_element.get_system_transform().basis);

## Extracts the 2D top-down yaw angle (in radians) from a 3D [param basis_3d].
static func _extract_yaw_2d(basis_3d: Basis) -> float:
	return atan2(basis_3d.x.z, basis_3d.x.x);

## Sets [member Node2D.position] and [member Node2D.rotation] internally without
## triggering an external-edit warning.
func _apply_orbital_transform_2d(new_position: Vector2, new_rotation: float) -> void:
	_updating_transform_internally = true;
	position = new_position;
	rotation = new_rotation;
	if Engine.is_editor_hint():
		scale = Vector2.ONE;
		skew = 0.0;
	if _transform_modified_externally:
		_transform_modified_externally = false;
		update_configuration_warnings();
	_updating_transform_internally = false;
	if Engine.is_editor_hint():
		queue_redraw();

## Returns the [Transform2D] that maps scaled 2D focus-relative coordinates into
## this node's local drawing space.
func _get_focus_to_local_transform() -> Transform2D:
	var target_element: JrOrreryElement = get_target_trajectory_element();
	if target_element == null:
		return Transform2D.IDENTITY;
	var focus_offset_2d: Vector2 = project_3d_to_2d(target_element.get_focus_relative_position());
	if is_using_local_transform():
		return transform.affine_inverse();
	var focus_in_parent: Transform2D = Transform2D(0.0, position - focus_offset_2d);
	return transform.affine_inverse() * focus_in_parent;
#endregion

#region Editor Ghost, Path Cache & Transform Guard
## Updates the projected 2D ghost position when [member orbiter] emits [signal JrOrreryElement.editor_ghost_updated].
func _on_orbiter_ghost_updated(_preview_time: float) -> void:
	if not Engine.is_editor_hint() or not show_editor_ghost:
		return;
	_refresh_ghost_position();
	queue_redraw();

## Rebuilds the 2D orbit path cache when [member orbiter] emits [signal JrOrreryElement.orbit_visuals_changed].
func _on_orbiter_visuals_changed() -> void:
	_on_orbiter_transform_updated();
	update_configuration_warnings();
	_rebuild_orbit_path_cache();

## Updates [member _ghost_relative_position] from [member orbiter]'s 3D ghost position.
func _refresh_ghost_position() -> void:
	var target_element: JrOrreryElement = get_target_trajectory_element();
	if target_element == null or not target_element.has_ghost_position():
		_has_ghost_position = false;
		return;
	_ghost_relative_position = project_3d_to_2d(target_element.get_ghost_relative_position());
	_has_ghost_position = true;

## Recomputes [member _cached_path_points] for 2D editor visualization and schedules a redraw.
func _rebuild_orbit_path_cache() -> void:
	if not Engine.is_editor_hint() and not draw_orbit_in_editor:
		_cached_path_points = [];
		_has_ghost_position = false;
		return;
	var target_element: JrOrreryElement = get_target_trajectory_element();
	if target_element == null or target_element.get_reference_plane() == null:
		_cached_path_points = [];
		_has_ghost_position = false;
		queue_redraw();
		orbit_visuals_changed.emit();
		return;

	_cached_path_points = target_element.get_trajectory_path_points_2d(
		editor_orbit_segments,
		position_scale
	);
	_refresh_ghost_position();
	queue_redraw();
	orbit_visuals_changed.emit();

## Detects when the user modifies a simulation-driven node's transform directly
## in the editor, emits a warning, and restores the calculated transform.
func _on_editor_local_transform_changed() -> void:
	if _updating_transform_internally or not is_inside_tree():
		return;
	if orbiter is JrOrbitDriver:
		_anchor_position = position;
		_anchor_rotation = rotation;
		return;
	if _is_transform_out_of_sync():
		push_warning(
			"JrOrbiter2D '%s': Transform was modified outside of script. Local transform is controlled by the assigned simulation node." % name
		);
		_on_orbiter_transform_updated();

## Returns [code]true[/code] if the node's current local transform deviates from
## the script-driven orbital or ballistic position or rotation.
func _is_transform_out_of_sync() -> bool:
	var target_element: JrOrreryElement = get_target_trajectory_element();
	if target_element == null:
		return false;
	if not scale.is_equal_approx(Vector2.ONE) or not is_zero_approx(skew):
		return true;
	var expected_pos: Vector2 = _compute_expected_2d_position(target_element);
	var expected_rot: float = _compute_expected_2d_rotation(target_element);
	if position.distance_squared_to(expected_pos) > 0.01:
		return true;
	if not is_equal_approx(rotation, expected_rot):
		return true;
	return false;
#endregion
