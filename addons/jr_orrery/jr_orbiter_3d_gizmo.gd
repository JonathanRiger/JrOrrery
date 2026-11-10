@tool
class_name JrOrbiter3DGizmo
extends RefCounted
## Interactive 3D viewport gizmo and grabber controller for [JrOrbiter3D], [JrRailOrbiter],
## and [JrBallisticOrbiter] (when in
## [constant JrBallisticOrbiter.InitialStateMode.KEPLERIAN_ELEMENTS] mode).
##
## Renders 3D-projected guide lines, reference arcs, and grabber handles over the
## 3D editor viewport, and intercepts mouse input to interactively adjust conic
## parameters ([member JrRailOrbiter.periapsis], [member JrRailOrbiter.apoapsis],
## [member JrRailOrbiter.eccentricity], [member JrRailOrbiter.argument_of_periapsis],
## and [member JrRailOrbiter.mean_anomaly_at_epoch]), orbital plane orientation
## ([member JrRailOrbiter.inclination] and [member JrRailOrbiter.longitude_of_the_ascending_node]),
## and body spin orientation ([member JrRailOrbiter.axial_tilt],
## [member JrRailOrbiter.axial_tilt_longitude], and [member JrRailOrbiter.rotation_at_epoch])
## directly on the target simulation node.

## Identifies which 3D orbital or rotational grabber handle is hovered or being dragged.
enum GrabberType {
	NONE,
	PERIAPSIS,
	APOAPSIS,
	SEMI_PARAMETER,
	MEAN_ANOMALY_AT_EPOCH,
	ASCENDING_NODE,
	INCLINATION,
	AXIAL_TILT,
	AXIAL_TILT_LONGITUDE,
	ROTATION_AT_EPOCH,
}

## Visual radius (in screen pixels) of each 3D viewport grabber handle.
const HANDLE_DRAW_RADIUS: float = 6.0

## Hit-test radius (in screen pixels) for clicking or hovering a 3D viewport grabber handle.
const HANDLE_HIT_RADIUS: float = 10.0

## Minimum apparent screen-space radius (in pixels) for the body-centered orientation gizmo.
const MIN_BODY_GIZMO_SCREEN_PX: float = 42.0

## Number of line segments used when rendering a full 360-degree 3D reference circle.
const CIRCLE_SEGMENTS: int = 36

## Number of line segments used when rendering a 3D angular sweep arc.
const ARC_SEGMENTS: int = 20

## Default orbit path color when editing a standalone [JrRailOrbiter] without a [JrOrbiter3D]
## follower.
const STANDALONE_ORBIT_COLOR := Color(0.4, 0.7, 1.0, 0.85)

## Fill color for the periapsis grabber handle.
const PERIAPSIS_HANDLE_COLOR := Color(1.0, 0.65, 0.2, 1.0)

## Fill color for the apoapsis grabber handle.
const APOAPSIS_HANDLE_COLOR := Color(0.25, 0.8, 1.0, 1.0)

## Fill color for the semi-parameter grabber handle.
const SEMI_PARAMETER_HANDLE_COLOR := Color(0.35, 0.95, 0.55, 1.0)

## Fill color for the mean anomaly at epoch grabber handle.
const EPOCH_ANOMALY_HANDLE_COLOR := Color(0.95, 0.4, 0.85, 1.0)

## Fill color for the right ascension of the ascending node (RAAN) grabber handle.
const ASCENDING_NODE_HANDLE_COLOR := Color(1.0, 0.9, 0.25, 1.0)

## Fill color for the orbital inclination grabber handle.
const INCLINATION_HANDLE_COLOR := Color(0.3, 0.95, 0.9, 1.0)

## Fill color for the orbiter axial tilt grabber handle.
const AXIAL_TILT_HANDLE_COLOR := Color(1.0, 0.35, 0.4, 1.0)

## Fill color for the orbiter axial tilt longitude grabber handle.
const AXIAL_TILT_LONGITUDE_HANDLE_COLOR := Color(0.65, 0.5, 1.0, 1.0)

## Fill color for the starting day angle (rotation at epoch) grabber handle.
const ROTATION_AT_EPOCH_HANDLE_COLOR := Color(0.55, 0.95, 0.3, 1.0)

## Owning [EditorPlugin] used to request overlay redraws and commit Undo/Redo actions.
var _plugin: EditorPlugin = null

## The [JrOrbiter3D] visual follower currently selected or linked in the 3D viewport (optional).
var _edited_orbiter: JrOrbiter3D = null

## The [JrOrreryElement] simulation node currently being edited in the 3D viewport.
var _edited_sim_orbiter: JrOrreryElement = null

## Most recently active 3D viewport [Camera3D].
var _last_camera: Camera3D = null

## Cached global transform of [member _last_camera] used to detect camera motion.
var _last_camera_transform := Transform3D.IDENTITY

## Cached vertical field of view or orthogonal size of [member _last_camera].
var _last_camera_projection_param: float = 0.0

## Active overlay [Control] during the current draw pass.
var _current_overlay: Control = null

## Active [Camera3D] during the current draw or input pass.
var _current_camera: Camera3D = null

## Active focus-to-world [Transform3D] during the current draw or input pass.
var _current_focus_to_world := Transform3D.IDENTITY

## The grabber handle currently being dragged, or [constant GrabberType.NONE].
var _active_grabber := GrabberType.NONE

## The grabber handle currently hovered by the cursor.
var _hovered_grabber := GrabberType.NONE

## Distance of the stationary opposite apsis recorded during an apsis drag operation.
var _drag_opposite_apsis_distance: float = 1.0

## Snapshot of orbital element properties recorded at the start of a drag for Undo/Redo.
var _undo_snapshot: Dictionary[StringName, float] = {}


## Initializes the 3D gizmo controller with a reference to the owning [param plugin].
func _init(plugin: EditorPlugin = null) -> void:
	_plugin = plugin


## Returns the owning [EditorPlugin] reference.
func get_plugin() -> EditorPlugin:
	return _plugin


## Updates the owning [param plugin] reference.
func set_plugin(plugin: EditorPlugin) -> void:
	_plugin = plugin


## Returns the currently edited [JrOrbiter3D], or [code]null[/code] if none is active.
func get_edited_orbiter() -> JrOrbiter3D:
	return _edited_orbiter


## Returns the currently edited Keplerian [JrOrreryElement] simulation node, or [code]null[/code].
func get_edited_sim_orbiter() -> JrOrreryElement:
	return _get_target_orbiter()


## Updates the currently edited [param orbiter] visual node and manages signal connections.
func set_edited_orbiter(orbiter: JrOrbiter3D) -> void:
	var sim: JrOrreryElement = orbiter.get_target_keplerian_element() if orbiter != null else null
	set_edited_target(sim, orbiter)


## Updates the currently edited [param sim_orbiter] and optional [param visual_orbiter]
## follower node, managing signal connections for 3D overlay redraws.
func set_edited_target(sim_orbiter: JrOrreryElement, visual_orbiter: JrOrbiter3D = null) -> void:
	if _edited_orbiter == visual_orbiter and _edited_sim_orbiter == sim_orbiter:
		return

	if _edited_orbiter != null and is_instance_valid(_edited_orbiter):
		if _edited_orbiter.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_orbiter.orbit_visuals_changed.disconnect(_request_redraw)
	if _edited_sim_orbiter != null and is_instance_valid(_edited_sim_orbiter):
		if _edited_sim_orbiter.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_sim_orbiter.orbit_visuals_changed.disconnect(_request_redraw)

	_edited_orbiter = visual_orbiter
	_edited_sim_orbiter = sim_orbiter
	_active_grabber = GrabberType.NONE
	_hovered_grabber = GrabberType.NONE

	if _edited_orbiter != null and is_instance_valid(_edited_orbiter):
		if not _edited_orbiter.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_orbiter.orbit_visuals_changed.connect(_request_redraw)
	if _edited_sim_orbiter != null and is_instance_valid(_edited_sim_orbiter):
		if not _edited_sim_orbiter.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_sim_orbiter.orbit_visuals_changed.connect(_request_redraw)

	if _edited_orbiter != null or _edited_sim_orbiter != null:
		_ensure_camera_initialized()

	_request_redraw()


## Returns [code]true[/code] if a valid Keplerian [JrOrreryElement] node and [JrReferenceDirection]
## are currently bound for 3D viewport editing.
func can_edit_orbiter() -> bool:
	if (
			_edited_orbiter != null
			and (not is_instance_valid(_edited_orbiter) or not _edited_orbiter.is_inside_tree())
	):
		return false
	var target_orbiter: JrOrreryElement = _get_target_orbiter()
	if (
			target_orbiter == null
			or not is_instance_valid(target_orbiter)
			or not target_orbiter.is_inside_tree()
	):
		return false
	if not target_orbiter.has_keplerian_gizmo() or not target_orbiter.is_orbit_valid():
		return false
	return target_orbiter.get_reference_plane() != null


## Checks whether the active 3D viewport camera has moved since the last frame
## and schedules an overlay redraw if needed.
func poll_camera_changes() -> void:
	if not can_edit_orbiter():
		return
	_ensure_camera_initialized()
	if _last_camera == null or not is_instance_valid(_last_camera):
		return

	var proj_param: float = (
			_last_camera.size
			if _last_camera.projection == Camera3D.PROJECTION_ORTHOGONAL
			else _last_camera.fov
	)
	if (
			_last_camera.global_transform != _last_camera_transform
			or not is_equal_approx(proj_param, _last_camera_projection_param)
	):
		_last_camera_transform = _last_camera.global_transform
		_last_camera_projection_param = proj_param
		_request_redraw()


## Draws all 3D orbital and rotational gizmo guides and grabber handles over [param overlay].
func draw_over_viewport(overlay: Control) -> void:
	if not can_edit_orbiter() or overlay == null:
		return

	var camera: Camera3D = _resolve_camera_for_overlay(overlay)
	if camera == null:
		return

	_current_overlay = overlay
	_current_camera = camera
	_current_focus_to_world = _compute_focus_to_world_transform()

	if _edited_orbiter == null:
		_draw_standalone_orbit_curve()

	var body_radius: float = _compute_body_gizmo_radius(camera, overlay.size)
	_draw_focus_orbital_guides()
	_draw_conic_guides()
	if not _is_simulation_playing() and _get_target_orbiter() is JrRailOrbiter:
		_draw_body_orientation_guides(body_radius)

	_draw_all_grabber_handles(body_radius)
	_draw_active_or_hovered_label(body_radius)


## Processes 3D viewport mouse input for hovering and dragging grabber handles.
## Returns [code]true[/code] if the event was consumed by a grabber interaction.
func handle_gui_input(viewport_camera: Camera3D, event: InputEvent) -> bool:
	if viewport_camera != null:
		_last_camera = viewport_camera
	if not can_edit_orbiter() or viewport_camera == null:
		return false

	_current_camera = viewport_camera
	_current_focus_to_world = _compute_focus_to_world_transform()

	if event is InputEventMouseButton:
		return _handle_mouse_button(event as InputEventMouseButton)
	elif event is InputEventMouseMotion:
		return _handle_mouse_motion(event as InputEventMouseMotion)

	return false


## Returns the active Keplerian [JrOrreryElement] being edited.
func _get_target_orbiter() -> JrOrreryElement:
	if _edited_orbiter != null and is_instance_valid(_edited_orbiter):
		var target: JrOrreryElement = _edited_orbiter.get_target_keplerian_element()
		if target != null:
			return target
	if (
			_edited_sim_orbiter != null
			and is_instance_valid(_edited_sim_orbiter)
			and _edited_sim_orbiter.has_keplerian_gizmo()
	):
		return _edited_sim_orbiter
	return null


## Returns the active [JrReferenceDirection] for the edited orbiter.
func _get_reference_plane() -> JrReferenceDirection:
	var target: JrOrreryElement = _get_target_orbiter()
	return target.get_reference_plane() if target != null else null


## Returns the active position scale factor.
func _get_position_scale() -> float:
	if _edited_orbiter != null and is_instance_valid(_edited_orbiter):
		return _edited_orbiter.position_scale
	return 1.0


## Returns the active editor ghost radius for body gizmo sizing.
func _get_ghost_radius() -> float:
	if _edited_orbiter != null and is_instance_valid(_edited_orbiter):
		return _edited_orbiter.editor_ghost_radius
	return 0.5


## Returns [code]true[/code] if the edited orbiter's driver is actively stepping time.
func _is_simulation_playing() -> bool:
	var target: JrOrreryElement = _get_target_orbiter()
	return target.is_simulation_playing() if target != null else false


## Computes the [Transform3D] mapping scaled driver-focus coordinates into 3D world space.
func _compute_focus_to_world_transform() -> Transform3D:
	if (
			_edited_orbiter != null
			and is_instance_valid(_edited_orbiter)
			and _edited_orbiter.is_inside_tree()
	):
		return _edited_orbiter.get_focus_to_world_transform()
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Transform3D.IDENTITY
	var focus_origin: Vector3 = (
			target.get_driver_relative_transform().origin
			- target.get_focus_relative_position()
	)
	return Transform3D(Basis.IDENTITY, focus_origin)


## Returns the scaled ascending node gizmo position in driver-focus space.
func _get_ascending_node_gizmo_position() -> Vector3:
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Vector3.ZERO
	return target.get_ascending_node_gizmo_position() * _get_position_scale()


## Returns the scaled inclination gizmo position in driver-focus space.
func _get_inclination_gizmo_position() -> Vector3:
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Vector3.ZERO
	return target.get_inclination_gizmo_position() * _get_position_scale()


## Returns the scaled axial tilt gizmo position in driver-focus space.
func _get_axial_tilt_gizmo_position(body_radius: float) -> Vector3:
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target.get_relative_position(0.0) * _get_position_scale()
	var north_pole: Vector3 = target.get_tilted_body_basis().y
	return body_pos + north_pole * (body_radius * JrRailOrbiter.AXIAL_TILT_GIZMO_RADIUS_SCALE)


## Returns the scaled axial tilt longitude gizmo position in driver-focus space.
func _get_axial_tilt_longitude_gizmo_position(body_radius: float) -> Vector3:
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target.get_relative_position(0.0) * _get_position_scale()
	var tilt_dir: Vector3 = target.get_axial_tilt_direction()
	return body_pos + tilt_dir * body_radius


## Returns the scaled starting day angle gizmo position in driver-focus space.
func _get_starting_day_angle_gizmo_position(body_radius: float) -> Vector3:
	var target: JrOrreryElement = _get_target_orbiter()
	if target == null:
		return Vector3.ZERO
	var body_pos: Vector3 = target.get_relative_position(0.0) * _get_position_scale()
	var prime_meridian: Vector3 = -target.get_body_basis(0.0).z
	return (
			body_pos
			+ prime_meridian * (body_radius * JrRailOrbiter.STARTING_DAY_GIZMO_RADIUS_SCALE)
	)


## Requests an overlay update on the owning [EditorPlugin].
func _request_redraw() -> void:
	if _plugin != null:
		_plugin.update_overlays()


## Initializes [member _last_camera] from the primary 3D editor viewport if not yet set.
func _ensure_camera_initialized() -> void:
	if _last_camera != null and is_instance_valid(_last_camera):
		return
	if not Engine.is_editor_hint():
		return
	var sub_vp: SubViewport = EditorInterface.get_editor_viewport_3d(0)
	if sub_vp != null:
		_last_camera = sub_vp.get_camera_3d()


## Resolves the [Camera3D] belonging to the 3D editor viewport that owns [param overlay].
func _resolve_camera_for_overlay(overlay: Control) -> Camera3D:
	if Engine.is_editor_hint():
		var overlay_parent: Node = overlay.get_parent()
		if overlay_parent != null:
			for i: int in range(4):
				var sub_vp: SubViewport = EditorInterface.get_editor_viewport_3d(i)
				if sub_vp != null and overlay_parent.is_ancestor_of(sub_vp):
					return sub_vp.get_camera_3d()
	if _last_camera != null and is_instance_valid(_last_camera):
		return _last_camera
	if Engine.is_editor_hint():
		var primary_vp: SubViewport = EditorInterface.get_editor_viewport_3d(0)
		return primary_vp.get_camera_3d() if primary_vp != null else null
	return null


## Computes the driver-focus space radius for the body-centered orientation gizmos,
## ensuring the handles stay outside the body's ghost radius and maintain at least
## [constant MIN_BODY_GIZMO_SCREEN_PX] pixels of separation on screen.
func _compute_body_gizmo_radius(camera: Camera3D, viewport_size: Vector2) -> float:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var pos_scale: float = _get_position_scale()
	var body_focus_pos: Vector3 = elements.get_relative_position(0.0, ref) * pos_scale
	var body_world_pos: Vector3 = _current_focus_to_world * body_focus_pos

	var vp_height: float = maxf(1.0, viewport_size.y)
	var world_units_per_px: float = 0.01
	if camera.projection == Camera3D.PROJECTION_ORTHOGONAL:
		world_units_per_px = camera.size / vp_height
	else:
		var cam_forward: Vector3 = -camera.global_transform.basis.z
		var depth: float = maxf(0.1, (body_world_pos - camera.global_position).dot(cam_forward))
		world_units_per_px = (2.0 * depth * tan(deg_to_rad(camera.fov) * 0.5)) / vp_height

	var focus_scale: float = maxf(
			0.0001,
			_current_focus_to_world.basis.get_scale().length() / sqrt(3.0)
	)
	var min_focus_radius: float = (MIN_BODY_GIZMO_SCREEN_PX * world_units_per_px) / focus_scale
	var pe_dist: float = elements.get_keplerian_prop(&"periapsis")
	var base_radius: float = maxf(_get_ghost_radius() * 2.2, pe_dist * pos_scale * 0.12)
	return maxf(base_radius, min_focus_radius)


## Returns [code]true[/code] if [param focus_pos] (in driver-focus space) lies in
## front of [member _current_camera]'s near clip plane.
func _is_focus_pos_visible(focus_pos: Vector3) -> bool:
	var world_pos: Vector3 = _current_focus_to_world * focus_pos
	var cam_forward: Vector3 = -_current_camera.global_transform.basis.z
	var min_dist: float = maxf(0.05, _current_camera.near)
	return (world_pos - _current_camera.global_position).dot(cam_forward) >= min_dist


## Projects a 3D point in driver-focus space onto 2D viewport screen coordinates.
func _project_focus_pos(focus_pos: Vector3) -> Vector2:
	return _current_camera.unproject_position(_current_focus_to_world * focus_pos)


## Draws a 3D line segment between [param start_focus] and [param end_focus]
## (in driver-focus space), clipped to the camera's near plane.
func _draw_line_3d(
		start_focus: Vector3,
		end_focus: Vector3,
		color: Color,
		width: float = 1.0
) -> void:
	var a: Vector3 = _current_focus_to_world * start_focus
	var b: Vector3 = _current_focus_to_world * end_focus
	var cam_origin: Vector3 = _current_camera.global_position
	var cam_forward: Vector3 = -_current_camera.global_transform.basis.z
	var min_dist: float = maxf(0.05, _current_camera.near)

	var da: float = (a - cam_origin).dot(cam_forward)
	var db: float = (b - cam_origin).dot(cam_forward)
	if da < min_dist and db < min_dist:
		return
	if da < min_dist:
		a = a.lerp(b, (min_dist - da) / (db - da))
	elif db < min_dist:
		b = b.lerp(a, (min_dist - db) / (da - db))

	var screen_a: Vector2 = _current_camera.unproject_position(a)
	var screen_b: Vector2 = _current_camera.unproject_position(b)
	_current_overlay.draw_line(screen_a, screen_b, color, width, true)


## Draws the 3D orbital curve on the viewport overlay when editing a standalone
## simulation node that does not have a [JrOrbiter3D] follower in the scene.
func _draw_standalone_orbit_curve() -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	if elements == null or ref == null:
		return
	var pts: PackedVector3Array = elements.get_orbit_path_points(ref, 64)
	for i: int in range(pts.size() - 1):
		_draw_line_3d(pts[i], pts[i + 1], STANDALONE_ORBIT_COLOR, 1.5)


## Draws a 3D circular arc defined by [param arc_frame] (where [code]origin[/code] is the
## center, [code]basis.x[/code] is the radius-scaled 0-radian vector, and [code]basis.y[/code]
## is the radius-scaled +90-degree vector) sweeping from [code]0.0[/code] to [param sweep_angle].
func _draw_arc_3d(
		arc_frame: Transform3D,
		sweep_angle: float,
		color: Color,
		width: float = 1.5
) -> void:
	if is_zero_approx(sweep_angle):
		return
	var step: float = sweep_angle / float(ARC_SEGMENTS)
	var prev_pt: Vector3 = arc_frame.origin + arc_frame.basis.x
	for i: int in range(1, ARC_SEGMENTS + 1):
		var theta: float = float(i) * step
		var next_pt: Vector3 = (
				arc_frame.origin
				+ arc_frame.basis.x * cos(theta)
				+ arc_frame.basis.y * sin(theta)
		)
		_draw_line_3d(prev_pt, next_pt, color, width)
		prev_pt = next_pt


## Draws a full 360-degree 3D reference circle defined by [param circle_frame]
## (where [code]origin[/code] is the center and [code]basis.x[/code], [code]basis.y[/code]
## are orthogonal radius-scaled axes in the circle's plane).
func _draw_circle_3d(circle_frame: Transform3D, color: Color) -> void:
	var step: float = TAU / float(CIRCLE_SEGMENTS)
	var prev_pt: Vector3 = circle_frame.origin + circle_frame.basis.x
	for i: int in range(1, CIRCLE_SEGMENTS + 1):
		var theta: float = float(i) * step
		var next_pt: Vector3 = (
				circle_frame.origin
				+ circle_frame.basis.x * cos(theta)
				+ circle_frame.basis.y * sin(theta)
		)
		_draw_line_3d(prev_pt, next_pt, color, 1.0)
		prev_pt = next_pt


## Draws the ascending node (RAAN) and orbital inclination guide lines and arcs around the focus.
func _draw_focus_orbital_guides() -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	if elements.is_locked_to_ecliptic():
		return
	var ref: JrReferenceDirection = _get_reference_plane()
	var r_plane: float = elements.get_orbital_plane_gizmo_radius() * _get_position_scale()

	var lan_color: Color = ASCENDING_NODE_HANDLE_COLOR.darkened(0.2)
	var lan_faint := Color(lan_color.r, lan_color.g, lan_color.b, 0.45)
	var vernal_vec: Vector3 = ref.vernal_point * r_plane
	var prog_vec: Vector3 = ref.progression_axis * r_plane
	var lan_pos: Vector3 = _get_ascending_node_gizmo_position()

	_draw_line_3d(Vector3.ZERO, vernal_vec, lan_faint, 1.0)
	_draw_line_3d(Vector3.ZERO, lan_pos, lan_color, 1.5)
	var lan_arc_frame := Transform3D(Basis(vernal_vec, prog_vec, Vector3.ZERO), Vector3.ZERO)
	var lan_val: float = elements.get_keplerian_prop(&"longitude_of_the_ascending_node")
	_draw_arc_3d(lan_arc_frame, lan_val, lan_color, 1.5)

	var inc_color: Color = INCLINATION_HANDLE_COLOR.darkened(0.2)
	var inc_faint := Color(inc_color.r, inc_color.g, inc_color.b, 0.45)
	var nodal_ecl: Basis = elements.get_nodal_ecliptic_basis(ref)
	var ecl_up_vec: Vector3 = nodal_ecl.y * r_plane
	var ecl_tilt_vec: Vector3 = -nodal_ecl.x * r_plane
	var inc_pos: Vector3 = _get_inclination_gizmo_position()

	_draw_line_3d(Vector3.ZERO, ecl_up_vec, inc_faint, 1.0)
	_draw_line_3d(Vector3.ZERO, inc_pos, inc_color, 1.5)
	var inc_arc_frame := Transform3D(Basis(ecl_up_vec, ecl_tilt_vec, Vector3.ZERO), Vector3.ZERO)
	var inc_val: float = elements.get_keplerian_prop(&"inclination")
	_draw_arc_3d(inc_arc_frame, inc_val, inc_color, 1.5)


## Draws guide lines from the parent focus to the periapsis, apoapsis, semi-parameter,
## and mean anomaly at epoch grabber positions.
func _draw_conic_guides() -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var pos_scale: float = _get_position_scale()

	var pe_pos: Vector3 = elements.get_periapsis_relative_position(ref) * pos_scale
	var sp_pos: Vector3 = elements.get_semi_parameter_relative_position(ref) * pos_scale
	_draw_line_3d(Vector3.ZERO, pe_pos, PERIAPSIS_HANDLE_COLOR.darkened(0.25), 1.0)
	_draw_line_3d(Vector3.ZERO, sp_pos, SEMI_PARAMETER_HANDLE_COLOR.darkened(0.25), 1.0)

	if elements.is_bound_orbit():
		var ap_pos: Vector3 = elements.get_apoapsis_relative_position(ref) * pos_scale
		_draw_line_3d(Vector3.ZERO, ap_pos, APOAPSIS_HANDLE_COLOR.darkened(0.25), 1.0)

	if not _is_simulation_playing():
		var epoch_pos: Vector3 = elements.get_relative_position(0.0, ref) * pos_scale
		_draw_line_3d(Vector3.ZERO, epoch_pos, EPOCH_ANOMALY_HANDLE_COLOR.darkened(0.25), 1.0)


## Draws the body-centered guide rings, lines, and arcs for axial tilt, axial tilt
## longitude, and starting day angle around the orbiter's epoch position.
func _draw_body_orientation_guides(body_radius: float) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var body_pos: Vector3 = elements.get_relative_position(0.0, ref) * _get_position_scale()
	var orbital_basis: Basis = elements.get_orbital_plane_basis(ref)
	var tilted_basis: Basis = elements.get_tilted_body_basis(ref)
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0

	if not elements.is_locked_to_ecliptic():
		var lon_color: Color = AXIAL_TILT_LONGITUDE_HANDLE_COLOR.darkened(0.15)
		var lon_ring_color := Color(lon_color.r, lon_color.g, lon_color.b, 0.35)
		var lon_u: Vector3 = (orbital_basis.x * dir_sign) * body_radius
		var lon_v: Vector3 = (-orbital_basis.z) * body_radius
		var lon_frame := Transform3D(Basis(lon_u, lon_v, Vector3.ZERO), body_pos)
		_draw_circle_3d(lon_frame, lon_ring_color)
		_draw_arc_3d(
				lon_frame,
				elements.get_keplerian_prop(&"axial_tilt_longitude"),
				lon_color,
				1.5
		)
		_draw_line_3d(
				body_pos,
				_get_axial_tilt_longitude_gizmo_position(body_radius),
				lon_color,
				1.5
		)

		var tilt_color: Color = AXIAL_TILT_HANDLE_COLOR.darkened(0.15)
		var tilt_faint := Color(tilt_color.r, tilt_color.g, tilt_color.b, 0.45)
		var pole_len: float = body_radius * JrRailOrbiter.AXIAL_TILT_GIZMO_RADIUS_SCALE
		var tilt_u: Vector3 = orbital_basis.y * pole_len
		var tilt_v: Vector3 = elements.get_axial_tilt_direction(ref) * pole_len
		_draw_line_3d(body_pos, body_pos + tilt_u, tilt_faint, 1.0)
		_draw_line_3d(body_pos, _get_axial_tilt_gizmo_position(body_radius), tilt_color, 1.5)
		var tilt_arc_frame := Transform3D(Basis(tilt_u, tilt_v, Vector3.ZERO), body_pos)
		_draw_arc_3d(tilt_arc_frame, elements.get_keplerian_prop(&"axial_tilt"), tilt_color, 1.5)

	var day_color: Color = ROTATION_AT_EPOCH_HANDLE_COLOR.darkened(0.15)
	var day_ring_color := Color(day_color.r, day_color.g, day_color.b, 0.35)
	var eq_radius: float = body_radius * JrRailOrbiter.STARTING_DAY_GIZMO_RADIUS_SCALE
	var eq_u: Vector3 = (-tilted_basis.z) * eq_radius
	var eq_v: Vector3 = (-dir_sign * tilted_basis.x) * eq_radius
	var eq_frame := Transform3D(Basis(eq_u, eq_v, Vector3.ZERO), body_pos)
	_draw_circle_3d(eq_frame, day_ring_color)
	_draw_arc_3d(eq_frame, elements.get_keplerian_prop(&"rotation_at_epoch"), day_color, 1.5)
	_draw_line_3d(body_pos, _get_starting_day_angle_gizmo_position(body_radius), day_color, 1.5)


## Draws all visible grabber handles on [member _current_overlay].
func _draw_all_grabber_handles(body_radius: float) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var pos_scale: float = _get_position_scale()
	var locked_2d: bool = elements.is_locked_to_ecliptic()

	if not locked_2d:
		_draw_handle_if_visible(
				_get_ascending_node_gizmo_position(),
				ASCENDING_NODE_HANDLE_COLOR,
				GrabberType.ASCENDING_NODE
		)
		_draw_handle_if_visible(
				_get_inclination_gizmo_position(),
				INCLINATION_HANDLE_COLOR,
				GrabberType.INCLINATION
		)

	if elements.is_bound_orbit():
		_draw_handle_if_visible(
				elements.get_apoapsis_relative_position(ref) * pos_scale,
				APOAPSIS_HANDLE_COLOR,
				GrabberType.APOAPSIS
		)

	_draw_handle_if_visible(
			elements.get_semi_parameter_relative_position(ref) * pos_scale,
			SEMI_PARAMETER_HANDLE_COLOR,
			GrabberType.SEMI_PARAMETER
	)
	_draw_handle_if_visible(
			elements.get_periapsis_relative_position(ref) * pos_scale,
			PERIAPSIS_HANDLE_COLOR,
			GrabberType.PERIAPSIS
	)

	if not _is_simulation_playing():
		_draw_handle_if_visible(
				elements.get_relative_position(0.0, ref) * pos_scale,
				EPOCH_ANOMALY_HANDLE_COLOR,
				GrabberType.MEAN_ANOMALY_AT_EPOCH
		)
		if elements is JrRailOrbiter:
			_draw_handle_if_visible(
					_get_starting_day_angle_gizmo_position(body_radius),
					ROTATION_AT_EPOCH_HANDLE_COLOR,
					GrabberType.ROTATION_AT_EPOCH
			)
			if not locked_2d:
				_draw_handle_if_visible(
						_get_axial_tilt_longitude_gizmo_position(body_radius),
						AXIAL_TILT_LONGITUDE_HANDLE_COLOR,
						GrabberType.AXIAL_TILT_LONGITUDE
				)
				_draw_handle_if_visible(
						_get_axial_tilt_gizmo_position(body_radius),
						AXIAL_TILT_HANDLE_COLOR,
						GrabberType.AXIAL_TILT
				)


## Projects [param focus_pos] and draws its circular grabber handle if in front of the camera.
func _draw_handle_if_visible(focus_pos: Vector3, color: Color, grabber: GrabberType) -> void:
	if not _is_focus_pos_visible(focus_pos):
		return
	var screen_pos: Vector2 = _project_focus_pos(focus_pos)
	var highlighted: bool = _is_handle_highlighted(grabber)
	var radius: float = HANDLE_DRAW_RADIUS + (2.0 if highlighted else 0.0)
	_current_overlay.draw_circle(
			screen_pos,
			radius + 1.5,
			Color.WHITE if highlighted else Color(0.1, 0.1, 0.1, 0.9)
	)
	_current_overlay.draw_circle(
			screen_pos,
			radius,
			color.lightened(0.2) if highlighted else color
	)


## Draws a descriptive text label next to the currently active or hovered grabber handle.
func _draw_active_or_hovered_label(body_radius: float) -> void:
	var target: GrabberType = (
			_active_grabber
			if _active_grabber != GrabberType.NONE
			else _hovered_grabber
	)
	if target == GrabberType.NONE:
		return

	var focus_pos: Vector3 = _get_grabber_focus_position(target, body_radius)
	if not _is_focus_pos_visible(focus_pos):
		return

	var screen_pos: Vector2 = _project_focus_pos(focus_pos) + Vector2(12.0, -10.0)
	var label_text: String = _format_grabber_label(target)
	var font: Font = _current_overlay.get_theme_default_font()
	var font_size: int = _current_overlay.get_theme_default_font_size()
	if font == null:
		return

	_current_overlay.draw_string(
			font,
			screen_pos + Vector2(1.0, 1.0),
			label_text,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1.0,
			font_size,
			Color(0.0, 0.0, 0.0, 0.85)
	)
	_current_overlay.draw_string(
			font,
			screen_pos,
			label_text,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1.0,
			font_size,
			Color.WHITE
	)


## Returns a formatted string describing [param grabber] and its current value.
func _format_grabber_label(grabber: GrabberType) -> String:
	var elements: JrOrreryElement = _get_target_orbiter()
	var pe: float = elements.get_keplerian_prop(&"periapsis")
	var ap: float = elements.get_keplerian_prop(&"apoapsis")
	var ecc: float = elements.get_keplerian_prop(&"eccentricity")
	var arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis")
	match grabber:
		GrabberType.PERIAPSIS:
			return "Periapsis: %.2f (ω: %.1f°)" % [pe, rad_to_deg(arg_pe)]
		GrabberType.APOAPSIS:
			return "Apoapsis: %.2f (ω: %.1f°)" % [ap, rad_to_deg(arg_pe)]
		GrabberType.SEMI_PARAMETER:
			return "Semi-Parameter (e: %.3f)" % ecc
		GrabberType.MEAN_ANOMALY_AT_EPOCH:
			return "Mean Anomaly at Epoch: %.1f°" % rad_to_deg(
					elements.get_keplerian_prop(&"mean_anomaly_at_epoch")
			)
		GrabberType.ASCENDING_NODE:
			return "Right Ascension of Ascending Node: %.1f°" % rad_to_deg(
					elements.get_keplerian_prop(&"longitude_of_the_ascending_node")
			)
		GrabberType.INCLINATION:
			return "Orbital Inclination: %.1f°" % rad_to_deg(
					elements.get_keplerian_prop(&"inclination")
			)
		GrabberType.AXIAL_TILT:
			return "Axial Tilt: %.1f°" % rad_to_deg(elements.get_keplerian_prop(&"axial_tilt"))
		GrabberType.AXIAL_TILT_LONGITUDE:
			return "Axial Tilt Longitude: %.1f°" % rad_to_deg(
					elements.get_keplerian_prop(&"axial_tilt_longitude")
			)
		GrabberType.ROTATION_AT_EPOCH:
			return "Starting Day Angle: %.1f°" % rad_to_deg(
					elements.get_keplerian_prop(&"rotation_at_epoch")
			)
	return ""


## Returns [code]true[/code] if [param grabber] is currently active or hovered.
func _is_handle_highlighted(grabber: GrabberType) -> bool:
	return (
			_active_grabber == grabber
			or (_active_grabber == GrabberType.NONE and _hovered_grabber == grabber)
	)


## Returns the 3D position (in scaled driver-focus space) of the specified [param grabber].
func _get_grabber_focus_position(grabber: GrabberType, body_radius: float) -> Vector3:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var pos_scale: float = _get_position_scale()
	match grabber:
		GrabberType.PERIAPSIS:
			return elements.get_periapsis_relative_position(ref) * pos_scale
		GrabberType.APOAPSIS:
			return elements.get_apoapsis_relative_position(ref) * pos_scale
		GrabberType.SEMI_PARAMETER:
			return elements.get_semi_parameter_relative_position(ref) * pos_scale
		GrabberType.MEAN_ANOMALY_AT_EPOCH:
			return elements.get_relative_position(0.0, ref) * pos_scale
		GrabberType.ASCENDING_NODE:
			return _get_ascending_node_gizmo_position()
		GrabberType.INCLINATION:
			return _get_inclination_gizmo_position()
		GrabberType.AXIAL_TILT:
			return _get_axial_tilt_gizmo_position(body_radius)
		GrabberType.AXIAL_TILT_LONGITUDE:
			return _get_axial_tilt_longitude_gizmo_position(body_radius)
		GrabberType.ROTATION_AT_EPOCH:
			return _get_starting_day_angle_gizmo_position(body_radius)
	return Vector3.ZERO


## Returns an ordered list of candidate [enum GrabberType] handles active for hit-testing.
func _get_active_grabber_candidates() -> Array[GrabberType]:
	var elements: JrOrreryElement = _get_target_orbiter()
	var locked_2d: bool = elements.is_locked_to_ecliptic()
	var candidates: Array[GrabberType] = []
	if not locked_2d:
		candidates.append(GrabberType.ASCENDING_NODE)
		candidates.append(GrabberType.INCLINATION)
	candidates.append(GrabberType.PERIAPSIS)
	candidates.append(GrabberType.SEMI_PARAMETER)
	if elements.is_bound_orbit():
		candidates.append(GrabberType.APOAPSIS)
	if not _is_simulation_playing():
		candidates.append(GrabberType.MEAN_ANOMALY_AT_EPOCH)
		if elements is JrRailOrbiter:
			candidates.append(GrabberType.ROTATION_AT_EPOCH)
			if not locked_2d:
				candidates.append(GrabberType.AXIAL_TILT_LONGITUDE)
				candidates.append(GrabberType.AXIAL_TILT)
	return candidates


## Tests which 3D grabber handle (if any) lies within [constant HANDLE_HIT_RADIUS] of
## [param screen_pos].
func _hit_test_grabbers(screen_pos: Vector2) -> GrabberType:
	var vp_size: Vector2 = _current_camera.get_viewport().get_visible_rect().size
	var body_radius: float = _compute_body_gizmo_radius(_current_camera, vp_size)
	var best_grabber := GrabberType.NONE
	var best_dist_sq: float = HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS

	for candidate: GrabberType in _get_active_grabber_candidates():
		var focus_pos: Vector3 = _get_grabber_focus_position(candidate, body_radius)
		if not _is_focus_pos_visible(focus_pos):
			continue
		var dist_sq: float = screen_pos.distance_squared_to(_project_focus_pos(focus_pos))
		if dist_sq <= best_dist_sq:
			best_dist_sq = dist_sq
			best_grabber = candidate

	return best_grabber


## Handles mouse button press and release events for starting and committing 3D grabber drags.
func _handle_mouse_button(event: InputEventMouseButton) -> bool:
	if event.button_index != MOUSE_BUTTON_LEFT:
		return false

	if event.pressed:
		var hit: GrabberType = _hit_test_grabbers(event.position)
		if hit == GrabberType.NONE:
			return false
		_begin_drag(hit)
		_request_redraw()
		return true

	if _active_grabber != GrabberType.NONE:
		_commit_drag()
		_request_redraw()
		return true

	return false


## Handles mouse motion events for updating 3D grabber hover state or dragging the active grabber.
func _handle_mouse_motion(event: InputEventMouseMotion) -> bool:
	if _active_grabber != GrabberType.NONE:
		_update_drag(event.position, event.ctrl_pressed, event.alt_pressed)
		_request_redraw()
		return true

	var hovered: GrabberType = _hit_test_grabbers(event.position)
	if hovered != _hovered_grabber:
		_hovered_grabber = hovered
		_request_redraw()
	return false


## Records initial 3D orbital properties when starting a grabber drag operation.
func _begin_drag(grabber: GrabberType) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	_active_grabber = grabber
	_undo_snapshot = _capture_elements_snapshot(elements)

	if grabber == GrabberType.PERIAPSIS:
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"apoapsis")
	elif grabber == GrabberType.APOAPSIS:
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"periapsis")


## Captures all gizmo-editable properties of [param elements] into a dictionary.
func _capture_elements_snapshot(elements: JrOrreryElement) -> Dictionary[StringName, float]:
	var snap: Dictionary[StringName, float] = {
		&"periapsis": elements.get_keplerian_prop(&"periapsis"),
		&"eccentricity": elements.get_keplerian_prop(&"eccentricity"),
		&"inclination": elements.get_keplerian_prop(&"inclination"),
		&"longitude_of_the_ascending_node": elements.get_keplerian_prop(
				&"longitude_of_the_ascending_node"
		),
		&"argument_of_periapsis": elements.get_keplerian_prop(&"argument_of_periapsis"),
		&"mean_anomaly_at_epoch": elements.get_keplerian_prop(&"mean_anomaly_at_epoch"),
	}
	if elements is JrRailOrbiter:
		snap[&"axial_tilt"] = elements.get_keplerian_prop(&"axial_tilt")
		snap[&"axial_tilt_longitude"] = elements.get_keplerian_prop(&"axial_tilt_longitude")
		snap[&"rotation_at_epoch"] = elements.get_keplerian_prop(&"rotation_at_epoch")
	return snap


## Commits the completed 3D drag operation to the editor's [EditorUndoRedoManager].
func _commit_drag() -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var final_snapshot: Dictionary[StringName, float] = _capture_elements_snapshot(elements)
	_active_grabber = GrabberType.NONE

	if not _has_snapshot_changed(_undo_snapshot, final_snapshot) or _plugin == null:
		return

	var undo_redo: EditorUndoRedoManager = _plugin.get_undo_redo()
	undo_redo.create_action("Adjust 3D Orbital Parameters")
	for prop: StringName in final_snapshot.keys():
		undo_redo.add_do_property(elements, prop, final_snapshot[prop])
		undo_redo.add_undo_property(elements, prop, _undo_snapshot[prop])
	undo_redo.commit_action(false)


## Returns [code]true[/code] if any property in [param after] differs from [param before].
func _has_snapshot_changed(
		before: Dictionary[StringName, float],
		after: Dictionary[StringName, float]
) -> bool:
	for prop: StringName in after.keys():
		if not is_equal_approx(before.get(prop, 0.0), after[prop]):
			return true
	return false


## Intersects the mouse ray at [param mouse_screen_pos] with the 3D plane defined by
## [param plane_frame] (in driver-focus space, where [code]origin[/code] is the plane
## center, [code]basis.x[/code] is the 0-radian axis, [code]basis.y[/code] is the
## +90-degree axis, and [code]basis.z[/code] is the plane normal) and returns the
## local 2D coordinates [code](u, v)[/code] on that plane.
func _intersect_drag_plane(plane_frame: Transform3D, mouse_screen_pos: Vector2) -> Vector2:
	var world_to_focus: Transform3D = _current_focus_to_world.affine_inverse()
	var ray_origin: Vector3 = world_to_focus * _current_camera.project_ray_origin(mouse_screen_pos)
	var cam_ray_normal: Vector3 = _current_camera.project_ray_normal(mouse_screen_pos)
	var ray_dir: Vector3 = (world_to_focus.basis * cam_ray_normal).normalized()

	var normal: Vector3 = plane_frame.basis.z.normalized()
	var denom: float = ray_dir.dot(normal)
	if absf(denom) < 0.15:
		var sign_bias: float = -1.0 if denom <= 0.0 else 1.0
		normal = (normal + ray_dir * (sign_bias * 0.25)).normalized()
		denom = ray_dir.dot(normal)

	var t: float = (plane_frame.origin - ray_origin).dot(normal) / denom
	if t <= 0.0:
		t = maxf(0.0, (plane_frame.origin - ray_origin).dot(ray_dir))

	var hit_offset: Vector3 = (ray_origin + ray_dir * t) - plane_frame.origin
	return Vector2(hit_offset.dot(plane_frame.basis.x), hit_offset.dot(plane_frame.basis.y))


## Returns the polar angle [code]atan2(v, u)[/code] where the mouse ray at
## [param mouse_screen_pos] intersects [param plane_frame].
func _get_drag_plane_angle(plane_frame: Transform3D, mouse_screen_pos: Vector2) -> float:
	var plane_uv: Vector2 = _intersect_drag_plane(plane_frame, mouse_screen_pos)
	return plane_uv.angle()


## Dispatches the active grabber drag update based on [param mouse_screen_pos] and modifier keys.
func _update_drag(mouse_screen_pos: Vector2, ctrl_pressed: bool, alt_pressed: bool) -> void:
	match _active_grabber:
		GrabberType.PERIAPSIS, GrabberType.APOAPSIS:
			var orbital_vec: Vector2 = _get_orbital_plane_cursor_vector(mouse_screen_pos)
			_drag_apsis_grabber(orbital_vec, ctrl_pressed, alt_pressed)
		GrabberType.SEMI_PARAMETER:
			_drag_semi_parameter_grabber(_get_orbital_plane_cursor_vector(mouse_screen_pos))
		GrabberType.MEAN_ANOMALY_AT_EPOCH:
			_drag_mean_anomaly_at_epoch_grabber(_get_orbital_plane_cursor_vector(mouse_screen_pos))
		GrabberType.ASCENDING_NODE:
			_drag_ascending_node_grabber(mouse_screen_pos)
		GrabberType.INCLINATION:
			_drag_inclination_grabber(mouse_screen_pos)
		GrabberType.AXIAL_TILT:
			_drag_axial_tilt_grabber(mouse_screen_pos)
		GrabberType.AXIAL_TILT_LONGITUDE:
			_drag_axial_tilt_longitude_grabber(mouse_screen_pos)
		GrabberType.ROTATION_AT_EPOCH:
			_drag_rotation_at_epoch_grabber(mouse_screen_pos)


## Projects the cursor onto the tilted orbital plane centered at the focus and returns
## the unscaled 2D nodal coordinates (where +X is the ascending node and +Y is +90 degrees
## in-plane).
func _get_orbital_plane_cursor_vector(mouse_screen_pos: Vector2) -> Vector2:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var inclined_nodal: Basis = elements.get_inclined_nodal_basis(ref)
	var orbital_plane_frame := Transform3D(
			Basis(-inclined_nodal.z, inclined_nodal.x, inclined_nodal.y),
			Vector3.ZERO
	)
	var scaled_vec: Vector2 = _intersect_drag_plane(orbital_plane_frame, mouse_screen_pos)
	return scaled_vec / maxf(0.0001, _get_position_scale())


## Updates periapsis, apoapsis, eccentricity, and argument of periapsis while dragging
## either the periapsis or apoapsis grabber, swapping handles if dragged past the opposite apsis.
func _drag_apsis_grabber(
		orbital_vec: Vector2,
		lock_eccentricity: bool,
		lock_arg_periapsis: bool
) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var distance: float = maxf(0.001, orbital_vec.length())
	var angle: float = orbital_vec.angle()

	if lock_eccentricity:
		_drag_apsis_locked_eccentricity(elements, distance, angle, lock_arg_periapsis)
		return

	if is_inf(_drag_opposite_apsis_distance):
		elements.set_keplerian_prop(&"periapsis", distance)
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle)
		return

	if distance <= _drag_opposite_apsis_distance:
		_active_grabber = GrabberType.PERIAPSIS
		elements.set_keplerian_prop(&"periapsis", distance)
		elements.set_keplerian_prop(&"apoapsis", _drag_opposite_apsis_distance)
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle)
	else:
		_active_grabber = GrabberType.APOAPSIS
		elements.set_keplerian_prop(&"periapsis", _drag_opposite_apsis_distance)
		elements.set_keplerian_prop(&"apoapsis", distance)
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle + PI)


## Handles apsis dragging when Ctrl is held to lock eccentricity.
func _drag_apsis_locked_eccentricity(
		elements: JrOrreryElement,
		distance: float,
		angle: float,
		lock_arg_periapsis: bool
) -> void:
	if _active_grabber == GrabberType.PERIAPSIS:
		elements.set_keplerian_prop(&"periapsis", distance)
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle)
		if elements.is_bound_orbit():
			_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"apoapsis")
	elif _active_grabber == GrabberType.APOAPSIS:
		var e: float = elements.get_keplerian_prop(&"eccentricity")
		elements.set_keplerian_prop(&"periapsis", maxf(0.001, distance * (1.0 - e) / (1.0 + e)))
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle + PI)
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"periapsis")


## Updates eccentricity from the dragged semi-parameter position without modifying periapsis.
func _drag_semi_parameter_grabber(orbital_vec: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var pe: float = elements.get_keplerian_prop(&"periapsis")
	var arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis")
	var perifocal_vec: Vector2 = orbital_vec.rotated(-arg_pe)
	var new_semi_parameter: float = maxf(pe, absf(perifocal_vec.y))
	elements.set_keplerian_prop(&"eccentricity", maxf(0.0, new_semi_parameter / pe - 1.0))


## Updates mean anomaly at epoch from the cursor's true anomaly angle around the parent focus.
func _drag_mean_anomaly_at_epoch_grabber(orbital_vec: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis")
	var perifocal_vec: Vector2 = orbital_vec.rotated(-arg_pe)
	var true_anomaly: float = perifocal_vec.angle()
	elements.set_keplerian_prop(
			&"mean_anomaly_at_epoch",
			elements.get_mean_anomaly_from_true_anomaly(true_anomaly)
	)


## Updates longitude of the ascending node (RAAN) by projecting the cursor onto the reference
## ecliptic plane.
func _drag_ascending_node_grabber(mouse_screen_pos: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var ecliptic_frame := Transform3D(
			Basis(ref.vernal_point, ref.progression_axis, ref.ecliptic_north_pole),
			Vector3.ZERO
	)
	elements.set_keplerian_prop(
			&"longitude_of_the_ascending_node",
			_get_drag_plane_angle(ecliptic_frame, mouse_screen_pos)
	)


## Updates orbital inclination by projecting the cursor onto the plane perpendicular to the
## ascending node line.
func _drag_inclination_grabber(mouse_screen_pos: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var nodal_ecl: Basis = elements.get_nodal_ecliptic_basis(ref)
	var inc_frame := Transform3D(
			Basis(nodal_ecl.y, -nodal_ecl.x, -nodal_ecl.z),
			Vector3.ZERO
	)
	elements.set_keplerian_prop(&"inclination", _get_drag_plane_angle(inc_frame, mouse_screen_pos))


## Updates axial tilt by projecting the cursor onto the body's axial tilt plane around the epoch
## position.
func _drag_axial_tilt_grabber(mouse_screen_pos: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var body_pos: Vector3 = elements.get_relative_position(0.0, ref) * _get_position_scale()
	var orbital_normal: Vector3 = elements.get_orbital_plane_basis(ref).y
	var tilt_dir: Vector3 = elements.get_axial_tilt_direction(ref)
	var tilt_hinge: Vector3 = elements.get_axial_tilt_hinge_axis(ref)
	var tilt_frame := Transform3D(
			Basis(orbital_normal, tilt_dir, tilt_hinge),
			body_pos
	)
	elements.set_keplerian_prop(&"axial_tilt", _get_drag_plane_angle(tilt_frame, mouse_screen_pos))


## Updates axial tilt longitude by projecting the cursor onto the orbital plane centered at the
## body's epoch position.
func _drag_axial_tilt_longitude_grabber(mouse_screen_pos: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var body_pos: Vector3 = elements.get_relative_position(0.0, ref) * _get_position_scale()
	var orbital_basis: Basis = elements.get_orbital_plane_basis(ref)
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var lon_frame := Transform3D(
			Basis(orbital_basis.x * dir_sign, -orbital_basis.z, orbital_basis.y),
			body_pos
	)
	elements.set_keplerian_prop(
			&"axial_tilt_longitude",
			_get_drag_plane_angle(lon_frame, mouse_screen_pos)
	)


## Updates starting day angle by projecting the cursor onto the body's tilted equatorial plane at
## the epoch position.
func _drag_rotation_at_epoch_grabber(mouse_screen_pos: Vector2) -> void:
	var elements: JrOrreryElement = _get_target_orbiter()
	var ref: JrReferenceDirection = _get_reference_plane()
	var body_pos: Vector3 = elements.get_relative_position(0.0, ref) * _get_position_scale()
	var tilted_basis: Basis = elements.get_tilted_body_basis(ref)
	var dir_sign: float = -1.0 if ref.positive_is_clockwise else 1.0
	var eq_frame := Transform3D(
			Basis(-tilted_basis.z, -dir_sign * tilted_basis.x, tilted_basis.y),
			body_pos
	)
	elements.set_keplerian_prop(
			&"rotation_at_epoch",
			_get_drag_plane_angle(eq_frame, mouse_screen_pos)
	)
