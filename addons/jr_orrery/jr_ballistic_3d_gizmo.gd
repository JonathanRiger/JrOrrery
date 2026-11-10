@tool

## Interactive 3D viewport gizmo controller for [JrBallisticOrbiter] and its [JrOrbiter3D] followers.
##
## Renders 3D focus-to-body guides, vertical elevation drop lines, a 3D velocity
## vector arrow, standalone trajectory preview curves (when no [JrOrbiter3D] follower
## is present), and four interactive grabbers ([constant GrabberType.INITIAL_POSITION],
## [constant GrabberType.INITIAL_ELEVATION], [constant GrabberType.INITIAL_VELOCITY],
## and [constant GrabberType.INITIAL_VELOCITY_ELEVATION]) with full [EditorUndoRedoManager] support.
extends RefCounted
class_name JrBallistic3DGizmo

## Identifies which 3D ballistic grabber handle is hovered or being dragged.
enum GrabberType {
	NONE,
	INITIAL_POSITION,
	INITIAL_ELEVATION,
	INITIAL_VELOCITY,
	INITIAL_VELOCITY_ELEVATION
}

## Visual radius (in screen pixels) of each 3D viewport grabber handle.
const HANDLE_DRAW_RADIUS: float = 6.0;

## Hit-test radius (in screen pixels) for clicking or hovering a 3D grabber handle.
const HANDLE_HIT_RADIUS: float = 10.0;

## Minimum apparent screen-space length (in pixels) of vertical elevation stem handles.
const ELEVATION_STEM_SCREEN_PX: float = 34.0;

## Default preview curve color when editing a standalone [JrBallistic] without a [JrOrbiter3D].
const STANDALONE_PREVIEW_COLOR: Color = Color(1.0, 0.72, 0.28, 0.88);

## Fill color for the horizontal initial position grabber handle.
const POSITION_HANDLE_COLOR: Color = Color(1.0, 0.68, 0.22, 1.0);

## Fill color for the vertical initial position elevation grabber handle.
const POSITION_ELEVATION_COLOR: Color = Color(0.75, 0.52, 1.0, 1.0);

## Fill color for the horizontal initial velocity grabber handle.
const VELOCITY_HANDLE_COLOR: Color = Color(0.25, 0.95, 0.68, 1.0);

## Fill color for the vertical initial velocity elevation grabber handle.
const VELOCITY_ELEVATION_COLOR: Color = Color(0.35, 0.82, 1.0, 1.0);

#region Private Members
## Owning [EditorPlugin] used to request overlay redraws and commit Undo/Redo actions.
var _plugin: EditorPlugin = null;

## Optional [JrOrbiter3D] visual follower bound to [member _edited_sim_ballistic].
var _edited_follower: JrOrbiter3D = null;

## Active [JrBallisticOrbiter] simulation node being edited in the 3D viewport.
var _edited_sim_ballistic: JrBallisticOrbiter = null;

## Most recently active 3D editor viewport [Camera3D].
var _last_camera: Camera3D = null;

## Cached camera transform used to detect 3D viewport camera motion.
var _last_camera_transform: Transform3D = Transform3D.IDENTITY;

## Cached camera FOV or orthogonal size used to detect zoom changes.
var _last_camera_proj_param: float = 0.0;

## Active overlay [Control] during the current draw pass.
var _current_overlay: Control = null;

## Active [Camera3D] during the current draw or input pass.
var _current_camera: Camera3D = null;

## Active focus-to-world [Transform3D] during the current draw or input pass.
var _current_focus_to_world: Transform3D = Transform3D.IDENTITY;

## Grabber handle currently being dragged, or [constant GrabberType.NONE].
var _active_grabber: GrabberType = GrabberType.NONE;

## Grabber handle currently hovered by the cursor.
var _hovered_grabber: GrabberType = GrabberType.NONE;

## Snapshot of [member JrBallistic.initial_position] recorded at drag start.
var _undo_initial_position: Vector3 = Vector3.ZERO;

## Snapshot of [member JrBallistic.initial_velocity] recorded at drag start.
var _undo_initial_velocity: Vector3 = Vector3.ZERO;

## Vertical stem offset recorded at drag start so vertical dragging feels 1:1 with the cursor.
var _drag_stem_offset: float = 0.0;
#endregion

## Initializes the 3D ballistic gizmo controller with its owning [param plugin].
func _init(plugin: EditorPlugin = null) -> void:
	_plugin = plugin;

## Updates the owning [param plugin] reference.
func set_plugin(plugin: EditorPlugin) -> void:
	_plugin = plugin;

## Updates the active [param sim_ballistic] and optional [param visual_follower],
## managing signal connections for 3D overlay redraws.
func set_edited_target(sim_ballistic: JrBallisticOrbiter, visual_follower: JrOrbiter3D = null) -> void:
	if _edited_sim_ballistic == sim_ballistic and _edited_follower == visual_follower:
		return;

	if _edited_follower != null and is_instance_valid(_edited_follower):
		if _edited_follower.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_follower.orbit_visuals_changed.disconnect(_request_redraw);
	if _edited_sim_ballistic != null and is_instance_valid(_edited_sim_ballistic):
		if _edited_sim_ballistic.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_sim_ballistic.orbit_visuals_changed.disconnect(_request_redraw);

	_edited_sim_ballistic = sim_ballistic;
	_edited_follower = visual_follower;
	_active_grabber = GrabberType.NONE;
	_hovered_grabber = GrabberType.NONE;

	if _edited_follower != null and is_instance_valid(_edited_follower):
		if not _edited_follower.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_follower.orbit_visuals_changed.connect(_request_redraw);
	if _edited_sim_ballistic != null and is_instance_valid(_edited_sim_ballistic):
		if not _edited_sim_ballistic.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_sim_ballistic.orbit_visuals_changed.connect(_request_redraw);

	if _edited_sim_ballistic != null or _edited_follower != null:
		_ensure_camera_initialized();
	_request_redraw();

## Returns [code]true[/code] if a valid [JrBallisticOrbiter] node is bound for 3D vector editing.
func can_edit() -> bool:
	if _edited_follower != null and (not is_instance_valid(_edited_follower) or not _edited_follower.is_inside_tree()):
		return false;
	var target: JrBallisticOrbiter = _get_target_ballistic();
	if target == null or not is_instance_valid(target) or not target.is_inside_tree():
		return false;
	if target.is_simulation_playing():
		return false;
	if target.initial_state_mode != JrBallisticOrbiter.InitialStateMode.STATE_VECTORS:
		return false;
	return target.get_reference_plane() != null;

## Polls 3D viewport camera motion to schedule overlay redraws when orbiting or zooming.
func poll_camera_changes() -> void:
	if not can_edit():
		return;
	_ensure_camera_initialized();
	if _last_camera == null or not is_instance_valid(_last_camera):
		return;

	var proj_param: float = _last_camera.size if _last_camera.projection == Camera3D.PROJECTION_ORTHOGONAL else _last_camera.fov;
	if _last_camera.global_transform != _last_camera_transform or not is_equal_approx(proj_param, _last_camera_proj_param):
		_last_camera_transform = _last_camera.global_transform;
		_last_camera_proj_param = proj_param;
		_request_redraw();

## Draws 3D ballistic guides, velocity vector arrow, grabber handles, and hover labels.
func draw_over_viewport(overlay: Control) -> void:
	if not can_edit() or overlay == null:
		return;
	var camera: Camera3D = _resolve_camera_for_overlay(overlay);
	if camera == null:
		return;

	_current_overlay = overlay;
	_current_camera = camera;
	_current_focus_to_world = _compute_focus_to_world_transform();

	if _edited_follower == null:
		_draw_standalone_preview_curve();

	var pos_stem: float = _compute_stem_length(_get_scaled_body_pos(), overlay.size);
	var vel_stem: float = _compute_stem_length(_get_scaled_vel_tip_pos(), overlay.size);

	_draw_guides_and_stems(pos_stem, vel_stem);
	_draw_handles(pos_stem, vel_stem);
	_draw_active_or_hovered_label(pos_stem, vel_stem);

## Intercepts 3D viewport mouse events to hover and drag ballistic grabber handles.
func handle_gui_input(viewport_camera: Camera3D, event: InputEvent) -> bool:
	if viewport_camera != null:
		_last_camera = viewport_camera;
	if not can_edit() or viewport_camera == null:
		return false;

	_current_camera = viewport_camera;
	_current_focus_to_world = _compute_focus_to_world_transform();

	if event is InputEventMouseButton:
		return _handle_mouse_button(event as InputEventMouseButton);
	if event is InputEventMouseMotion:
		return _handle_mouse_motion(event as InputEventMouseMotion);
	return false;

#region Target, Camera & Coordinate Helpers
## Requests an overlay update on the owning [EditorPlugin].
func _request_redraw() -> void:
	if _plugin != null:
		_plugin.update_overlays();

## Returns the active [JrBallisticOrbiter] simulation node.
func _get_target_ballistic() -> JrBallisticOrbiter:
	if _edited_follower != null and is_instance_valid(_edited_follower):
		var from_follower: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
		if from_follower != null:
			return from_follower;
	if _edited_sim_ballistic != null and is_instance_valid(_edited_sim_ballistic):
		return _edited_sim_ballistic;
	return null;

## Returns the active position scale factor.
func _get_position_scale() -> float:
	if _edited_follower != null and is_instance_valid(_edited_follower):
		return _edited_follower.position_scale;
	return 1.0;

## Returns the reference north-pole unit vector in driver-focus space.
func _get_up_axis() -> Vector3:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var ref: JrReferenceDirection = target.get_reference_plane() if target != null else null;
	return ref.ecliptic_north_pole if ref != null else Vector3.UP;

## Returns the scaled initial body position in driver-focus space.
func _get_scaled_body_pos() -> Vector3:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	return target.get_focus_initial_position() * _get_position_scale() if target != null else Vector3.ZERO;

## Returns the scaled velocity vector tip position in driver-focus space.
func _get_scaled_vel_tip_pos() -> Vector3:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	return target.get_velocity_gizmo_focus_position() * _get_position_scale() if target != null else Vector3.ZERO;

## Computes the [Transform3D] mapping scaled driver-focus coordinates into 3D world space.
func _compute_focus_to_world_transform() -> Transform3D:
	if _edited_follower != null and is_instance_valid(_edited_follower) and _edited_follower.is_inside_tree():
		return _edited_follower.get_focus_to_world_transform();
	var target: JrBallisticOrbiter = _get_target_ballistic();
	if target == null:
		return Transform3D.IDENTITY;
	var focus_origin: Vector3 = target.get_driver_relative_transform().origin - target.get_focus_relative_position();
	return Transform3D(Basis.IDENTITY, focus_origin);

## Initializes [member _last_camera] from the primary 3D editor viewport.
func _ensure_camera_initialized() -> void:
	if (_last_camera != null and is_instance_valid(_last_camera)) or not Engine.is_editor_hint():
		return;
	var sub_vp: SubViewport = EditorInterface.get_editor_viewport_3d(0);
	if sub_vp != null:
		_last_camera = sub_vp.get_camera_3d();

## Resolves the [Camera3D] associated with [param overlay].
func _resolve_camera_for_overlay(overlay: Control) -> Camera3D:
	if Engine.is_editor_hint():
		var overlay_parent: Node = overlay.get_parent();
		if overlay_parent != null:
			for i: int in range(4):
				var sub_vp: SubViewport = EditorInterface.get_editor_viewport_3d(i);
				if sub_vp != null and overlay_parent.is_ancestor_of(sub_vp):
					return sub_vp.get_camera_3d();
	if _last_camera != null and is_instance_valid(_last_camera):
		return _last_camera;
	return null;

## Computes a camera-distance-compensated stem length in scaled focus units.
func _compute_stem_length(focus_anchor: Vector3, viewport_size: Vector2) -> float:
	var world_pos: Vector3 = _current_focus_to_world * focus_anchor;
	var vp_height: float = maxf(1.0, viewport_size.y);
	var units_per_px: float = 0.01;
	if _current_camera.projection == Camera3D.PROJECTION_ORTHOGONAL:
		units_per_px = _current_camera.size / vp_height;
	else:
		var cam_fwd: Vector3 = -_current_camera.global_transform.basis.z;
		var depth: float = maxf(0.1, (world_pos - _current_camera.global_position).dot(cam_fwd));
		units_per_px = (2.0 * depth * tan(deg_to_rad(_current_camera.fov) * 0.5)) / vp_height;

	var focus_scale: float = maxf(0.0001, _current_focus_to_world.basis.get_scale().length() / sqrt(3.0));
	return (ELEVATION_STEM_SCREEN_PX * units_per_px) / focus_scale;

## Returns [code]true[/code] if [param focus_pos] is in front of the near clip plane.
func _is_focus_pos_visible(focus_pos: Vector3) -> bool:
	var world_pos: Vector3 = _current_focus_to_world * focus_pos;
	var cam_fwd: Vector3 = -_current_camera.global_transform.basis.z;
	return (world_pos - _current_camera.global_position).dot(cam_fwd) >= maxf(0.05, _current_camera.near);

## Projects [param focus_pos] into 2D viewport screen coordinates.
func _project_focus_pos(focus_pos: Vector3) -> Vector2:
	return _current_camera.unproject_position(_current_focus_to_world * focus_pos);
#endregion

#region 3D Overlay Drawing
## Draws a near-plane-clipped 3D line segment between [param start_focus] and [param end_focus].
func _draw_line_3d(start_focus: Vector3, end_focus: Vector3, color: Color, width: float = 1.0) -> void:
	var a: Vector3 = _current_focus_to_world * start_focus;
	var b: Vector3 = _current_focus_to_world * end_focus;
	var cam_origin: Vector3 = _current_camera.global_position;
	var cam_fwd: Vector3 = -_current_camera.global_transform.basis.z;
	var min_dist: float = maxf(0.05, _current_camera.near);

	var da: float = (a - cam_origin).dot(cam_fwd);
	var db: float = (b - cam_origin).dot(cam_fwd);
	if da < min_dist and db < min_dist:
		return;
	if da < min_dist:
		a = a.lerp(b, (min_dist - da) / (db - da));
	elif db < min_dist:
		b = b.lerp(a, (min_dist - db) / (da - db));

	_current_overlay.draw_line(
		_current_camera.unproject_position(a),
		_current_camera.unproject_position(b),
		color,
		width,
		true
	);

## Draws the trajectory preview curve when editing a standalone [JrBallisticOrbiter].
func _draw_standalone_preview_curve() -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	if target == null:
		return;
	var pts: PackedVector3Array = target.get_trajectory_path_points();
	for i: int in range(pts.size() - 1):
		_draw_line_3d(pts[i], pts[i + 1], STANDALONE_PREVIEW_COLOR, 1.5);

## Draws focus guide lines, reference-plane drop lines, velocity vector, and vertical stems.
func _draw_guides_and_stems(pos_stem: float, vel_stem: float) -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var locked_2d: bool = target != null and target.is_locked_to_ecliptic();
	var up_axis: Vector3 = _get_up_axis();
	var body_pos: Vector3 = _get_scaled_body_pos();
	var vel_tip: Vector3 = _get_scaled_vel_tip_pos();

	_draw_line_3d(Vector3.ZERO, body_pos, POSITION_HANDLE_COLOR.darkened(0.25), 1.2);
	_draw_line_3d(body_pos, vel_tip, VELOCITY_HANDLE_COLOR.darkened(0.1), 2.0);
	if not locked_2d:
		var plane_proj_pos: Vector3 = body_pos - up_axis * body_pos.dot(up_axis);
		_draw_line_3d(plane_proj_pos, body_pos, Color(1.0, 0.68, 0.22, 0.4), 1.0);
		_draw_line_3d(body_pos, body_pos + up_axis * pos_stem, POSITION_ELEVATION_COLOR.darkened(0.15), 1.5);
		_draw_line_3d(vel_tip, vel_tip + up_axis * vel_stem, VELOCITY_ELEVATION_COLOR.darkened(0.15), 1.5);

## Draws all four 3D grabber handles if visible.
func _draw_handles(pos_stem: float, vel_stem: float) -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var locked_2d: bool = target != null and target.is_locked_to_ecliptic();
	var up_axis: Vector3 = _get_up_axis();
	var body_pos: Vector3 = _get_scaled_body_pos();
	var vel_tip: Vector3 = _get_scaled_vel_tip_pos();

	if not locked_2d:
		_draw_single_handle(body_pos + up_axis * pos_stem, POSITION_ELEVATION_COLOR, GrabberType.INITIAL_ELEVATION);
		_draw_single_handle(vel_tip + up_axis * vel_stem, VELOCITY_ELEVATION_COLOR, GrabberType.INITIAL_VELOCITY_ELEVATION);
	_draw_single_handle(vel_tip, VELOCITY_HANDLE_COLOR, GrabberType.INITIAL_VELOCITY);
	_draw_single_handle(body_pos, POSITION_HANDLE_COLOR, GrabberType.INITIAL_POSITION);

## Projects and draws a single circular grabber handle on [member _current_overlay].
func _draw_single_handle(focus_pos: Vector3, color: Color, grabber: GrabberType) -> void:
	if not _is_focus_pos_visible(focus_pos):
		return;
	var screen_pos: Vector2 = _project_focus_pos(focus_pos);
	var highlighted: bool = _active_grabber == grabber or (_active_grabber == GrabberType.NONE and _hovered_grabber == grabber);
	var radius: float = HANDLE_DRAW_RADIUS + (2.0 if highlighted else 0.0);
	_current_overlay.draw_circle(screen_pos, radius + 1.5, Color.WHITE if highlighted else Color(0.1, 0.1, 0.1, 0.9));
	_current_overlay.draw_circle(screen_pos, radius, color.lightened(0.2) if highlighted else color);

## Draws a text readout next to the hovered or active 3D grabber handle.
func _draw_active_or_hovered_label(pos_stem: float, vel_stem: float) -> void:
	var grabber: GrabberType = _active_grabber if _active_grabber != GrabberType.NONE else _hovered_grabber;
	if grabber == GrabberType.NONE:
		return;
	var focus_pos: Vector3 = _get_grabber_focus_pos(grabber, pos_stem, vel_stem);
	if not _is_focus_pos_visible(focus_pos):
		return;

	var font: Font = _current_overlay.get_theme_default_font();
	if font == null:
		return;
	var font_size: int = _current_overlay.get_theme_default_font_size();
	var screen_pos: Vector2 = _project_focus_pos(focus_pos) + Vector2(12.0, -10.0);
	var text: String = _format_grabber_label(grabber);

	_current_overlay.draw_string(font, screen_pos + Vector2.ONE, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color(0.0, 0.0, 0.0, 0.85));
	_current_overlay.draw_string(font, screen_pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE);

## Formats descriptive label text for [param grabber].
func _format_grabber_label(grabber: GrabberType) -> String:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var p: Vector3 = target.initial_position;
	var v: Vector3 = target.initial_velocity;
	match grabber:
		GrabberType.INITIAL_POSITION:
			return "Initial Position (X, Z): (%.2f, %.2f, %.2f)" % [p.x, p.y, p.z];
		GrabberType.INITIAL_ELEVATION:
			return "Initial Elevation (Y): %.2f" % p.y;
		GrabberType.INITIAL_VELOCITY:
			return "Initial Velocity (X, Z): (%.2f, %.2f, %.2f) |v|=%.2f" % [v.x, v.y, v.z, v.length()];
		GrabberType.INITIAL_VELOCITY_ELEVATION:
			return "Initial Vertical Velocity (Vy): %.2f" % v.y;
	return "";
#endregion

#region Hit-Testing, Ray Intersections & Undo/Redo
## Returns the 3D position in scaled focus space of [param grabber].
func _get_grabber_focus_pos(grabber: GrabberType, pos_stem: float, vel_stem: float) -> Vector3:
	var up_axis: Vector3 = _get_up_axis();
	var body_pos: Vector3 = _get_scaled_body_pos();
	var vel_tip: Vector3 = _get_scaled_vel_tip_pos();
	match grabber:
		GrabberType.INITIAL_POSITION:
			return body_pos;
		GrabberType.INITIAL_ELEVATION:
			return body_pos + up_axis * pos_stem;
		GrabberType.INITIAL_VELOCITY:
			return vel_tip;
		GrabberType.INITIAL_VELOCITY_ELEVATION:
			return vel_tip + up_axis * vel_stem;
	return Vector3.ZERO;

## Hit-tests all four 3D grabbers against [param screen_pos].
func _hit_test(screen_pos: Vector2) -> GrabberType:
	var vp_size: Vector2 = _current_camera.get_viewport().get_visible_rect().size;
	var pos_stem: float = _compute_stem_length(_get_scaled_body_pos(), vp_size);
	var vel_stem: float = _compute_stem_length(_get_scaled_vel_tip_pos(), vp_size);

	var target: JrBallisticOrbiter = _get_target_ballistic();
	var locked_2d: bool = target != null and target.is_locked_to_ecliptic();
	var candidates: Array[GrabberType] = [];
	if not locked_2d:
		candidates.append(GrabberType.INITIAL_ELEVATION);
		candidates.append(GrabberType.INITIAL_VELOCITY_ELEVATION);
	candidates.append(GrabberType.INITIAL_VELOCITY);
	candidates.append(GrabberType.INITIAL_POSITION);
	var best: GrabberType = GrabberType.NONE;
	var best_dist_sq: float = HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS;

	for candidate: GrabberType in candidates:
		var focus_pos: Vector3 = _get_grabber_focus_pos(candidate, pos_stem, vel_stem);
		if not _is_focus_pos_visible(focus_pos):
			continue;
		var dist_sq: float = screen_pos.distance_squared_to(_project_focus_pos(focus_pos));
		if dist_sq <= best_dist_sq:
			best_dist_sq = dist_sq;
			best = candidate;
	return best;

## Handles mouse button press and release events to start and commit 3D drags.
func _handle_mouse_button(event: InputEventMouseButton) -> bool:
	if event.button_index != MOUSE_BUTTON_LEFT:
		return false;

	if event.pressed:
		var hit: GrabberType = _hit_test(event.position);
		if hit == GrabberType.NONE:
			return false;
		_begin_drag(hit);
		_request_redraw();
		return true;

	if _active_grabber != GrabberType.NONE:
		_commit_drag();
		_request_redraw();
		return true;

	return false;

## Handles mouse motion events to update hover state or drag the active 3D grabber.
func _handle_mouse_motion(event: InputEventMouseMotion) -> bool:
	if _active_grabber != GrabberType.NONE:
		_update_drag(event.position);
		_request_redraw();
		return true;

	var hovered: GrabberType = _hit_test(event.position);
	if hovered != _hovered_grabber:
		_hovered_grabber = hovered;
		_request_redraw();
	return false;

## Snapshots initial properties when beginning a 3D grabber drag.
func _begin_drag(grabber: GrabberType) -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	_active_grabber = grabber;
	_undo_initial_position = target.initial_position;
	_undo_initial_velocity = target.initial_velocity;

	var vp_size: Vector2 = _current_camera.get_viewport().get_visible_rect().size;
	if grabber == GrabberType.INITIAL_ELEVATION:
		_drag_stem_offset = _compute_stem_length(_get_scaled_body_pos(), vp_size);
	elif grabber == GrabberType.INITIAL_VELOCITY_ELEVATION:
		_drag_stem_offset = _compute_stem_length(_get_scaled_vel_tip_pos(), vp_size);
	else:
		_drag_stem_offset = 0.0;

## Updates the dragged ballistic property by intersecting the cursor ray with a
## horizontal reference plane or vertical camera-facing billboard plane.
func _update_drag(mouse_screen_pos: Vector2) -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var inv_pos_scale: float = 1.0 / maxf(0.0001, _get_position_scale());
	var vel_scale_inv: float = 1.0 / maxf(0.01, target.velocity_gizmo_scale);

	match _active_grabber:
		GrabberType.INITIAL_POSITION:
			var hit_focus: Vector3 = _intersect_horizontal_plane(_get_scaled_body_pos(), mouse_screen_pos);
			var curr_local: Vector3 = target.initial_position;
			var new_local: Vector3 = target.get_reference_plane().rotation_basis.transposed() * (hit_focus * inv_pos_scale);
			target.initial_position = Vector3(new_local.x, curr_local.y, new_local.z);
		GrabberType.INITIAL_ELEVATION:
			var y_scaled: float = _intersect_vertical_axis(_get_scaled_body_pos(), mouse_screen_pos) - _drag_stem_offset;
			var pos_local: Vector3 = target.initial_position;
			target.initial_position = Vector3(pos_local.x, y_scaled * inv_pos_scale, pos_local.z);
		GrabberType.INITIAL_VELOCITY:
			var hit_vel_focus: Vector3 = _intersect_horizontal_plane(_get_scaled_vel_tip_pos(), mouse_screen_pos);
			var delta_unscaled: Vector3 = (hit_vel_focus * inv_pos_scale) - target.get_focus_initial_position();
			var delta_local: Vector3 = target.get_reference_plane().rotation_basis.transposed() * delta_unscaled;
			var curr_vel: Vector3 = target.initial_velocity;
			target.initial_velocity = Vector3(delta_local.x * vel_scale_inv, curr_vel.y, delta_local.z * vel_scale_inv);
		GrabberType.INITIAL_VELOCITY_ELEVATION:
			var anchor: Vector3 = _get_scaled_vel_tip_pos();
			var tip_h: float = _intersect_vertical_axis(anchor, mouse_screen_pos) - _drag_stem_offset;
			var body_h: float = _get_scaled_body_pos().dot(_get_up_axis());
			var vy: float = ((tip_h - body_h) * inv_pos_scale) * vel_scale_inv;
			var vel_local: Vector3 = target.initial_velocity;
			target.initial_velocity = Vector3(vel_local.x, vy, vel_local.z);

## Intersects the mouse ray with the horizontal plane orthogonal to [method _get_up_axis]
## passing through [param plane_origin_focus].
func _intersect_horizontal_plane(plane_origin_focus: Vector3, mouse_screen_pos: Vector2) -> Vector3:
	var world_to_focus: Transform3D = _current_focus_to_world.affine_inverse();
	var ray_origin: Vector3 = world_to_focus * _current_camera.project_ray_origin(mouse_screen_pos);
	var ray_dir: Vector3 = (world_to_focus.basis * _current_camera.project_ray_normal(mouse_screen_pos)).normalized();
	var normal: Vector3 = _get_up_axis();
	var denom: float = ray_dir.dot(normal);
	if absf(denom) < 0.08:
		var bias: float = -1.0 if denom <= 0.0 else 1.0;
		normal = (normal + ray_dir * (bias * 0.2)).normalized();
		denom = ray_dir.dot(normal);
	var t: float = maxf(0.0, (plane_origin_focus - ray_origin).dot(normal) / denom);
	return ray_origin + ray_dir * t;

## Projects the mouse ray onto the vertical axis passing through [param axis_anchor_focus]
## and returns the signed height coordinate along [method _get_up_axis].
func _intersect_vertical_axis(axis_anchor_focus: Vector3, mouse_screen_pos: Vector2) -> float:
	var world_to_focus: Transform3D = _current_focus_to_world.affine_inverse();
	var ray_origin: Vector3 = world_to_focus * _current_camera.project_ray_origin(mouse_screen_pos);
	var ray_dir: Vector3 = (world_to_focus.basis * _current_camera.project_ray_normal(mouse_screen_pos)).normalized();
	var up_axis: Vector3 = _get_up_axis();

	var cam_to_anchor: Vector3 = axis_anchor_focus - ray_origin;
	var side: Vector3 = cam_to_anchor.cross(up_axis);
	var plane_normal: Vector3 = up_axis.cross(side).normalized();
	if plane_normal.length_squared() < 0.01:
		return axis_anchor_focus.dot(up_axis);

	var denom: float = ray_dir.dot(plane_normal);
	if absf(denom) < 0.05:
		return axis_anchor_focus.dot(up_axis);
	var t: float = maxf(0.0, (axis_anchor_focus - ray_origin).dot(plane_normal) / denom);
	var hit_pos: Vector3 = ray_origin + ray_dir * t;
	return hit_pos.dot(up_axis);

## Commits modified 3D ballistic properties to [EditorUndoRedoManager].
func _commit_drag() -> void:
	var target: JrBallisticOrbiter = _get_target_ballistic();
	var final_pos: Vector3 = target.initial_position;
	var final_vel: Vector3 = target.initial_velocity;
	_active_grabber = GrabberType.NONE;

	if final_pos.is_equal_approx(_undo_initial_position) and final_vel.is_equal_approx(_undo_initial_velocity):
		return;
	if _plugin == null:
		return;

	var undo_redo: EditorUndoRedoManager = _plugin.get_undo_redo();
	undo_redo.create_action("Adjust 3D Ballistic Initial State");
	undo_redo.add_do_property(target, "initial_position", final_pos);
	undo_redo.add_do_property(target, "initial_velocity", final_vel);
	undo_redo.add_undo_property(target, "initial_position", _undo_initial_position);
	undo_redo.add_undo_property(target, "initial_velocity", _undo_initial_velocity);
	undo_redo.commit_action(false);
#endregion
