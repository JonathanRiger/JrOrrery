@tool

## Interactive 2D viewport gizmo controller for [JrBallisticOrbiter] and its [JrOrbiter2D] followers.
##
## Renders focus-to-body guide lines, a 2D velocity vector arrow, hover readouts,
## and interactive grabber handles for [member JrBallisticOrbiter.initial_position] and
## [member JrBallisticOrbiter.initial_velocity] over the 2D editor viewport with full
## [EditorUndoRedoManager] support.
extends RefCounted
class_name JrBallistic2DGizmo

## Identifies which 2D ballistic handle is hovered or being dragged.
enum GrabberType {
	NONE,
	INITIAL_POSITION,
	INITIAL_VELOCITY
}

## Visual radius (in screen pixels) of each 2D grabber handle.
const HANDLE_DRAW_RADIUS: float = 6.0;

## Hit-test radius (in screen pixels) for clicking or hovering a 2D grabber handle.
const HANDLE_HIT_RADIUS: float = 10.0;

## Fill color for the initial position grabber handle.
const POSITION_HANDLE_COLOR: Color = Color(1.0, 0.68, 0.22, 1.0);

## Fill color for the initial velocity vector grabber handle.
const VELOCITY_HANDLE_COLOR: Color = Color(0.25, 0.95, 0.68, 1.0);

#region Private Members
## Owning [EditorPlugin] used to schedule overlay redraws and commit Undo/Redo actions.
var _plugin: EditorPlugin = null;

## Active [JrOrbiter2D] follower bound to a [JrBallistic] simulation node.
var _edited_follower: JrOrbiter2D = null;

## Grabber handle currently being dragged, or [constant GrabberType.NONE].
var _active_grabber: GrabberType = GrabberType.NONE;

## Grabber handle currently hovered by the cursor.
var _hovered_grabber: GrabberType = GrabberType.NONE;

## Initial position snapshot captured at the start of a drag for Undo/Redo.
var _undo_initial_position: Vector3 = Vector3.ZERO;

## Initial velocity snapshot captured at the start of a drag for Undo/Redo.
var _undo_initial_velocity: Vector3 = Vector3.ZERO;
#endregion

## Initializes the 2D ballistic gizmo controller with its owning [param plugin].
func _init(plugin: EditorPlugin = null) -> void:
	_plugin = plugin;

## Updates the owning [param plugin] reference.
func set_plugin(plugin: EditorPlugin) -> void:
	_plugin = plugin;

## Assigns the active [param follower] ([JrOrbiter2D] bound to a [JrBallistic])
## and manages signal connections for viewport overlay redraws.
func set_edited_follower(follower: JrOrbiter2D) -> void:
	if _edited_follower == follower:
		return;
	if _edited_follower != null and is_instance_valid(_edited_follower):
		if _edited_follower.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_follower.orbit_visuals_changed.disconnect(_request_redraw);

	_edited_follower = follower;
	_active_grabber = GrabberType.NONE;
	_hovered_grabber = GrabberType.NONE;

	if _edited_follower != null and is_instance_valid(_edited_follower):
		if not _edited_follower.orbit_visuals_changed.is_connected(_request_redraw):
			_edited_follower.orbit_visuals_changed.connect(_request_redraw);
	_request_redraw();

## Returns [code]true[/code] when a valid [JrOrbiter2D] bound to a [JrBallisticOrbiter]
## in [constant JrBallisticOrbiter.InitialStateMode.STATE_VECTORS] mode is active in the scene tree
## and simulation playback is stopped.
func can_edit() -> bool:
	if _edited_follower == null or not is_instance_valid(_edited_follower) or not _edited_follower.is_inside_tree():
		return false;
	var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
	if ballistic == null or ballistic.is_simulation_playing():
		return false;
	if ballistic.initial_state_mode != JrBallisticOrbiter.InitialStateMode.STATE_VECTORS:
		return false;
	return _edited_follower.get_reference_plane() != null;

## Draws focus guide lines, velocity arrow, grabber handles, and hover readouts
## over the 2D editor viewport [param overlay].
func draw_over_viewport(overlay: Control) -> void:
	if not can_edit() or overlay == null:
		return;

	var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();
	var focus_screen: Vector2 = focus_to_screen * Vector2.ZERO;

	var pos_2d: Vector2 = _edited_follower.project_3d_to_2d(ballistic.get_focus_initial_position());
	var vel_tip_2d: Vector2 = _edited_follower.project_3d_to_2d(ballistic.get_velocity_gizmo_focus_position());
	var pos_screen: Vector2 = focus_to_screen * pos_2d;
	var vel_screen: Vector2 = focus_to_screen * vel_tip_2d;

	overlay.draw_line(focus_screen, pos_screen, POSITION_HANDLE_COLOR.darkened(0.3), 1.0, true);
	_draw_velocity_arrow_2d(overlay, pos_screen, vel_screen);

	_draw_grabber_handle(overlay, pos_screen, POSITION_HANDLE_COLOR, _is_highlighted(GrabberType.INITIAL_POSITION));
	_draw_grabber_handle(overlay, vel_screen, VELOCITY_HANDLE_COLOR, _is_highlighted(GrabberType.INITIAL_VELOCITY));
	_draw_active_label(overlay, ballistic, pos_screen, vel_screen);

## Processes 2D viewport mouse input for hovering and dragging ballistic handles.
func handle_gui_input(event: InputEvent) -> bool:
	if not can_edit():
		return false;
	if event is InputEventMouseButton:
		return _handle_mouse_button(event as InputEventMouseButton);
	if event is InputEventMouseMotion:
		return _handle_mouse_motion(event as InputEventMouseMotion);
	return false;

#region Drawing & Hit-Testing Helpers
## Requests a viewport overlay redraw from the owning [EditorPlugin].
func _request_redraw() -> void:
	if _plugin != null:
		_plugin.update_overlays();

## Returns the [Transform2D] mapping scaled 2D focus coordinates into screen pixels.
func _get_focus_to_screen_transform() -> Transform2D:
	return _edited_follower.get_viewport_transform() * _edited_follower.get_focus_to_world_transform();

## Returns [code]true[/code] if [param grabber] is currently dragged or hovered.
func _is_highlighted(grabber: GrabberType) -> bool:
	return _active_grabber == grabber or (_active_grabber == GrabberType.NONE and _hovered_grabber == grabber);

## Draws a 2D arrow shaft and head between [param start_screen] and [param end_screen].
func _draw_velocity_arrow_2d(overlay: Control, start_screen: Vector2, end_screen: Vector2) -> void:
	var color: Color = VELOCITY_HANDLE_COLOR.darkened(0.15);
	overlay.draw_line(start_screen, end_screen, color, 2.0, true);
	var delta: Vector2 = end_screen - start_screen;
	if delta.length_squared() < 36.0:
		return;
	var dir: Vector2 = delta.normalized();
	var wing_a: Vector2 = end_screen - dir.rotated(0.45) * 9.0;
	var wing_b: Vector2 = end_screen - dir.rotated(-0.45) * 9.0;
	overlay.draw_line(end_screen, wing_a, color, 2.0, true);
	overlay.draw_line(end_screen, wing_b, color, 2.0, true);

## Draws a circular grabber handle on [param overlay] at [param screen_pos].
func _draw_grabber_handle(overlay: Control, screen_pos: Vector2, color: Color, highlighted: bool) -> void:
	var radius: float = HANDLE_DRAW_RADIUS + (2.0 if highlighted else 0.0);
	overlay.draw_circle(screen_pos, radius + 1.5, Color.WHITE if highlighted else Color(0.1, 0.1, 0.1, 0.9));
	overlay.draw_circle(screen_pos, radius, color.lightened(0.2) if highlighted else color);

## Draws a text readout next to the hovered or dragged handle.
func _draw_active_label(
	overlay: Control,
	ballistic: JrBallisticOrbiter,
	pos_screen: Vector2,
	vel_screen: Vector2
) -> void:
	var target: GrabberType = _active_grabber if _active_grabber != GrabberType.NONE else _hovered_grabber;
	if target == GrabberType.NONE:
		return;

	var font: Font = overlay.get_theme_default_font();
	if font == null:
		return;
	var font_size: int = overlay.get_theme_default_font_size();
	var anchor: Vector2 = pos_screen if target == GrabberType.INITIAL_POSITION else vel_screen;
	var label_pos: Vector2 = anchor + Vector2(12.0, -10.0);
	var text: String = _format_label_text(target, ballistic);

	overlay.draw_string(font, label_pos + Vector2.ONE, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color(0.0, 0.0, 0.0, 0.85));
	overlay.draw_string(font, label_pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE);

## Formats the readout text for [param grabber].
func _format_label_text(grabber: GrabberType, ballistic: JrBallisticOrbiter) -> String:
	if grabber == GrabberType.INITIAL_POSITION:
		var p: Vector3 = ballistic.initial_position;
		return "Initial Position: (%.2f, %.2f, %.2f)" % [p.x, p.y, p.z];
	var v: Vector3 = ballistic.initial_velocity;
	return "Initial Velocity: (%.2f, %.2f, %.2f) |v|=%.2f" % [v.x, v.y, v.z, v.length()];

## Hit-tests the 2D grabber handles against [param screen_pos].
func _hit_test(screen_pos: Vector2) -> GrabberType:
	var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();
	var pos_screen: Vector2 = focus_to_screen * _edited_follower.project_3d_to_2d(ballistic.get_focus_initial_position());
	var vel_screen: Vector2 = focus_to_screen * _edited_follower.project_3d_to_2d(ballistic.get_velocity_gizmo_focus_position());

	var best: GrabberType = GrabberType.NONE;
	var best_dist_sq: float = HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS;

	var vel_dist_sq: float = screen_pos.distance_squared_to(vel_screen);
	if vel_dist_sq <= best_dist_sq:
		best_dist_sq = vel_dist_sq;
		best = GrabberType.INITIAL_VELOCITY;

	var pos_dist_sq: float = screen_pos.distance_squared_to(pos_screen);
	if pos_dist_sq <= best_dist_sq:
		best = GrabberType.INITIAL_POSITION;

	return best;
#endregion

#region Dragging & Undo/Redo
## Handles left mouse press and release for starting and committing drags.
func _handle_mouse_button(event: InputEventMouseButton) -> bool:
	if event.button_index != MOUSE_BUTTON_LEFT:
		return false;

	if event.pressed:
		var hit: GrabberType = _hit_test(event.position);
		if hit == GrabberType.NONE:
			return false;
		var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
		_active_grabber = hit;
		_undo_initial_position = ballistic.initial_position;
		_undo_initial_velocity = ballistic.initial_velocity;
		_request_redraw();
		return true;

	if _active_grabber != GrabberType.NONE:
		_commit_drag();
		_request_redraw();
		return true;

	return false;

## Handles mouse motion to update hover state or drag the active handle.
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

## Updates [member JrBallisticOrbiter.initial_position] or [member JrBallisticOrbiter.initial_velocity]
## from [param mouse_screen_pos] while preserving the out-of-plane Y component.
func _update_drag(mouse_screen_pos: Vector2) -> void:
	var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();
	var focus_2d: Vector2 = focus_to_screen.affine_inverse() * mouse_screen_pos;
	var unscaled_3d: Vector3 = _edited_follower.unproject_2d_to_3d(focus_2d);

	if _active_grabber == GrabberType.INITIAL_POSITION:
		var curr_focus_pos: Vector3 = ballistic.get_focus_initial_position();
		ballistic.set_focus_initial_position(Vector3(unscaled_3d.x, curr_focus_pos.y, unscaled_3d.z));
	elif _active_grabber == GrabberType.INITIAL_VELOCITY:
		var curr_focus_pos: Vector3 = ballistic.get_focus_initial_position();
		var curr_focus_vel: Vector3 = ballistic.get_focus_initial_velocity();
		var scale_inv: float = 1.0 / maxf(0.01, ballistic.velocity_gizmo_scale);
		var new_vx: float = (unscaled_3d.x - curr_focus_pos.x) * scale_inv;
		var new_vz: float = (unscaled_3d.z - curr_focus_pos.z) * scale_inv;
		ballistic.set_focus_initial_velocity(Vector3(new_vx, curr_focus_vel.y, new_vz));

## Commits modified ballistic initial properties to [EditorUndoRedoManager].
func _commit_drag() -> void:
	var ballistic: JrBallisticOrbiter = _edited_follower.get_target_ballistic();
	var final_pos: Vector3 = ballistic.initial_position;
	var final_vel: Vector3 = ballistic.initial_velocity;
	_active_grabber = GrabberType.NONE;

	if final_pos.is_equal_approx(_undo_initial_position) and final_vel.is_equal_approx(_undo_initial_velocity):
		return;
	if _plugin == null:
		return;

	var undo_redo: EditorUndoRedoManager = _plugin.get_undo_redo();
	undo_redo.create_action("Adjust Ballistic Initial State");
	undo_redo.add_do_property(ballistic, "initial_position", final_pos);
	undo_redo.add_do_property(ballistic, "initial_velocity", final_vel);
	undo_redo.add_undo_property(ballistic, "initial_position", _undo_initial_position);
	undo_redo.add_undo_property(ballistic, "initial_velocity", _undo_initial_velocity);
	undo_redo.commit_action(false);
#endregion
