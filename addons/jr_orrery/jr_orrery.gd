@tool
@icon("res://addons/jr_orrery/icons/jr_orrery.svg")

## EditorPlugin entry point for JR Orrery.
##
## Registers custom simulation nodes ([JrOrbitDriver], [JrRailOrbiter], and [JrBallisticOrbiter])
## and visual follower nodes ([JrOrbiter2D] and [JrOrbiter3D]), and provides
## interactive 2D and 3D viewport grabber handles for editing [JrRailOrbiter]
## and [JrBallisticOrbiter] parameters directly in the scene view.
extends EditorPlugin

## Identifies which orbital handle is currently hovered or being dragged in the 2D viewport.
enum GrabberType {
	NONE,
	PERIAPSIS,
	APOAPSIS,
	SEMI_PARAMETER,
	MEAN_ANOMALY_AT_EPOCH
}

## Visual radius (in screen pixels) of each viewport grabber handle.
const HANDLE_DRAW_RADIUS: float = 6.0;

## Hit-test radius (in screen pixels) for clicking or hovering a viewport grabber handle.
const HANDLE_HIT_RADIUS: float = 10.0;

## Fill color for the periapsis grabber handle.
const PERIAPSIS_HANDLE_COLOR: Color = Color(1.0, 0.65, 0.2, 1.0);

## Fill color for the apoapsis grabber handle.
const APOAPSIS_HANDLE_COLOR: Color = Color(0.25, 0.8, 1.0, 1.0);

## Fill color for the semi-parameter grabber handle.
const SEMI_PARAMETER_HANDLE_COLOR: Color = Color(0.35, 0.95, 0.55, 1.0);

## Fill color for the mean anomaly at epoch grabber handle.
const EPOCH_ANOMALY_HANDLE_COLOR: Color = Color(0.95, 0.4, 0.85, 1.0);

#region Private Members
## The [JrOrbiter2D] currently selected and edited in the 2D viewport.
var _edited_orbiter: JrOrbiter2D = null;

## The grabber handle currently being dragged, or [constant GrabberType.NONE].
var _active_grabber: GrabberType = GrabberType.NONE;

## The grabber handle currently hovered by the cursor.
var _hovered_grabber: GrabberType = GrabberType.NONE;

## Distance of the stationary opposite apsis recorded during an apsis drag operation.
var _drag_opposite_apsis_distance: float = 1.0;

## Initial periapsis value recorded at the start of a drag for Undo/Redo support.
var _undo_periapsis: float = 1.0;

## Initial eccentricity value recorded at the start of a drag for Undo/Redo support.
var _undo_eccentricity: float = 0.0;

## Initial argument of periapsis recorded at the start of a drag for Undo/Redo support.
var _undo_argument_of_periapsis: float = 0.0;

## Initial mean anomaly at epoch recorded at the start of a drag for Undo/Redo support.
var _undo_mean_anomaly_at_epoch: float = 0.0;

## Interactive 3D viewport gizmo controller for [JrOrbiter3D], [JrRailOrbiter], and Keplerian [JrBallisticOrbiter] nodes.
var _gizmo_3d: JrOrbiter3DGizmo = null;

## Interactive 2D viewport gizmo controller for [JrBallisticOrbiter] and its [JrOrbiter2D] followers.
var _ballistic_gizmo_2d: JrBallistic2DGizmo = null;

## Interactive 3D viewport gizmo controller for [JrBallisticOrbiter] and its [JrOrbiter3D] followers.
var _ballistic_gizmo_3d: JrBallistic3DGizmo = null;
#endregion

## Called when the plugin is enabled in Project Settings.
func _enable_plugin() -> void:
	pass;

## Called when the plugin is disabled in Project Settings.
func _disable_plugin() -> void:
	pass;

## Registers custom nodes when the plugin enters the editor tree.
func _enter_tree() -> void:
	_ensure_gizmo_3d();
	_ensure_ballistic_gizmos();
	add_custom_type(
		"JrOrbitDriver",
		"Node",
		preload("res://addons/jr_orrery/jr_orbit_driver.gd"),
		preload("res://addons/jr_orrery/icons/orbit_driver.svg")
	);
	add_custom_type(
		"JrRailOrbiter",
		"Node",
		preload("res://addons/jr_orrery/jr_rail_orbiter.gd"),
		preload("res://addons/jr_orrery/icons/orbiter.svg")
	);
	add_custom_type(
		"JrBallisticOrbiter",
		"Node",
		preload("res://addons/jr_orrery/jr_ballistic_orbiter.gd"),
		preload("res://addons/jr_orrery/icons/ballistic.svg")
	);
	add_custom_type(
		"JrOrbiter2D",
		"Node2D",
		preload("res://addons/jr_orrery/jr_orbiter_2d.gd"),
		preload("res://addons/jr_orrery/icons/orbiter_2d.svg")
	);
	add_custom_type(
		"JrOrbiter3D",
		"Node3D",
		preload("res://addons/jr_orrery/jr_orbiter_3d.gd"),
		preload("res://addons/jr_orrery/icons/orbiter_3d.svg")
	);

## Unregisters custom nodes when the plugin exits the editor tree.
func _exit_tree() -> void:
	var selection: EditorSelection = EditorInterface.get_selection();
	if selection != null and selection.selection_changed.is_connected(_on_editor_selection_changed):
		selection.selection_changed.disconnect(_on_editor_selection_changed);
	_set_edited_orbiter(null);
	if _gizmo_3d != null:
		_gizmo_3d.set_edited_target(null, null);
		_gizmo_3d = null;
	if _ballistic_gizmo_2d != null:
		_ballistic_gizmo_2d.set_edited_follower(null);
		_ballistic_gizmo_2d = null;
	if _ballistic_gizmo_3d != null:
		_ballistic_gizmo_3d.set_edited_target(null, null);
		_ballistic_gizmo_3d = null;
	remove_custom_type("JrOrbiter3D");
	remove_custom_type("JrOrbiter2D");
	remove_custom_type("JrBallisticOrbiter");
	remove_custom_type("JrRailOrbiter");
	remove_custom_type("JrOrbitDriver");

## Polls 3D camera transform changes and ensures hot-reloaded gizmo state stays synchronized.
func _process(_delta: float) -> void:
	_ensure_gizmo_3d().poll_camera_changes();
	_ensure_ballistic_gizmos();
	_ballistic_gizmo_3d.poll_camera_changes();

## Returns [code]true[/code] when [param object] is a [JrOrbiter2D], [JrOrbiter3D],
## [JrRailOrbiter], [JrBallisticOrbiter], or a direct visual child of a [JrOrbiter2D]/[JrOrbiter3D].
func _handles(object: Object) -> bool:
	if object is JrRailOrbiter or object is JrBallisticOrbiter:
		return true;
	if _resolve_orbiter_2d(object) != null:
		return true;
	if _resolve_orbiter_3d(object) != null:
		return true;
	return false;

## Assigns the currently selected [JrOrbiter2D], [JrOrbiter3D], [JrRailOrbiter], or [JrBallisticOrbiter]
## for viewport handle editing.
func _edit(object: Object) -> void:
	var gizmo_3d: JrOrbiter3DGizmo = _ensure_gizmo_3d();
	_ensure_ballistic_gizmos();
	var scene_root: Node = EditorInterface.get_edited_scene_root() if Engine.is_editor_hint() else null;

	if object is JrBallisticOrbiter:
		var sim_ballistic: JrBallisticOrbiter = object as JrBallisticOrbiter;
		if sim_ballistic.has_keplerian_gizmo():
			_clear_ballistic_gizmos();
			_set_edited_orbiter(_find_follower_2d(scene_root, sim_ballistic));
			gizmo_3d.set_edited_target(sim_ballistic, _find_follower_3d(scene_root, sim_ballistic));
		else:
			_edit_ballistic_target(sim_ballistic, null, null, scene_root);
		return;

	if object is JrRailOrbiter:
		var sim_orbiter: JrRailOrbiter = object as JrRailOrbiter;
		_clear_ballistic_gizmos();
		_set_edited_orbiter(_find_follower_2d(scene_root, sim_orbiter));
		gizmo_3d.set_edited_target(sim_orbiter, _find_follower_3d(scene_root, sim_orbiter));
		return;

	var orbiter_2d: JrOrbiter2D = _resolve_orbiter_2d(object);
	if orbiter_2d != null:
		_edit_from_follower_2d(orbiter_2d, scene_root);
		return;

	var orbiter_3d: JrOrbiter3D = _resolve_orbiter_3d(object);
	if orbiter_3d != null:
		_edit_from_follower_3d(orbiter_3d, scene_root);
		return;

	_clear_all_edited_targets();

## Clears the active orbiter and ballistic gizmos when the plugin's gizmos are hidden.
func _make_visible(visible: bool) -> void:
	if not visible:
		_clear_all_edited_targets();
	update_overlays();

## Draws the active 3D orbital or ballistic grabbers over the 3D editor viewport.
func _forward_3d_draw_over_viewport(overlay: Control) -> void:
	_ensure_gizmo_3d().draw_over_viewport(overlay);
	_ensure_ballistic_gizmos();
	_ballistic_gizmo_3d.draw_over_viewport(overlay);

## Intercepts 3D viewport mouse input to hover and drag [JrOrbiter3D] or [JrBallisticOrbiter] handles.
func _forward_3d_gui_input(viewport_camera: Camera3D, event: InputEvent) -> int:
	_ensure_ballistic_gizmos();
	if _ballistic_gizmo_3d.handle_gui_input(viewport_camera, event):
		return EditorPlugin.AFTER_GUI_INPUT_STOP;
	if _ensure_gizmo_3d().handle_gui_input(viewport_camera, event):
		return EditorPlugin.AFTER_GUI_INPUT_STOP;
	return EditorPlugin.AFTER_GUI_INPUT_PASS;

## Draws the periapsis, apoapsis, semi-parameter, and epoch anomaly grabbers (or
## ballistic position/velocity grabbers) over the 2D editor viewport.
func _forward_canvas_draw_over_viewport(overlay: Control) -> void:
	_ensure_ballistic_gizmos();
	if _ballistic_gizmo_2d.can_edit():
		_ballistic_gizmo_2d.draw_over_viewport(overlay);
		return;
	if not _can_edit_orbiter():
		return;

	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var ref: JrReferenceDirection = _edited_orbiter.get_reference_plane();
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();
	var focus_screen: Vector2 = focus_to_screen * Vector2.ZERO;

	var pe_2d: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_periapsis_relative_position(ref));
	var sp_2d: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_semi_parameter_relative_position(ref));
	var pe_screen: Vector2 = focus_to_screen * pe_2d;
	var sp_screen: Vector2 = focus_to_screen * sp_2d;

	overlay.draw_line(focus_screen, pe_screen, PERIAPSIS_HANDLE_COLOR.darkened(0.25), 1.0, true);
	overlay.draw_line(focus_screen, sp_screen, SEMI_PARAMETER_HANDLE_COLOR.darkened(0.25), 1.0, true);

	if elements.is_bound_orbit():
		var ap_2d: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_apoapsis_relative_position(ref));
		var ap_screen: Vector2 = focus_to_screen * ap_2d;
		overlay.draw_line(focus_screen, ap_screen, APOAPSIS_HANDLE_COLOR.darkened(0.25), 1.0, true);
		_draw_grabber_handle(overlay, ap_screen, APOAPSIS_HANDLE_COLOR, _is_handle_highlighted(GrabberType.APOAPSIS));

	if not _edited_orbiter.is_simulation_playing():
		var epoch_2d: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_relative_position(0.0, ref));
		var epoch_screen: Vector2 = focus_to_screen * epoch_2d;
		overlay.draw_line(focus_screen, epoch_screen, EPOCH_ANOMALY_HANDLE_COLOR.darkened(0.25), 1.0, true);
		_draw_grabber_handle(overlay, epoch_screen, EPOCH_ANOMALY_HANDLE_COLOR, _is_handle_highlighted(GrabberType.MEAN_ANOMALY_AT_EPOCH));

	_draw_grabber_handle(overlay, sp_screen, SEMI_PARAMETER_HANDLE_COLOR, _is_handle_highlighted(GrabberType.SEMI_PARAMETER));
	_draw_grabber_handle(overlay, pe_screen, PERIAPSIS_HANDLE_COLOR, _is_handle_highlighted(GrabberType.PERIAPSIS));

## Intercepts 2D viewport mouse input to drag orbital or ballistic grabbers.
func _forward_canvas_gui_input(event: InputEvent) -> bool:
	_ensure_ballistic_gizmos();
	if _ballistic_gizmo_2d.handle_gui_input(event):
		return true;
	if not _can_edit_orbiter():
		return false;

	if event is InputEventMouseButton:
		return _handle_mouse_button(event as InputEventMouseButton);
	elif event is InputEventMouseMotion:
		return _handle_mouse_motion(event as InputEventMouseMotion);

	return false;

#region Follower Lookup & State Helpers
## Lazily initializes [member _gizmo_3d] and connects [signal EditorSelection.selection_changed]
## so 3D gizmos recover automatically even after script hot-reloads in the editor.
func _ensure_gizmo_3d() -> JrOrbiter3DGizmo:
	var was_null: bool = _gizmo_3d == null;
	if _gizmo_3d == null:
		_gizmo_3d = JrOrbiter3DGizmo.new(self);
	elif _gizmo_3d.get_plugin() == null:
		_gizmo_3d.set_plugin(self);

	if Engine.is_editor_hint():
		var selection: EditorSelection = EditorInterface.get_selection();
		if selection != null and not selection.selection_changed.is_connected(_on_editor_selection_changed):
			selection.selection_changed.connect(_on_editor_selection_changed);
		if was_null:
			_on_editor_selection_changed();

	return _gizmo_3d;

## Synchronizes the active 2D/3D edited orbiter with the current [EditorSelection].
func _on_editor_selection_changed() -> void:
	if not Engine.is_editor_hint():
		return;
	var selection: EditorSelection = EditorInterface.get_selection();
	if selection == null:
		return;
	var selected_nodes: Array[Node] = selection.get_selected_nodes();
	if selected_nodes.size() == 1 and _handles(selected_nodes[0]):
		_edit(selected_nodes[0]);

## Resolves [param object] to a [JrOrbiter2D] if it is one or a direct child [CanvasItem] of one.
func _resolve_orbiter_2d(object: Object) -> JrOrbiter2D:
	if object is JrOrbiter2D:
		return object as JrOrbiter2D;
	if object is CanvasItem and (object as Node).get_parent() is JrOrbiter2D:
		return (object as Node).get_parent() as JrOrbiter2D;
	return null;

## Resolves [param object] to a [JrOrbiter3D] if it is one or a direct child [Node3D] of one.
func _resolve_orbiter_3d(object: Object) -> JrOrbiter3D:
	if object is JrOrbiter3D:
		return object as JrOrbiter3D;
	if object is Node3D and (object as Node).get_parent() is JrOrbiter3D:
		return (object as Node).get_parent() as JrOrbiter3D;
	return null;

## Lazily initializes [member _ballistic_gizmo_2d] and [member _ballistic_gizmo_3d].
func _ensure_ballistic_gizmos() -> void:
	if _ballistic_gizmo_2d == null:
		_ballistic_gizmo_2d = JrBallistic2DGizmo.new(self);
	else:
		_ballistic_gizmo_2d.set_plugin(self);

	if _ballistic_gizmo_3d == null:
		_ballistic_gizmo_3d = JrBallistic3DGizmo.new(self);
	else:
		_ballistic_gizmo_3d.set_plugin(self);

## Clears edited targets across both conic and ballistic gizmo controllers.
func _clear_all_edited_targets() -> void:
	_set_edited_orbiter(null);
	if _gizmo_3d != null:
		_gizmo_3d.set_edited_target(null, null);
	_clear_ballistic_gizmos();

## Clears active targets on the 2D and 3D ballistic gizmo controllers.
func _clear_ballistic_gizmos() -> void:
	if _ballistic_gizmo_2d != null:
		_ballistic_gizmo_2d.set_edited_follower(null);
	if _ballistic_gizmo_3d != null:
		_ballistic_gizmo_3d.set_edited_target(null, null);

## Binds a [JrBallisticOrbiter] simulation node and its 2D/3D followers to the ballistic gizmos.
func _edit_ballistic_target(
	ballistic: JrBallisticOrbiter,
	explicit_2d: JrOrbiter2D,
	explicit_3d: JrOrbiter3D,
	scene_root: Node
) -> void:
	_set_edited_orbiter(null);
	_ensure_gizmo_3d().set_edited_target(null, null);
	var follower_2d: JrOrbiter2D = explicit_2d if explicit_2d != null else _find_follower_2d(scene_root, ballistic);
	var follower_3d: JrOrbiter3D = explicit_3d if explicit_3d != null else _find_follower_3d(scene_root, ballistic);
	_ballistic_gizmo_2d.set_edited_follower(follower_2d);
	_ballistic_gizmo_3d.set_edited_target(ballistic, follower_3d);

## Routes selection of a [JrOrbiter2D] follower to either the ballistic or conic gizmos.
func _edit_from_follower_2d(orbiter_2d: JrOrbiter2D, scene_root: Node) -> void:
	var ballistic_2d: JrBallisticOrbiter = orbiter_2d.get_target_ballistic();
	if ballistic_2d != null and not ballistic_2d.has_keplerian_gizmo():
		_edit_ballistic_target(ballistic_2d, orbiter_2d, null, scene_root);
		return;
	_clear_ballistic_gizmos();
	var sim_from_2d: JrOrreryElement = orbiter_2d.get_target_keplerian_element();
	_set_edited_orbiter(orbiter_2d);
	_ensure_gizmo_3d().set_edited_target(sim_from_2d, _find_follower_3d(scene_root, sim_from_2d));

## Routes selection of a [JrOrbiter3D] follower to either the ballistic or conic gizmos.
func _edit_from_follower_3d(orbiter_3d: JrOrbiter3D, scene_root: Node) -> void:
	var ballistic_3d: JrBallisticOrbiter = orbiter_3d.get_target_ballistic();
	if ballistic_3d != null and not ballistic_3d.has_keplerian_gizmo():
		_edit_ballistic_target(ballistic_3d, null, orbiter_3d, scene_root);
		return;
	_clear_ballistic_gizmos();
	var sim_from_3d: JrOrreryElement = orbiter_3d.get_target_keplerian_element();
	_set_edited_orbiter(_find_follower_2d(scene_root, sim_from_3d));
	_ensure_gizmo_3d().set_edited_orbiter(orbiter_3d);

## Searches the subtree rooted at [param root] for the first [JrOrbiter2D]
## whose [member JrOrbiter2D.orbiter] is [param target].
func _find_follower_2d(root: Node, target: JrOrreryElement) -> JrOrbiter2D:
	if root == null or target == null:
		return null;
	if root is JrOrbiter2D and (root as JrOrbiter2D).orbiter == target:
		return root as JrOrbiter2D;
	for child: Node in root.get_children():
		var found: JrOrbiter2D = _find_follower_2d(child, target);
		if found != null:
			return found;
	return null;

## Searches the subtree rooted at [param root] for the first [JrOrbiter3D]
## whose [member JrOrbiter3D.orbiter] is [param target].
func _find_follower_3d(root: Node, target: JrOrreryElement) -> JrOrbiter3D:
	if root == null or target == null:
		return null;
	if root is JrOrbiter3D and (root as JrOrbiter3D).orbiter == target:
		return root as JrOrbiter3D;
	for child: Node in root.get_children():
		var found: JrOrbiter3D = _find_follower_3d(child, target);
		if found != null:
			return found;
	return null;

## Updates [member _edited_orbiter] and manages signal connections for viewport overlay redraws.
func _set_edited_orbiter(orbiter: JrOrbiter2D) -> void:
	if _edited_orbiter == orbiter:
		return;
	if _edited_orbiter != null and _edited_orbiter.orbit_visuals_changed.is_connected(update_overlays):
		_edited_orbiter.orbit_visuals_changed.disconnect(update_overlays);

	_edited_orbiter = orbiter;
	_active_grabber = GrabberType.NONE;
	_hovered_grabber = GrabberType.NONE;

	if _edited_orbiter != null and not _edited_orbiter.orbit_visuals_changed.is_connected(update_overlays):
		_edited_orbiter.orbit_visuals_changed.connect(update_overlays);
	update_overlays();

## Returns [code]true[/code] if [member _edited_orbiter] is valid, in the tree,
## and references a valid Keplerian [JrOrreryElement] node.
func _can_edit_orbiter() -> bool:
	if _edited_orbiter == null or not is_instance_valid(_edited_orbiter) or not _edited_orbiter.is_inside_tree():
		return false;
	var target_orbiter: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	if target_orbiter == null or not target_orbiter.is_orbit_valid():
		return false;
	return _edited_orbiter.get_reference_plane() != null;

## Returns the [Transform2D] that maps scaled 2D coordinates from the parent focus
## space into 2D editor viewport overlay pixel coordinates.
func _get_focus_to_screen_transform() -> Transform2D:
	return _edited_orbiter.get_viewport_transform() * _edited_orbiter.get_focus_to_world_transform();

## Returns [code]true[/code] if [param grabber] is currently active or hovered.
func _is_handle_highlighted(grabber: GrabberType) -> bool:
	return _active_grabber == grabber or (_active_grabber == GrabberType.NONE and _hovered_grabber == grabber);

## Draws a circular grabber handle on [param overlay] at [param screen_pos].
func _draw_grabber_handle(overlay: Control, screen_pos: Vector2, color: Color, highlighted: bool) -> void:
	var radius: float = HANDLE_DRAW_RADIUS + (2.0 if highlighted else 0.0);
	overlay.draw_circle(screen_pos, radius + 1.5, Color.WHITE if highlighted else Color(0.1, 0.1, 0.1, 0.9));
	overlay.draw_circle(screen_pos, radius, color.lightened(0.2) if highlighted else color);
#endregion

#region 2D Grabber Hit-Testing & Dragging
## Tests which grabber handle (if any) lies within [constant HANDLE_HIT_RADIUS] of [param screen_pos].
func _hit_test_grabbers(screen_pos: Vector2) -> GrabberType:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var ref: JrReferenceDirection = _edited_orbiter.get_reference_plane();
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();

	var best_grabber: GrabberType = GrabberType.NONE;
	var best_dist_sq: float = HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS;

	var pe_pos: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_periapsis_relative_position(ref));
	var pe_dist_sq: float = screen_pos.distance_squared_to(focus_to_screen * pe_pos);
	if pe_dist_sq <= best_dist_sq:
		best_dist_sq = pe_dist_sq;
		best_grabber = GrabberType.PERIAPSIS;

	if elements.is_bound_orbit():
		var ap_pos: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_apoapsis_relative_position(ref));
		var ap_dist_sq: float = screen_pos.distance_squared_to(focus_to_screen * ap_pos);
		if ap_dist_sq <= best_dist_sq:
			best_dist_sq = ap_dist_sq;
			best_grabber = GrabberType.APOAPSIS;

	var sp_pos: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_semi_parameter_relative_position(ref));
	var sp_dist_sq: float = screen_pos.distance_squared_to(focus_to_screen * sp_pos);
	if sp_dist_sq <= best_dist_sq:
		best_dist_sq = sp_dist_sq;
		best_grabber = GrabberType.SEMI_PARAMETER;

	if not _edited_orbiter.is_simulation_playing():
		var epoch_pos: Vector2 = _edited_orbiter.project_3d_to_2d(elements.get_relative_position(0.0, ref));
		var epoch_dist_sq: float = screen_pos.distance_squared_to(focus_to_screen * epoch_pos);
		if epoch_dist_sq <= best_dist_sq:
			best_grabber = GrabberType.MEAN_ANOMALY_AT_EPOCH;

	return best_grabber;

## Handles mouse button press and release events for starting and committing grabber drags.
func _handle_mouse_button(event: InputEventMouseButton) -> bool:
	if event.button_index != MOUSE_BUTTON_LEFT:
		return false;

	if event.pressed:
		var hit: GrabberType = _hit_test_grabbers(event.position);
		if hit == GrabberType.NONE:
			return false;
		_begin_drag(hit);
		update_overlays();
		return true;

	if _active_grabber != GrabberType.NONE:
		_commit_drag();
		update_overlays();
		return true;

	return false;

## Handles mouse motion events for updating hover state or dragging the active grabber.
func _handle_mouse_motion(event: InputEventMouseMotion) -> bool:
	if _active_grabber != GrabberType.NONE:
		_update_drag(event.position, event.ctrl_pressed, event.alt_pressed);
		update_overlays();
		return true;

	var hovered: GrabberType = _hit_test_grabbers(event.position);
	if hovered != _hovered_grabber:
		_hovered_grabber = hovered;
		update_overlays();
	return false;

## Records initial orbital parameter state when starting a grabber drag operation.
func _begin_drag(grabber: GrabberType) -> void:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	_active_grabber = grabber;
	_undo_periapsis = elements.get_keplerian_prop(&"periapsis");
	_undo_eccentricity = elements.get_keplerian_prop(&"eccentricity");
	_undo_argument_of_periapsis = elements.get_keplerian_prop(&"argument_of_periapsis");
	_undo_mean_anomaly_at_epoch = elements.get_keplerian_prop(&"mean_anomaly_at_epoch");

	if grabber == GrabberType.PERIAPSIS:
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"apoapsis");
	elif grabber == GrabberType.APOAPSIS:
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"periapsis");

## Commits the completed drag operation to the editor's [EditorUndoRedoManager].
func _commit_drag() -> void:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var final_periapsis: float = elements.get_keplerian_prop(&"periapsis");
	var final_eccentricity: float = elements.get_keplerian_prop(&"eccentricity");
	var final_arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis");
	var final_mean_anomaly: float = elements.get_keplerian_prop(&"mean_anomaly_at_epoch");

	_active_grabber = GrabberType.NONE;

	if (
		is_equal_approx(final_periapsis, _undo_periapsis)
		and is_equal_approx(final_eccentricity, _undo_eccentricity)
		and is_equal_approx(final_arg_pe, _undo_argument_of_periapsis)
		and is_equal_approx(final_mean_anomaly, _undo_mean_anomaly_at_epoch)
	):
		return;

	var undo_redo: EditorUndoRedoManager = get_undo_redo();
	undo_redo.create_action("Adjust Orbital Parameters");
	undo_redo.add_do_property(elements, "periapsis", final_periapsis);
	undo_redo.add_do_property(elements, "eccentricity", final_eccentricity);
	undo_redo.add_do_property(elements, "argument_of_periapsis", final_arg_pe);
	undo_redo.add_do_property(elements, "mean_anomaly_at_epoch", final_mean_anomaly);
	undo_redo.add_undo_property(elements, "periapsis", _undo_periapsis);
	undo_redo.add_undo_property(elements, "eccentricity", _undo_eccentricity);
	undo_redo.add_undo_property(elements, "argument_of_periapsis", _undo_argument_of_periapsis);
	undo_redo.add_undo_property(elements, "mean_anomaly_at_epoch", _undo_mean_anomaly_at_epoch);
	undo_redo.commit_action(false);

## Updates the orbital parameters based on the current cursor position and modifier keys.
func _update_drag(mouse_screen_pos: Vector2, ctrl_pressed: bool, alt_pressed: bool) -> void:
	var focus_to_screen: Transform2D = _get_focus_to_screen_transform();
	var rel_pos_2d: Vector2 = focus_to_screen.affine_inverse() * mouse_screen_pos;
	var rel_pos_3d: Vector3 = _edited_orbiter.unproject_2d_to_3d(rel_pos_2d);
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var ref: JrReferenceDirection = _edited_orbiter.get_reference_plane();
	var inclined_nodal: Basis = elements.get_inclined_nodal_basis(ref);
	var orbital_vec: Vector2 = Vector2(
		rel_pos_3d.dot(-inclined_nodal.z),
		rel_pos_3d.dot(inclined_nodal.x)
	);

	if _active_grabber == GrabberType.SEMI_PARAMETER:
		_drag_semi_parameter_grabber(orbital_vec);
	elif _active_grabber == GrabberType.MEAN_ANOMALY_AT_EPOCH:
		_drag_mean_anomaly_at_epoch_grabber(orbital_vec);
	else:
		_drag_apsis_grabber(orbital_vec, ctrl_pressed, alt_pressed);

## Updates periapsis, apoapsis, eccentricity, and argument of periapsis while dragging
## either the periapsis or apoapsis grabber, automatically swapping handles if one
## apsis is dragged past the other.
func _drag_apsis_grabber(orbital_vec: Vector2, lock_eccentricity: bool, lock_arg_periapsis: bool) -> void:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var distance: float = maxf(0.001, orbital_vec.length());
	var angle: float = orbital_vec.angle();

	if lock_eccentricity:
		_drag_apsis_locked_eccentricity(elements, distance, angle, lock_arg_periapsis);
		return;

	if is_inf(_drag_opposite_apsis_distance):
		elements.set_keplerian_prop(&"periapsis", distance);
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle);
		return;

	if distance <= _drag_opposite_apsis_distance:
		_active_grabber = GrabberType.PERIAPSIS;
		elements.set_keplerian_prop(&"periapsis", distance);
		elements.set_keplerian_prop(&"apoapsis", _drag_opposite_apsis_distance);
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle);
	else:
		_active_grabber = GrabberType.APOAPSIS;
		elements.set_keplerian_prop(&"periapsis", _drag_opposite_apsis_distance);
		elements.set_keplerian_prop(&"apoapsis", distance);
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle + PI);

## Handles apsis dragging when Ctrl is held to lock eccentricity.
func _drag_apsis_locked_eccentricity(
	elements: JrOrreryElement,
	distance: float,
	angle: float,
	lock_arg_periapsis: bool
) -> void:
	if _active_grabber == GrabberType.PERIAPSIS:
		elements.set_keplerian_prop(&"periapsis", distance);
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle);
		if elements.is_bound_orbit():
			_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"apoapsis");
	elif _active_grabber == GrabberType.APOAPSIS:
		var e: float = elements.get_keplerian_prop(&"eccentricity");
		elements.set_keplerian_prop(&"periapsis", maxf(0.001, distance * (1.0 - e) / (1.0 + e)));
		if not lock_arg_periapsis:
			elements.set_keplerian_prop(&"argument_of_periapsis", angle + PI);
		_drag_opposite_apsis_distance = elements.get_keplerian_prop(&"periapsis");

## Updates eccentricity from the dragged semi-parameter position without modifying periapsis.
func _drag_semi_parameter_grabber(orbital_vec: Vector2) -> void:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var pe: float = elements.get_keplerian_prop(&"periapsis");
	var arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis");
	var perifocal_vec: Vector2 = orbital_vec.rotated(-arg_pe);
	var new_semi_parameter: float = maxf(pe, absf(perifocal_vec.y));
	elements.set_keplerian_prop(&"eccentricity", maxf(0.0, new_semi_parameter / pe - 1.0));

## Updates mean anomaly at epoch from the cursor's true anomaly angle around the parent focus.
func _drag_mean_anomaly_at_epoch_grabber(orbital_vec: Vector2) -> void:
	var elements: JrOrreryElement = _edited_orbiter.get_target_keplerian_element();
	var arg_pe: float = elements.get_keplerian_prop(&"argument_of_periapsis");
	var perifocal_vec: Vector2 = orbital_vec.rotated(-arg_pe);
	var true_anomaly: float = perifocal_vec.angle();
	elements.set_keplerian_prop(&"mean_anomaly_at_epoch", elements.get_mean_anomaly_from_true_anomaly(true_anomaly));
#endregion
