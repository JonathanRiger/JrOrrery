@icon("res://addons/jr_orrery/icons/demo_controller.svg")

## Unified runtime controller for the JR Orrery 2D/3D demo launcher and scenario sub-scenes.
##
## Drives both a 2D view hierarchy ([Camera2D] and [JrOrbiter2D] followers) and
## a 3D view hierarchy ([Camera3D] and [JrOrbiter3D] followers) from a single
## shared [JrOrbitDriver] simulation tree. Supports seamless runtime transfer
## between the 2D and 3D views, scenario switching across showcase sub-scenes,
## camera zooming/panning/orbiting, runtime orbit trajectory rendering, LCA-relative
## ballistic wake trails, and synchronized body perspective selection.
extends Node
class_name JrDemoController

## Identifies which visual projection and camera type is currently active.
enum ViewMode {
	## Top-down 2D canvas projection using [Camera2D] and [JrOrbiter2D] followers.
	VIEW_2D,
	## Full 3D perspective view using [Camera3D] and [JrOrbiter3D] followers.
	VIEW_3D
}

## Minimum allowed 2D camera zoom factor.
const MIN_ZOOM_2D: float = 0.04;

## Maximum allowed 2D camera zoom factor.
const MAX_ZOOM_2D: float = 6.0;

## Minimum allowed 3D camera distance from the camera pivot.
const MIN_CAMERA_DISTANCE_3D: float = 2.0;

## Maximum allowed 3D camera distance from the camera pivot.
const MAX_CAMERA_DISTANCE_3D: float = 900.0;

## Product coupling 2D camera zoom and 3D camera distance when transferring views.
const ZOOM_DISTANCE_PRODUCT: float = 37.4;

## Multiplicative zoom step applied per mouse wheel tick.
const ZOOM_STEP: float = 1.15;

## Radians of 3D camera rotation per pixel of mouse drag when orbiting the camera.
const ORBIT_SENSITIVITY_3D: float = 0.0065;

## Screen-space hit radius (in pixels) used when double-clicking a 2D body.
const CLICK_HIT_RADIUS_2D_PX: float = 28.0;

## Screen-space hit radius (in pixels) used when double-clicking a 3D body.
const CLICK_HIT_RADIUS_3D_PX: float = 32.0;

## Default clear color matching the 3D space environment background.
const SPACE_BG_COLOR: Color = Color(0.03, 0.04, 0.07, 1.0);

## Maximum number of historical anchor-relative positions retained per runtime ballistic trail.
const MAX_BALLISTIC_TRAIL_POINTS: int = 260;

## Minimum squared visual 3D distance before appending a new trail vertex.
const MIN_BALLISTIC_TRAIL_STEP_SQ: float = 0.02;

#region Private Members
## Currently active view mode ([constant ViewMode.VIEW_2D] or [constant ViewMode.VIEW_3D]).
var _current_view_mode: ViewMode = ViewMode.VIEW_3D;

## Index of the currently active showcase scenario in [member scenario_scenes].
var _current_scenario_index: int = 0;

## Root node of the currently instantiated scenario sub-scene, if loaded via [member scenario_scenes].
var _active_scenario_node: Node = null;

## Whether the user is currently panning the 2D camera by holding a mouse button.
var _is_panning_2d: bool = false;

## Whether the user is currently orbiting the 3D camera by holding the right mouse button.
var _is_orbiting_3d: bool = false;

## Current yaw angle (in radians) of the 3D camera pivot.
var _camera_yaw_3d: float = 0.0;

## Current pitch angle (in radians) of the 3D camera pivot.
var _camera_pitch_3d: float = -0.55;

## Current distance of [member camera_3d] along the 3D pivot's local +Z axis.
var _camera_distance_3d: float = 68.0;

## Runtime [Line2D] nodes rendering the 2D orbital paths of [member selectable_bodies_2d].
var _orbit_lines_2d: Dictionary[JrOrbiter2D, Line2D] = {};

## Runtime [MeshInstance3D] nodes rendering the 3D orbital paths of [member selectable_bodies_3d].
var _orbit_meshes_3d: Dictionary[JrOrbiter3D, MeshInstance3D] = {};

## Recorded anchor-relative trail samples (xyz = anchor-rel pos, w = sim time) per [JrBallisticOrbiter].
var _ballistic_trails: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {};

## Anchor element associated with the recorded samples in [member _ballistic_trails].
var _ballistic_trail_anchors: Dictionary[JrBallisticOrbiter, JrOrreryElement] = {};

## Runtime [Line2D] nodes rendering live 2D historical trails behind [JrBallisticOrbiter] followers.
var _ballistic_lines_2d: Dictionary[JrOrbiter2D, Line2D] = {};

## Runtime [MeshInstance3D] nodes rendering live 3D historical trails behind [JrBallisticOrbiter] followers.
var _ballistic_meshes_3d: Dictionary[JrOrbiter3D, MeshInstance3D] = {};

## Runtime [Node2D] canvas nodes rendering dashed 2D forward trajectory forecasts ahead of [JrBallisticOrbiter] followers.
var _ballistic_forecast_canvases_2d: Dictionary[JrOrbiter2D, Node2D] = {};

## Runtime [MeshInstance3D] nodes rendering dashed 3D forward trajectory forecasts ahead of [JrBallisticOrbiter] followers.
var _ballistic_forecast_meshes_3d: Dictionary[JrOrbiter3D, MeshInstance3D] = {};
#endregion

#region Exports
## Initial view and camera mode activated when the demo scene starts.
@export
var initial_view_mode: ViewMode = ViewMode.VIEW_3D;

## Shared [JrOrbitDriver] simulation root driving both the 2D and 3D visual trees.
@export
var orbit_driver: JrOrbitDriver = null;

## If [code]true[/code], renders each orbiter's trajectory path at runtime in both 2D and 3D.
@export
var show_orbit_paths: bool = true:
	set(value):
		if show_orbit_paths != value:
			show_orbit_paths = value;
			_update_orbit_paths_visibility();

## If [code]true[/code], collapses the HUD instructions body down to a compact header bar.
@export
var instructions_collapsed: bool = false:
	set(value):
		if instructions_collapsed != value:
			instructions_collapsed = value;
			_apply_instructions_collapsed_state();

@export_group("Scenarios")
## Showcase scenario sub-scenes available for selection in the demo launcher.
@export
var scenario_scenes: Array[PackedScene] = [];

## Display titles corresponding to each entry in [member scenario_scenes].
@export
var scenario_titles: PackedStringArray = [];

## Overview descriptions corresponding to each entry in [member scenario_scenes].
@export
var scenario_descriptions: PackedStringArray = [];

@export_group("2D View")
## Root [Node2D] container holding the 2D visual hierarchy and [Camera2D].
@export
var view_2d_root: Node2D = null;

## Active [Camera2D] used when [constant ViewMode.VIEW_2D] is selected.
@export
var camera_2d: Camera2D = null;

## Visual [Node2D] representing the central star or system origin in the 2D view.
@export
var central_star_2d: Node2D = null;

## Orbiting 2D visual bodies that can be double-clicked in 2D view to become the active reference perspective.
@export
var selectable_bodies_2d: Array[JrOrbiter2D] = [];

@export_group("3D View")
## Root [Node3D] container holding the 3D visual hierarchy and [Camera3D] rig.
@export
var view_3d_root: Node3D = null;

## Pivot [Node3D] around which [member camera_3d] orbits the selected body.
@export
var camera_pivot_3d: Node3D = null;

## Active [Camera3D] used when [constant ViewMode.VIEW_3D] is selected.
@export
var camera_3d: Camera3D = null;

## Visual [Node3D] representing the central star or system origin in the 3D view.
@export
var central_star_3d: Node3D = null;

## Orbiting 3D visual bodies that can be double-clicked in 3D view to become the active reference perspective.
@export
var selectable_bodies_3d: Array[JrOrbiter3D] = [];

@export_group("UI References")
## Optional root [PanelContainer] wrapping the HUD info and instructions panel.
@export
var info_panel: PanelContainer = null;

## Optional [Control] container holding the collapsible HUD instructions content.
@export
var instructions_container: Control = null;

## Optional UI [Button] allowing the user to collapse or expand the instructions panel.
@export
var collapse_button: Button = null;

## Optional UI [Label] displaying the demo title for the active view mode.
@export
var title_label: Label = null;

## Optional UI [OptionButton] allowing the user to switch between showcase scenarios.
@export
var scenario_selector: OptionButton = null;

## Optional UI [Label] displaying the description of the active showcase scenario.
@export
var overview_label: Label = null;

## Optional UI [Label] displaying the currently active camera type and projection mode.
@export
var camera_status_label: Label = null;

## Optional UI [Label] listing the controls for the currently active view mode.
@export
var controls_label: Label = null;

## Optional UI [Label] displaying the name of the currently active reference body and lock mode.
@export
var reference_status_label: Label = null;

## Optional UI [Button] allowing the user to transfer between 2D and 3D views.
@export
var switch_view_button: Button = null;
#endregion

#region Lifecycle & Frame Stepping
## Resolves node references, populates the scenario selector, initializes the 3D camera rig,
## builds runtime 2D and 3D orbital path renderers, connects UI signals, and activates [member initial_view_mode].
func _ready() -> void:
	process_priority = 1;
	RenderingServer.set_default_clear_color(SPACE_BG_COLOR);
	_ensure_ui_references();
	_populate_scenario_selector();
	_ensure_node_references();
	_init_3d_camera_rig();
	_build_2d_orbit_path_lines();
	_build_3d_orbit_path_meshes();
	_init_ballistic_trail_renderers();
	_init_ballistic_forecast_renderers();
	_connect_ui_signals();
	_apply_instructions_collapsed_state();
	_apply_view_mode(initial_view_mode, false);

## Synchronizes the active view's runtime orbital paths, ballistic trails, forecasts, and camera rig each frame.
func _process(_delta: float) -> void:
	_record_ballistic_trail_points();
	if _current_view_mode == ViewMode.VIEW_2D:
		_sync_2d_orbit_line_positions();
		_sync_2d_ballistic_trails();
		_sync_2d_ballistic_forecasts();
	else:
		_sync_3d_camera_pivot_to_selected_body();
		_sync_3d_orbit_mesh_transforms();
		_sync_3d_ballistic_trails();
		_sync_3d_ballistic_forecasts();

## Intercepts keyboard shortcuts before GUI focus navigation so Tab, V, R, H, and 1-5
## work reliably regardless of UI focus state.
func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		_handle_key_input(event as InputEventKey);

## Processes unhandled mouse events for camera control and body selection.
func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		_handle_mouse_button(event as InputEventMouseButton);
	elif event is InputEventMouseMotion:
		_handle_mouse_motion(event as InputEventMouseMotion);
#endregion

#region Scenario Management
## Populates [member scenario_selector] with [member scenario_titles] if present.
func _populate_scenario_selector() -> void:
	if scenario_selector == null:
		return;
	scenario_selector.clear();
	var count: int = scenario_scenes.size();
	for i: int in range(count):
		var label_text: String = scenario_titles[i] if i < scenario_titles.size() else ("Scenario %d" % (i + 1));
		scenario_selector.add_item(label_text, i);
	if count > 0:
		scenario_selector.selected = clampi(_current_scenario_index, 0, count - 1);

## Loads and activates the showcase scenario at [param index] in [member scenario_scenes].
func load_scenario(index: int) -> void:
	if index < 0 or index >= scenario_scenes.size():
		return;
	var packed_scene: PackedScene = scenario_scenes[index];
	if packed_scene == null:
		return;

	_current_scenario_index = index;
	_cleanup_runtime_visual_renderers();

	var prev_scenario: Node = _active_scenario_node;
	if prev_scenario != null and is_instance_valid(prev_scenario):
		prev_scenario.name = "PreviousScenario";
		var prev_driver: JrOrbitDriver = prev_scenario.get_node_or_null("JrOrbitDriver") as JrOrbitDriver;
		if prev_driver != null:
			prev_driver.begin_batch_updates();
	_active_scenario_node = null;

	var instance: Node = packed_scene.instantiate();
	instance.name = "ActiveScenario";
	add_child(instance);
	_active_scenario_node = instance;

	if prev_scenario != null and is_instance_valid(prev_scenario):
		remove_child(prev_scenario);
		prev_scenario.queue_free();
	_remove_existing_scenario_nodes();

	_clear_bound_scene_references();
	_ensure_node_references();
	_init_3d_camera_rig();
	_build_2d_orbit_path_lines();
	_build_3d_orbit_path_meshes();
	_init_ballistic_trail_renderers();
	_init_ballistic_forecast_renderers();
	_apply_view_mode(_current_view_mode, false);

	if scenario_selector != null and scenario_selector.selected != index:
		scenario_selector.selected = index;

## Clears all dynamically created 2D/3D orbit lines, ballistic wake trails, and forecast nodes.
func _cleanup_runtime_visual_renderers() -> void:
	for body_2d: JrOrbiter2D in _orbit_lines_2d:
		var line: Line2D = _orbit_lines_2d[body_2d];
		if line != null and is_instance_valid(line):
			line.queue_free();
	_orbit_lines_2d.clear();

	for body_3d: JrOrbiter3D in _orbit_meshes_3d:
		var mesh_inst: MeshInstance3D = _orbit_meshes_3d[body_3d];
		if mesh_inst != null and is_instance_valid(mesh_inst):
			mesh_inst.queue_free();
	_orbit_meshes_3d.clear();

	for b_2d: JrOrbiter2D in _ballistic_lines_2d:
		var b_line: Line2D = _ballistic_lines_2d[b_2d];
		if b_line != null and is_instance_valid(b_line):
			b_line.queue_free();
	_ballistic_lines_2d.clear();

	for b_3d: JrOrbiter3D in _ballistic_meshes_3d:
		var b_mesh: MeshInstance3D = _ballistic_meshes_3d[b_3d];
		if b_mesh != null and is_instance_valid(b_mesh):
			b_mesh.queue_free();
	_ballistic_meshes_3d.clear();

	for f_2d: JrOrbiter2D in _ballistic_forecast_canvases_2d:
		var canvas: Node2D = _ballistic_forecast_canvases_2d[f_2d];
		if canvas != null and is_instance_valid(canvas):
			canvas.queue_free();
	_ballistic_forecast_canvases_2d.clear();

	for f_3d: JrOrbiter3D in _ballistic_forecast_meshes_3d:
		var f_mesh: MeshInstance3D = _ballistic_forecast_meshes_3d[f_3d];
		if f_mesh != null and is_instance_valid(f_mesh):
			f_mesh.queue_free();
	_ballistic_forecast_meshes_3d.clear();

	_ballistic_trails.clear();
	_ballistic_trail_anchors.clear();

## Removes and frees any direct non-active simulation/view nodes before loading a new scenario.
func _remove_existing_scenario_nodes() -> void:
	for child: Node in get_children():
		if child != _active_scenario_node and child.name.begins_with("ActiveScenario"):
			remove_child(child);
			child.queue_free();

	var direct_driver: Node = get_node_or_null("JrOrbitDriver");
	if direct_driver != null:
		remove_child(direct_driver);
		direct_driver.queue_free();

	var direct_3d: Node = get_node_or_null("View3D");
	if direct_3d != null:
		remove_child(direct_3d);
		direct_3d.queue_free();

	var direct_2d: Node = get_node_or_null("View2D");
	if direct_2d != null:
		remove_child(direct_2d);
		direct_2d.queue_free();

## Resets cached simulation and 2D/3D view references prior to re-binding a newly loaded scenario.
func _clear_bound_scene_references() -> void:
	orbit_driver = null;
	view_2d_root = null;
	camera_2d = null;
	central_star_2d = null;
	selectable_bodies_2d.clear();
	view_3d_root = null;
	camera_pivot_3d = null;
	camera_3d = null;
	central_star_3d = null;
	selectable_bodies_3d.clear();

## Callback invoked when the user selects a scenario in [member scenario_selector].
func _on_scenario_selected(index: int) -> void:
	if index != _current_scenario_index:
		load_scenario(index);
#endregion

#region Reference Resolution Helpers
## Returns the active scenario container node (`ActiveScenario` if present, otherwise `self`).
func _get_scenario_search_root() -> Node:
	if _active_scenario_node != null and is_instance_valid(_active_scenario_node):
		return _active_scenario_node;
	var existing_active: Node = get_node_or_null("ActiveScenario");
	if existing_active != null:
		_active_scenario_node = existing_active;
		return existing_active;
	return self;

## Ensures all exported node references are populated, falling back to scene tree
## paths if any export was not serialized.
func _ensure_node_references() -> void:
	var search_root: Node = _get_scenario_search_root();
	if orbit_driver == null or not is_instance_valid(orbit_driver):
		orbit_driver = search_root.get_node_or_null("JrOrbitDriver") as JrOrbitDriver;
	_ensure_2d_references(search_root);
	_ensure_3d_references(search_root);
	_ensure_ui_references();

## Resolves fallback 2D view node references under [param search_root] if unassigned.
func _ensure_2d_references(search_root: Node) -> void:
	if view_2d_root == null or not is_instance_valid(view_2d_root):
		view_2d_root = search_root.get_node_or_null("View2D") as Node2D;
	if view_2d_root == null:
		return;
	if camera_2d == null or not is_instance_valid(camera_2d):
		camera_2d = view_2d_root.get_node_or_null("Camera2D") as Camera2D;
	if central_star_2d == null or not is_instance_valid(central_star_2d):
		central_star_2d = view_2d_root.get_node_or_null("CentralStar") as Node2D;
	if selectable_bodies_2d.is_empty():
		_collect_descendant_orbiters_2d(view_2d_root);

## Recursively collects descendant [JrOrbiter2D] nodes under [param root_node].
func _collect_descendant_orbiters_2d(root_node: Node) -> void:
	for child: Node in root_node.get_children():
		if child is JrOrbiter2D:
			var orb_2d: JrOrbiter2D = child as JrOrbiter2D;
			if orb_2d != central_star_2d and not orb_2d.has_meta(&"swarm_background"):
				selectable_bodies_2d.append(orb_2d);
		_collect_descendant_orbiters_2d(child);

## Resolves fallback 3D view node references under [param search_root] if unassigned.
func _ensure_3d_references(search_root: Node) -> void:
	if view_3d_root == null or not is_instance_valid(view_3d_root):
		view_3d_root = search_root.get_node_or_null("View3D") as Node3D;
	if view_3d_root == null:
		return;
	if camera_pivot_3d == null or not is_instance_valid(camera_pivot_3d):
		camera_pivot_3d = view_3d_root.get_node_or_null("CameraPivot") as Node3D;
	if camera_3d == null or not is_instance_valid(camera_3d):
		camera_3d = view_3d_root.get_node_or_null("CameraPivot/Camera3D") as Camera3D;
	if central_star_3d == null or not is_instance_valid(central_star_3d):
		central_star_3d = view_3d_root.get_node_or_null("CentralStar") as Node3D;
	if selectable_bodies_3d.is_empty():
		_collect_descendant_orbiters_3d(view_3d_root);

## Recursively collects descendant [JrOrbiter3D] nodes under [param root_node].
func _collect_descendant_orbiters_3d(root_node: Node) -> void:
	for child: Node in root_node.get_children():
		if child is JrOrbiter3D:
			var orb_3d: JrOrbiter3D = child as JrOrbiter3D;
			if orb_3d != central_star_3d and not orb_3d.has_meta(&"swarm_background"):
				selectable_bodies_3d.append(orb_3d);
		_collect_descendant_orbiters_3d(child);

## Resolves fallback UI container, label, and button references if unassigned.
func _ensure_ui_references() -> void:
	var panel_path: String = "UILayer/HUDRoot/InfoPanel";
	var vbox_path: String = panel_path + "/MarginContainer/VBoxContainer/";
	_ensure_ui_container_references(panel_path, vbox_path);
	_ensure_ui_content_references(vbox_path + "InstructionsContainer/");

## Resolves fallback HUD panel, header title, and collapse button references.
func _ensure_ui_container_references(panel_path: String, vbox_path: String) -> void:
	if info_panel == null:
		info_panel = get_node_or_null(panel_path) as PanelContainer;
	if instructions_container == null:
		instructions_container = get_node_or_null(vbox_path + "InstructionsContainer") as Control;
	if title_label == null:
		title_label = get_node_or_null(vbox_path + "HeaderRow/TitleLabel") as Label;
	if collapse_button == null:
		collapse_button = get_node_or_null(vbox_path + "HeaderRow/CollapseButton") as Button;

## Resolves fallback HUD detail labels and buttons inside [param details_path].
func _ensure_ui_content_references(details_path: String) -> void:
	if scenario_selector == null:
		scenario_selector = get_node_or_null(details_path + "ScenarioRow/ScenarioSelector") as OptionButton;
	if overview_label == null:
		overview_label = get_node_or_null(details_path + "OverviewLabel") as Label;
	if camera_status_label == null:
		camera_status_label = get_node_or_null(details_path + "CameraStatusLabel") as Label;
	if controls_label == null:
		controls_label = get_node_or_null(details_path + "ControlsLabel") as Label;
	if reference_status_label == null:
		reference_status_label = get_node_or_null(details_path + "ReferenceStatusLabel") as Label;
	if switch_view_button == null:
		switch_view_button = get_node_or_null(details_path + "SwitchViewButton") as Button;
#endregion

#region View Transfer & Camera Switching
## Returns the currently active [enum ViewMode].
func get_view_mode() -> ViewMode:
	return _current_view_mode;

## Toggles between [constant ViewMode.VIEW_2D] ([Camera2D]) and [constant ViewMode.VIEW_3D] ([Camera3D]).
func toggle_view_mode() -> void:
	var next_mode: ViewMode = ViewMode.VIEW_2D if _current_view_mode == ViewMode.VIEW_3D else ViewMode.VIEW_3D;
	set_view_mode(next_mode);

## Transfers the demo to [param mode], switching the active camera type and
## synchronizing camera framing around the selected reference body.
func set_view_mode(mode: ViewMode) -> void:
	if _current_view_mode == mode:
		return;
	_apply_view_mode(mode, true);

## Applies [param mode], toggling 2D/3D visibility and switching between [Camera2D] and [Camera3D].
func _apply_view_mode(mode: ViewMode, sync_zoom: bool) -> void:
	_current_view_mode = mode;
	_is_panning_2d = false;
	_is_orbiting_3d = false;

	if mode == ViewMode.VIEW_2D:
		_activate_2d_view(sync_zoom);
	else:
		_activate_3d_view(sync_zoom);

	_update_hud();

## Enables the 2D view hierarchy and switches the active camera to [member camera_2d].
func _activate_2d_view(sync_zoom: bool) -> void:
	if sync_zoom and camera_2d != null and _camera_distance_3d > 0.0:
		var target_zoom: float = clampf(ZOOM_DISTANCE_PRODUCT / _camera_distance_3d, MIN_ZOOM_2D, MAX_ZOOM_2D);
		camera_2d.zoom = Vector2(target_zoom, target_zoom);

	if view_3d_root != null:
		view_3d_root.visible = false;
	if view_2d_root != null:
		view_2d_root.visible = true;

	if camera_3d != null:
		camera_3d.current = false;
	if camera_2d != null:
		camera_2d.enabled = true;
		camera_2d.make_current();

	_sync_2d_orbit_line_positions();
	_sync_2d_ballistic_trails();
	_sync_2d_ballistic_forecasts();
	_sync_2d_camera_to_selected_body();

## Enables the 3D view hierarchy and switches the active camera to [member camera_3d].
func _activate_3d_view(sync_zoom: bool) -> void:
	if sync_zoom and camera_2d != null and camera_2d.zoom.x > 0.0:
		_camera_distance_3d = clampf(
			ZOOM_DISTANCE_PRODUCT / camera_2d.zoom.x,
			MIN_CAMERA_DISTANCE_3D,
			MAX_CAMERA_DISTANCE_3D
		);
		_apply_3d_camera_rig_transform();

	if view_2d_root != null:
		view_2d_root.visible = false;
	if view_3d_root != null:
		view_3d_root.visible = true;

	if camera_2d != null:
		camera_2d.enabled = false;
	if camera_3d != null:
		camera_3d.current = true;
		camera_3d.make_current();

	_sync_3d_camera_pivot_to_selected_body();
	_sync_3d_orbit_mesh_transforms();
	_sync_3d_ballistic_trails();
	_sync_3d_ballistic_forecasts();

## Connects UI button signals for view switching, collapsing the instructions panel, and scenario selection.
func _connect_ui_signals() -> void:
	if switch_view_button != null and not switch_view_button.pressed.is_connected(toggle_view_mode):
		switch_view_button.pressed.connect(toggle_view_mode);
	if collapse_button != null and not collapse_button.pressed.is_connected(toggle_instructions_collapsed):
		collapse_button.pressed.connect(toggle_instructions_collapsed);
	if scenario_selector != null and not scenario_selector.item_selected.is_connected(_on_scenario_selected):
		scenario_selector.item_selected.connect(_on_scenario_selected);

## Toggles whether the HUD instructions content is collapsed or expanded.
func toggle_instructions_collapsed() -> void:
	set_instructions_collapsed(not instructions_collapsed);

## Sets whether the HUD instructions content is collapsed and shrinks or expands [member info_panel].
func set_instructions_collapsed(collapsed: bool) -> void:
	instructions_collapsed = collapsed;

## Applies [member instructions_collapsed] to [member instructions_container], updates
## [member collapse_button], and resizes [member info_panel] to fit its visible children.
func _apply_instructions_collapsed_state() -> void:
	if instructions_container != null:
		instructions_container.visible = not instructions_collapsed;
	if collapse_button != null:
		collapse_button.text = "▼ Instructions [H]" if instructions_collapsed else "▲ Hide [H]";
	if info_panel != null:
		info_panel.reset_size();
		info_panel.reset_size.call_deferred();
#endregion

#region Input Dispatch & Keyboard Shortcuts
## Handles keyboard shortcuts for transferring views (Tab / V), cycling perspective
## rotation mode (R), collapsing/expanding the instructions panel (H), and selecting scenarios (1-5).
func _handle_key_input(event: InputEventKey) -> void:
	if not event.pressed or event.echo:
		return;

	var key: Key = event.keycode if event.keycode != KEY_NONE else event.physical_keycode;
	if key == KEY_TAB or key == KEY_V:
		toggle_view_mode();
		get_viewport().set_input_as_handled();
		return;

	if key == KEY_H:
		toggle_instructions_collapsed();
		get_viewport().set_input_as_handled();
		return;

	if key == KEY_R and orbit_driver != null:
		var next_mode: int = (int(orbit_driver.perspective_rotation_mode) + 1) % 3;
		orbit_driver.perspective_rotation_mode = next_mode as JrOrbitDriver.PerspectiveRotationMode;
		_update_hud();
		get_viewport().set_input_as_handled();
		return;

	if key >= KEY_1 and key <= KEY_5:
		var target_idx: int = int(key - KEY_1);
		if target_idx < scenario_scenes.size():
			load_scenario(target_idx);
			get_viewport().set_input_as_handled();

## Routes mouse button events to the active 2D or 3D camera controller.
func _handle_mouse_button(event: InputEventMouseButton) -> void:
	if _current_view_mode == ViewMode.VIEW_2D:
		_handle_2d_mouse_button(event);
	else:
		_handle_3d_mouse_button(event);

## Routes mouse motion events to the active 2D or 3D camera controller.
func _handle_mouse_motion(event: InputEventMouseMotion) -> void:
	if _current_view_mode == ViewMode.VIEW_2D:
		_handle_2d_mouse_motion(event);
	else:
		_handle_3d_mouse_motion(event);
#endregion

#region 2D Camera & Selection Controls
## Handles 2D mouse wheel zoom, double-click body selection, and pan state transitions.
func _handle_2d_mouse_button(event: InputEventMouseButton) -> void:
	if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
		_zoom_2d_camera(ZOOM_STEP);
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
		_zoom_2d_camera(1.0 / ZOOM_STEP);
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_LEFT:
		if event.double_click and event.pressed:
			_is_panning_2d = false;
			_select_2d_body_at_cursor();
			get_viewport().set_input_as_handled();
			return;
		_is_panning_2d = event.pressed;
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_MIDDLE or event.button_index == MOUSE_BUTTON_RIGHT:
		_is_panning_2d = event.pressed;
		get_viewport().set_input_as_handled();

## Translates [member camera_2d] opposite to mouse drag delta while 2D panning is active.
func _handle_2d_mouse_motion(event: InputEventMouseMotion) -> void:
	if not _is_panning_2d or camera_2d == null:
		return;
	camera_2d.position -= event.relative / camera_2d.zoom;
	get_viewport().set_input_as_handled();

## Zooms [member camera_2d] by [param factor] around the current 2D mouse cursor position.
func _zoom_2d_camera(factor: float) -> void:
	if camera_2d == null:
		return;
	var world_mouse_before: Vector2 = _get_2d_mouse_world_position();
	var target_zoom: float = clampf(camera_2d.zoom.x * factor, MIN_ZOOM_2D, MAX_ZOOM_2D);
	camera_2d.zoom = Vector2(target_zoom, target_zoom);
	var world_mouse_after: Vector2 = _get_2d_mouse_world_position();
	camera_2d.position += world_mouse_before - world_mouse_after;

## Finds the closest 2D body under the cursor and assigns its simulation element as
## [member JrOrbitDriver.perspective_node].
func _select_2d_body_at_cursor() -> void:
	if orbit_driver == null:
		return;

	var mouse_world_pos: Vector2 = _get_2d_mouse_world_position();
	var zoom_scale: float = camera_2d.zoom.x if camera_2d != null and camera_2d.zoom.x > 0.0 else 1.0;
	var max_world_distance: float = CLICK_HIT_RADIUS_2D_PX / zoom_scale;

	var closest_body: JrOrbiter2D = _find_closest_2d_body(mouse_world_pos, max_world_distance);
	if closest_body != null and closest_body.get_target_trajectory_element() != null:
		_set_perspective_orbiter(closest_body.get_target_trajectory_element());
		return;

	var star_pos: Vector2 = _get_central_star_2d_world_position();
	if star_pos.distance_to(mouse_world_pos) <= max_world_distance:
		_set_perspective_orbiter(null);

## Returns the mouse cursor position in 2D canvas world coordinates.
func _get_2d_mouse_world_position() -> Vector2:
	if view_2d_root != null and view_2d_root.is_inside_tree():
		return view_2d_root.get_global_mouse_position();
	if camera_2d != null and camera_2d.is_inside_tree():
		return camera_2d.get_global_mouse_position();
	return Vector2.ZERO;

## Returns the 2D world position of [member central_star_2d].
func _get_central_star_2d_world_position() -> Vector2:
	if central_star_2d != null and central_star_2d.is_inside_tree():
		return central_star_2d.global_position;
	return Vector2.ZERO;

## Returns the [JrOrbiter2D] in [member selectable_bodies_2d] closest to [param world_pos]
## within [param max_distance], or [code]null[/code] if none is within range.
func _find_closest_2d_body(world_pos: Vector2, max_distance: float) -> JrOrbiter2D:
	var best_body: JrOrbiter2D = null;
	var best_distance: float = max_distance;

	for body: JrOrbiter2D in selectable_bodies_2d:
		if body == null or not body.is_inside_tree():
			continue;
		var dist: float = body.global_position.distance_to(world_pos);
		if dist <= best_distance:
			best_distance = dist;
			best_body = body;

	return best_body;

## Centers [member camera_2d] on the active [member JrOrbitDriver.perspective_node]'s
## 2D visual follower (or on [member central_star_2d] when in heliocentric mode).
func _sync_2d_camera_to_selected_body() -> void:
	if camera_2d == null or orbit_driver == null:
		return;
	if orbit_driver.perspective_node != null:
		var visual_2d: JrOrbiter2D = _find_2d_visual_for_orbiter(orbit_driver.perspective_node);
		if visual_2d != null and visual_2d.is_inside_tree():
			camera_2d.global_position = visual_2d.global_position;
			return;
	camera_2d.global_position = _get_central_star_2d_world_position();

## Returns the [JrOrbiter2D] in [member selectable_bodies_2d] referencing [param sim_element].
func _find_2d_visual_for_orbiter(sim_element: JrOrreryElement) -> JrOrbiter2D:
	for body: JrOrbiter2D in selectable_bodies_2d:
		if body != null and body.orbiter == sim_element:
			return body;
	return null;
#endregion

#region 3D Camera & Selection Controls
## Captures the initial yaw, pitch, and distance from [member camera_pivot_3d] and [member camera_3d].
func _init_3d_camera_rig() -> void:
	if camera_pivot_3d != null:
		var euler: Vector3 = camera_pivot_3d.rotation;
		_camera_pitch_3d = euler.x;
		_camera_yaw_3d = euler.y;
	if camera_3d != null:
		_camera_distance_3d = clampf(camera_3d.position.z, MIN_CAMERA_DISTANCE_3D, MAX_CAMERA_DISTANCE_3D);
	_apply_3d_camera_rig_transform();

## Applies [member _camera_yaw_3d], [member _camera_pitch_3d], and [member _camera_distance_3d]
## to [member camera_pivot_3d] and [member camera_3d].
func _apply_3d_camera_rig_transform() -> void:
	if camera_pivot_3d != null:
		camera_pivot_3d.rotation = Vector3(_camera_pitch_3d, _camera_yaw_3d, 0.0);
	if camera_3d != null:
		camera_3d.position = Vector3(0.0, 0.0, _camera_distance_3d);

## Handles 3D mouse wheel zoom, double-click body selection, and right-click camera orbit state.
func _handle_3d_mouse_button(event: InputEventMouseButton) -> void:
	if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
		_zoom_3d_camera(1.0 / ZOOM_STEP);
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
		_zoom_3d_camera(ZOOM_STEP);
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_LEFT and event.double_click and event.pressed:
		_select_3d_body_at_cursor(event.position);
		get_viewport().set_input_as_handled();
		return;

	if event.button_index == MOUSE_BUTTON_RIGHT:
		_is_orbiting_3d = event.pressed;
		get_viewport().set_input_as_handled();

## Orbits the 3D camera rig around the selected body based on right-click mouse drag delta.
func _handle_3d_mouse_motion(event: InputEventMouseMotion) -> void:
	if not _is_orbiting_3d:
		return;
	_camera_yaw_3d -= event.relative.x * ORBIT_SENSITIVITY_3D;
	_camera_pitch_3d = clampf(_camera_pitch_3d - event.relative.y * ORBIT_SENSITIVITY_3D, -PI * 0.48, PI * 0.48);
	_apply_3d_camera_rig_transform();
	get_viewport().set_input_as_handled();

## Scales [member _camera_distance_3d] by [param factor] within the configured distance bounds.
func _zoom_3d_camera(factor: float) -> void:
	_camera_distance_3d = clampf(_camera_distance_3d * factor, MIN_CAMERA_DISTANCE_3D, MAX_CAMERA_DISTANCE_3D);
	_apply_3d_camera_rig_transform();

## Finds the closest 3D body under [param screen_pos] and assigns its simulation element as
## [member JrOrbitDriver.perspective_node].
func _select_3d_body_at_cursor(screen_pos: Vector2) -> void:
	if orbit_driver == null or camera_3d == null:
		return;

	var closest_body: JrOrbiter3D = _find_closest_3d_body_on_screen(screen_pos);
	if closest_body != null and closest_body.get_target_trajectory_element() != null:
		_set_perspective_orbiter(closest_body.get_target_trajectory_element());
		return;

	var star_world_pos: Vector3 = _get_central_star_3d_world_position();
	if not camera_3d.is_position_behind(star_world_pos):
		var star_screen: Vector2 = camera_3d.unproject_position(star_world_pos);
		if star_screen.distance_to(screen_pos) <= CLICK_HIT_RADIUS_3D_PX:
			_set_perspective_orbiter(null);

## Returns the 3D world position of [member central_star_3d].
func _get_central_star_3d_world_position() -> Vector3:
	if central_star_3d != null and central_star_3d.is_inside_tree():
		return central_star_3d.global_position;
	return Vector3.ZERO;

## Keeps [member camera_pivot_3d] positioned at the active [member JrOrbitDriver.perspective_node]'s
## 3D visual follower (or at [member central_star_3d] when in heliocentric mode).
func _sync_3d_camera_pivot_to_selected_body() -> void:
	if camera_pivot_3d == null or orbit_driver == null:
		return;
	if orbit_driver.perspective_node != null:
		var visual_3d: JrOrbiter3D = _find_3d_visual_for_orbiter(orbit_driver.perspective_node);
		if visual_3d != null and visual_3d.is_inside_tree():
			camera_pivot_3d.global_position = visual_3d.global_position;
			return;
	camera_pivot_3d.global_position = _get_central_star_3d_world_position();

## Returns the [JrOrbiter3D] in [member selectable_bodies_3d] referencing [param sim_element].
func _find_3d_visual_for_orbiter(sim_element: JrOrreryElement) -> JrOrbiter3D:
	for body: JrOrbiter3D in selectable_bodies_3d:
		if body != null and body.orbiter == sim_element:
			return body;
	return null;

## Returns the [JrOrbiter3D] in [member selectable_bodies_3d] whose projected screen
## position is closest to [param screen_pos] within [constant CLICK_HIT_RADIUS_3D_PX].
func _find_closest_3d_body_on_screen(screen_pos: Vector2) -> JrOrbiter3D:
	var best_body: JrOrbiter3D = null;
	var best_distance: float = CLICK_HIT_RADIUS_3D_PX;

	for body: JrOrbiter3D in selectable_bodies_3d:
		if body == null or not body.is_inside_tree():
			continue;
		if camera_3d.is_position_behind(body.global_position):
			continue;
		var body_screen: Vector2 = camera_3d.unproject_position(body.global_position);
		var dist: float = body_screen.distance_to(screen_pos);
		if dist <= best_distance:
			best_distance = dist;
			best_body = body;

	return best_body;

## Assigns [param target] as [member JrOrbitDriver.perspective_node] and synchronizes
## both the 2D and 3D cameras and UI status labels.
func _set_perspective_orbiter(target: JrOrreryElement) -> void:
	if orbit_driver == null:
		return;
	orbit_driver.perspective_node = target;
	_sync_2d_camera_to_selected_body();
	_sync_3d_camera_pivot_to_selected_body();
	_update_hud();
#endregion

#region Runtime 2D & 3D Orbit Path Rendering
## Creates a [Line2D] under each 2D rail orbiter's parent node to display its orbital path at runtime.
func _build_2d_orbit_path_lines() -> void:
	for body: JrOrbiter2D in selectable_bodies_2d:
		if body == null or not body.draw_orbit_in_editor or body.get_target_orbiter() == null:
			continue;
		var parent_node: Node = body.get_parent();
		if parent_node == null:
			continue;

		var line: Line2D = Line2D.new();
		line.z_index = -1;
		line.antialiased = true;
		line.width = body.editor_orbit_width;
		line.default_color = body.editor_orbit_color;
		line.visible = show_orbit_paths;
		_refresh_2d_orbit_line_points(body, line);
		parent_node.add_child(line);
		_orbit_lines_2d[body] = line;

		if not body.orbit_visuals_changed.is_connected(_on_2d_body_visuals_changed.bind(body)):
			body.orbit_visuals_changed.connect(_on_2d_body_visuals_changed.bind(body));

## Updates the polyline vertices and style of [param line] from [param body]'s referenced [JrRailOrbiter].
func _refresh_2d_orbit_line_points(body: JrOrbiter2D, line: Line2D) -> void:
	if body == null or line == null:
		return;
	var target_orbiter: JrRailOrbiter = body.get_target_orbiter();
	if target_orbiter == null:
		return;
	var ref_dir: JrReferenceDirection = body.get_reference_plane();
	if orbit_driver != null:
		ref_dir = orbit_driver.reference_direction;
	line.width = _compute_2d_line_width(body);
	line.default_color = body.editor_orbit_color;
	line.points = target_orbiter.get_orbit_path_points_2d(
		ref_dir,
		body.editor_orbit_segments,
		body.position_scale
	);

## Computes a zoom-compensated width for [param body]'s 2D orbit line so it remains
## crisp across camera zoom levels.
func _compute_2d_line_width(body: JrOrbiter2D) -> float:
	var zoom_scale: float = camera_2d.zoom.x if camera_2d != null and camera_2d.zoom.x > 0.0 else 1.0;
	return body.editor_orbit_width / clampf(zoom_scale, 0.08, 4.0);

## Keeps each runtime [Line2D] centered on its 2D body's parent focus and updates its width.
func _sync_2d_orbit_line_positions() -> void:
	var sys_basis: Basis = orbit_driver.get_system_transform().basis if orbit_driver != null else Basis.IDENTITY;
	for body: JrOrbiter2D in _orbit_lines_2d:
		var line: Line2D = _orbit_lines_2d[body];
		var target_orbiter: JrRailOrbiter = body.get_target_orbiter() if body != null else null;
		if line == null or body == null or target_orbiter == null:
			continue;
		line.width = _compute_2d_line_width(body);
		var parent_rot: float = 0.0;
		if body.get_parent() is JrOrbiter2D and (body.get_parent() as JrOrbiter2D).orbiter is not JrOrbitDriver:
			parent_rot = (body.get_parent() as Node2D).rotation;
		if body.is_using_local_transform():
			line.position = Vector2.ZERO;
			line.rotation = -parent_rot;
		else:
			var focus_offset_2d: Vector2 = body.project_3d_to_2d(sys_basis * target_orbiter.get_focus_relative_position());
			line.position = body.position - focus_offset_2d;
			line.rotation = atan2(sys_basis.x.z, sys_basis.x.x);

## Refreshes a 2D body's runtime [Line2D] when its orbital parameters or visuals change.
func _on_2d_body_visuals_changed(body: JrOrbiter2D) -> void:
	var line: Line2D = _orbit_lines_2d.get(body, null);
	if line != null:
		_refresh_2d_orbit_line_points(body, line);

## Creates a [MeshInstance3D] under each 3D rail orbiter's parent focus to display its
## 3D orbital path at runtime.
func _build_3d_orbit_path_meshes() -> void:
	for body: JrOrbiter3D in selectable_bodies_3d:
		if body == null or not body.draw_orbit_in_editor or body.get_target_orbiter() == null:
			continue;
		var parent_focus: Node = body.get_parent();
		if parent_focus == null:
			continue;

		var mesh_instance: MeshInstance3D = MeshInstance3D.new();
		mesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
		mesh_instance.visible = show_orbit_paths;

		var mat: StandardMaterial3D = StandardMaterial3D.new();
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED;
		mat.vertex_color_use_as_albedo = true;
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA;
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED;
		mesh_instance.material_override = mat;

		_refresh_3d_orbit_mesh_points(body, mesh_instance);
		parent_focus.add_child(mesh_instance);
		_orbit_meshes_3d[body] = mesh_instance;

		if not body.orbit_visuals_changed.is_connected(_on_3d_body_visuals_changed.bind(body)):
			body.orbit_visuals_changed.connect(_on_3d_body_visuals_changed.bind(body));

	_sync_3d_orbit_mesh_transforms();

## Rebuilds the [ImmediateMesh] line strip for [param body]'s runtime 3D orbit path.
func _refresh_3d_orbit_mesh_points(body: JrOrbiter3D, mesh_instance: MeshInstance3D) -> void:
	if body == null or mesh_instance == null:
		return;
	var target_orbiter: JrRailOrbiter = body.get_target_orbiter();
	if target_orbiter == null:
		return;
	var ref_dir: JrReferenceDirection = body.get_reference_plane();
	if orbit_driver != null:
		ref_dir = orbit_driver.reference_direction;

	var points: PackedVector3Array = target_orbiter.get_orbit_path_points(ref_dir, body.editor_orbit_segments);
	var imm_mesh: ImmediateMesh = mesh_instance.mesh as ImmediateMesh;
	if imm_mesh == null:
		imm_mesh = ImmediateMesh.new();
		mesh_instance.mesh = imm_mesh;
	imm_mesh.clear_surfaces();
	if points.size() < 2:
		return;

	var pos_scale: float = body.position_scale;
	imm_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP);
	imm_mesh.surface_set_color(body.editor_orbit_color);
	for pt: Vector3 in points:
		imm_mesh.surface_add_vertex(pt * pos_scale);
	imm_mesh.surface_end();

## Aligns runtime 3D orbit path meshes with their parent focus in both nested and flat hierarchies.
func _sync_3d_orbit_mesh_transforms() -> void:
	for body: JrOrbiter3D in _orbit_meshes_3d:
		var mesh_instance: MeshInstance3D = _orbit_meshes_3d[body];
		if mesh_instance == null or body == null:
			continue;
		if body.is_using_local_transform():
			mesh_instance.position = Vector3.ZERO;
			mesh_instance.basis = body.get_parent_driver_basis().transposed();
		else:
			mesh_instance.global_transform = body.get_focus_to_world_transform();

## Refreshes a 3D body's runtime orbit mesh when its orbital parameters or visuals change.
func _on_3d_body_visuals_changed(body: JrOrbiter3D) -> void:
	var mesh_instance: MeshInstance3D = _orbit_meshes_3d.get(body, null);
	if mesh_instance != null:
		_refresh_3d_orbit_mesh_points(body, mesh_instance);

## Synchronizes visibility of all runtime 2D and 3D orbital path, historical trail,
## and forward forecast renderers with [member show_orbit_paths].
func _update_orbit_paths_visibility() -> void:
	for body_2d: JrOrbiter2D in _orbit_lines_2d:
		var line: Line2D = _orbit_lines_2d[body_2d];
		if line != null:
			line.visible = show_orbit_paths;

	for body_3d: JrOrbiter3D in _orbit_meshes_3d:
		var mesh_instance: MeshInstance3D = _orbit_meshes_3d[body_3d];
		if mesh_instance != null:
			mesh_instance.visible = show_orbit_paths;

	for ballistic_2d: JrOrbiter2D in _ballistic_lines_2d:
		var trail_line: Line2D = _ballistic_lines_2d[ballistic_2d];
		if trail_line != null:
			trail_line.visible = show_orbit_paths;

	for ballistic_3d: JrOrbiter3D in _ballistic_meshes_3d:
		var trail_mesh: MeshInstance3D = _ballistic_meshes_3d[ballistic_3d];
		if trail_mesh != null:
			trail_mesh.visible = show_orbit_paths;

	for forecast_2d: JrOrbiter2D in _ballistic_forecast_canvases_2d:
		var canvas_2d: Node2D = _ballistic_forecast_canvases_2d[forecast_2d];
		if canvas_2d != null:
			canvas_2d.visible = show_orbit_paths;

	for forecast_3d: JrOrbiter3D in _ballistic_forecast_meshes_3d:
		var forecast_mesh: MeshInstance3D = _ballistic_forecast_meshes_3d[forecast_3d];
		if forecast_mesh != null:
			forecast_mesh.visible = show_orbit_paths;

## Creates runtime [Line2D] and [MeshInstance3D] historical wake renderers for
## every selectable [JrBallisticOrbiter] body.
func _init_ballistic_trail_renderers() -> void:
	if view_2d_root != null:
		for body_2d: JrOrbiter2D in selectable_bodies_2d:
			var ballistic_2d: JrBallisticOrbiter = body_2d.get_target_ballistic() if body_2d != null else null;
			if ballistic_2d == null:
				continue;
			_ensure_ballistic_trail_buffer(ballistic_2d);
			var line: Line2D = Line2D.new();
			line.z_index = -1;
			line.antialiased = true;
			line.width = body_2d.editor_orbit_width;
			var col: Color = body_2d.editor_orbit_color;
			line.default_color = col;
			var grad: Gradient = Gradient.new();
			grad.set_color(0, Color(col.r, col.g, col.b, col.a * 0.05));
			grad.set_color(1, Color(col.r, col.g, col.b, col.a * 0.55));
			line.gradient = grad;
			line.visible = show_orbit_paths;
			view_2d_root.add_child(line);
			_ballistic_lines_2d[body_2d] = line;

	if view_3d_root != null:
		for body_3d: JrOrbiter3D in selectable_bodies_3d:
			var ballistic_3d: JrBallisticOrbiter = body_3d.get_target_ballistic() if body_3d != null else null;
			if ballistic_3d == null:
				continue;
			_ensure_ballistic_trail_buffer(ballistic_3d);
			var mesh_inst: MeshInstance3D = _create_unshaded_line_mesh_3d();
			mesh_inst.visible = show_orbit_paths;
			view_3d_root.add_child(mesh_inst);
			_ballistic_meshes_3d[body_3d] = mesh_inst;

## Creates runtime 2D and 3D dashed forward trajectory forecast renderers for
## every selectable [JrBallisticOrbiter] body.
func _init_ballistic_forecast_renderers() -> void:
	if view_2d_root != null:
		for body_2d: JrOrbiter2D in selectable_bodies_2d:
			var ballistic_2d: JrBallisticOrbiter = body_2d.get_target_ballistic() if body_2d != null else null;
			if ballistic_2d == null:
				continue;
			var canvas: Node2D = Node2D.new();
			canvas.z_index = -1;
			canvas.visible = show_orbit_paths;
			canvas.draw.connect(_on_draw_2d_forecast.bind(canvas, body_2d));
			view_2d_root.add_child(canvas);
			_ballistic_forecast_canvases_2d[body_2d] = canvas;

	if view_3d_root != null:
		for body_3d: JrOrbiter3D in selectable_bodies_3d:
			var ballistic_3d: JrBallisticOrbiter = body_3d.get_target_ballistic() if body_3d != null else null;
			if ballistic_3d == null:
				continue;
			var mesh_inst: MeshInstance3D = _create_unshaded_line_mesh_3d();
			mesh_inst.visible = show_orbit_paths;
			view_3d_root.add_child(mesh_inst);
			_ballistic_forecast_meshes_3d[body_3d] = mesh_inst;

## Instantiates a shadowless unshaded vertex-colored [MeshInstance3D] backed by an [ImmediateMesh].
func _create_unshaded_line_mesh_3d() -> MeshInstance3D:
	var mesh_inst: MeshInstance3D = MeshInstance3D.new();
	mesh_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	mesh_inst.mesh = ImmediateMesh.new();
	var mat: StandardMaterial3D = StandardMaterial3D.new();
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED;
	mat.vertex_color_use_as_albedo = true;
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA;
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED;
	mesh_inst.material_override = mat;
	return mesh_inst;

## Ensures [param ballistic] has an initialized anchor-relative trail buffer seeded with its start sample.
func _ensure_ballistic_trail_buffer(ballistic: JrBallisticOrbiter) -> void:
	if _ballistic_trails.has(ballistic):
		return;
	var p: Vector3 = ballistic.get_anchor_relative_position();
	var sim_t: float = orbit_driver.time if orbit_driver != null else 0.0;
	var initial_samples: PackedVector4Array = [Vector4(p.x, p.y, p.z, sim_t)];
	_ballistic_trails[ballistic] = initial_samples;
	_ballistic_trail_anchors[ballistic] = ballistic.get_anchor_element();

## Appends new timestamped anchor-relative positions for each tracked [JrBallisticOrbiter] body as it moves.
func _record_ballistic_trail_points() -> void:
	var sim_t: float = orbit_driver.time if orbit_driver != null else 0.0;
	for ballistic: JrBallisticOrbiter in _ballistic_trails.keys():
		if ballistic == null or not is_instance_valid(ballistic):
			continue;
		var curr_anchor: JrOrreryElement = ballistic.get_anchor_element();
		var prev_anchor: JrOrreryElement = _ballistic_trail_anchors.get(ballistic, null);
		var curr_pos: Vector3 = ballistic.get_anchor_relative_position();

		if curr_anchor != prev_anchor:
			_ballistic_trail_anchors[ballistic] = curr_anchor;
			_ballistic_trails[ballistic] = [Vector4(curr_pos.x, curr_pos.y, curr_pos.z, sim_t)];
			continue;

		var trail: PackedVector4Array = _ballistic_trails[ballistic];
		var pos_scale: float = _get_3d_scale_for_ballistic(ballistic);
		var should_append: bool = trail.is_empty();
		if not should_append:
			var last_s: Vector4 = trail[trail.size() - 1];
			var last_pos: Vector3 = Vector3(last_s.x, last_s.y, last_s.z);
			var scaled_dist_sq: float = last_pos.distance_squared_to(curr_pos) * (pos_scale * pos_scale);
			should_append = scaled_dist_sq >= MIN_BALLISTIC_TRAIL_STEP_SQ;
		if should_append:
			trail.append(Vector4(curr_pos.x, curr_pos.y, curr_pos.z, sim_t));
			if trail.size() > MAX_BALLISTIC_TRAIL_POINTS:
				trail.remove_at(0);
			_ballistic_trails[ballistic] = trail;

## Returns the 3D [member JrOrbiter3D.position_scale] associated with [param ballistic].
func _get_3d_scale_for_ballistic(ballistic: JrBallisticOrbiter) -> float:
	var visual_3d: JrOrbiter3D = _find_3d_visual_for_orbiter(ballistic);
	if visual_3d != null:
		return visual_3d.position_scale;
	return 1.0;

## Returns the 2D canvas coordinates of the active perspective origin in [member view_2d_root].
func _compute_2d_perspective_origin() -> Vector2:
	if orbit_driver != null and orbit_driver.perspective_node != null and orbit_driver.perspective_node != orbit_driver:
		var visual_2d: JrOrbiter2D = _find_2d_visual_for_orbiter(orbit_driver.perspective_node);
		if visual_2d != null and visual_2d.is_inside_tree() and view_2d_root != null:
			return view_2d_root.to_local(visual_2d.global_position);
	if central_star_2d != null and central_star_2d.is_inside_tree() and view_2d_root != null:
		return view_2d_root.to_local(central_star_2d.global_position);
	return Vector2.ZERO;

## Transforms a timestamped anchor-relative trail [param sample] for [param ballistic]
## into the lowest common ancestor reference frame at [code]sample.w[/code].
func _compute_trail_relative_point(
	ballistic: JrBallisticOrbiter,
	sample: Vector4
) -> Vector3:
	var anchor_rel_pos: Vector3 = Vector3(sample.x, sample.y, sample.z);
	if orbit_driver == null:
		return anchor_rel_pos;
	var anchor_elem: JrOrreryElement = _ballistic_trail_anchors.get(ballistic, ballistic.get_anchor_element());
	return orbit_driver.evaluate_trail_sample_relative_to_perspective(
		anchor_elem,
		anchor_rel_pos,
		sample.w
	);

## Updates 2D [Line2D] historical wakes relative to the active perspective body.
func _sync_2d_ballistic_trails() -> void:
	var persp_origin_2d: Vector2 = _compute_2d_perspective_origin();
	var sys_basis: Basis = orbit_driver.get_system_transform().basis if orbit_driver != null else Basis.IDENTITY;

	for body_2d: JrOrbiter2D in _ballistic_lines_2d:
		var line: Line2D = _ballistic_lines_2d[body_2d];
		var ballistic: JrBallisticOrbiter = body_2d.get_target_ballistic() if body_2d != null else null;
		if line == null or body_2d == null or ballistic == null:
			continue;

		line.position = persp_origin_2d;
		line.width = _compute_2d_line_width(body_2d);
		var trail: PackedVector4Array = _ballistic_trails.get(ballistic, PackedVector4Array());
		var count: int = trail.size();
		var pts_2d: PackedVector2Array = [];
		pts_2d.resize(count + 1);
		for i: int in range(count):
			pts_2d[i] = body_2d.project_3d_to_2d(sys_basis * _compute_trail_relative_point(ballistic, trail[i]));
		var curr_offset: Vector3 = (
			orbit_driver.evaluate_current_element_offset_from_perspective(ballistic)
			if orbit_driver != null
			else ballistic.get_anchor_relative_position()
		);
		pts_2d[count] = body_2d.project_3d_to_2d(sys_basis * curr_offset);
		line.points = pts_2d;

## Updates 2D forecast canvas anchors and schedules redraws for dashed trajectory predictions.
func _sync_2d_ballistic_forecasts() -> void:
	var persp_origin_2d: Vector2 = _compute_2d_perspective_origin();
	for body_2d: JrOrbiter2D in _ballistic_forecast_canvases_2d:
		var canvas: Node2D = _ballistic_forecast_canvases_2d[body_2d];
		if canvas == null or body_2d == null:
			continue;
		canvas.position = persp_origin_2d;
		canvas.queue_redraw();

## Draws the dashed, alpha-tapered 2D forward trajectory forecast for [param body_2d].
func _on_draw_2d_forecast(canvas: Node2D, body_2d: JrOrbiter2D) -> void:
	if canvas == null or body_2d == null:
		return;
	var ballistic: JrBallisticOrbiter = body_2d.get_target_ballistic();
	if ballistic == null:
		return;
	var pts_3d: PackedVector3Array = ballistic.get_forecast_points();
	var count: int = pts_3d.size();
	if count < 2:
		return;

	var ref_ids: PackedInt64Array = ballistic.get_forecast_reference_ids();
	var has_ref_ids: bool = (ref_ids.size() == count);
	var seg_points: PackedVector2Array = [];
	var seg_colors: PackedColorArray = [];
	var base_col: Color = body_2d.editor_orbit_color.lightened(0.22);
	var denom: float = maxf(1.0, float(count - 1));

	for k: int in range(count - 1):
		if (k % 3) == 2:
			continue;
		if has_ref_ids and ref_ids[k] != ref_ids[k + 1]:
			continue;
		var progress: float = float(k) / denom;
		var alpha: float = base_col.a * lerpf(0.95, 0.18, progress);
		seg_points.append(body_2d.project_3d_to_2d(pts_3d[k]));
		seg_points.append(body_2d.project_3d_to_2d(pts_3d[k + 1]));
		seg_colors.append(Color(base_col.r, base_col.g, base_col.b, alpha));

	if seg_points.size() >= 2:
		canvas.draw_multiline_colors(seg_points, seg_colors, _compute_2d_line_width(body_2d), true);

## Rebuilds the fading 3D [ImmediateMesh] historical wake behind each [JrBallisticOrbiter] body
## in the active perspective frame.
func _sync_3d_ballistic_trails() -> void:
	var sys_basis: Basis = orbit_driver.get_system_transform().basis if orbit_driver != null else Basis.IDENTITY;

	for body_3d: JrOrbiter3D in _ballistic_meshes_3d:
		var mesh_inst: MeshInstance3D = _ballistic_meshes_3d[body_3d];
		var ballistic: JrBallisticOrbiter = body_3d.get_target_ballistic() if body_3d != null else null;
		if mesh_inst == null or body_3d == null or ballistic == null:
			continue;
		var imm: ImmediateMesh = mesh_inst.mesh as ImmediateMesh;
		if imm == null:
			continue;
		imm.clear_surfaces();

		mesh_inst.basis = sys_basis;
		var trail: PackedVector4Array = _ballistic_trails.get(ballistic, PackedVector4Array());
		var count: int = trail.size();
		if count < 1:
			continue;

		var base_col: Color = body_3d.editor_orbit_color;
		var pos_scale: float = body_3d.position_scale;
		var denom: float = maxf(1.0, float(count));
		imm.surface_begin(Mesh.PRIMITIVE_LINE_STRIP);
		for i: int in range(count):
			var alpha: float = base_col.a * lerpf(0.05, 0.55, float(i) / denom);
			imm.surface_set_color(Color(base_col.r, base_col.g, base_col.b, alpha));
			imm.surface_add_vertex(_compute_trail_relative_point(ballistic, trail[i]) * pos_scale);
		var curr_offset: Vector3 = (
			orbit_driver.evaluate_current_element_offset_from_perspective(ballistic)
			if orbit_driver != null
			else ballistic.get_anchor_relative_position()
		);
		imm.surface_set_color(Color(base_col.r, base_col.g, base_col.b, base_col.a * 0.58));
		imm.surface_add_vertex(curr_offset * pos_scale);
		imm.surface_end();

## Rebuilds the dashed, alpha-tapered 3D [ImmediateMesh] forward trajectory forecast
## ahead of each [JrBallisticOrbiter] body in the active perspective frame.
func _sync_3d_ballistic_forecasts() -> void:
	for body_3d: JrOrbiter3D in _ballistic_forecast_meshes_3d:
		var mesh_inst: MeshInstance3D = _ballistic_forecast_meshes_3d[body_3d];
		var ballistic: JrBallisticOrbiter = body_3d.get_target_ballistic() if body_3d != null else null;
		if mesh_inst == null or body_3d == null or ballistic == null:
			continue;
		var imm: ImmediateMesh = mesh_inst.mesh as ImmediateMesh;
		if imm == null:
			continue;
		imm.clear_surfaces();

		var pts: PackedVector3Array = ballistic.get_forecast_points();
		var count: int = pts.size();
		if count < 2:
			continue;

		var ref_ids: PackedInt64Array = ballistic.get_forecast_reference_ids();
		var has_ref_ids: bool = (ref_ids.size() == count);
		var base_col: Color = body_3d.editor_orbit_color.lightened(0.22);
		var pos_scale: float = body_3d.position_scale;
		var denom: float = maxf(1.0, float(count - 1));
		var has_segments: bool = false;

		for k: int in range(count - 1):
			if (k % 3) == 2:
				continue;
			if has_ref_ids and ref_ids[k] != ref_ids[k + 1]:
				continue;
			if not has_segments:
				imm.surface_begin(Mesh.PRIMITIVE_LINES);
				has_segments = true;
			var alpha_a: float = base_col.a * lerpf(0.95, 0.18, float(k) / denom);
			var alpha_b: float = base_col.a * lerpf(0.95, 0.18, float(k + 1) / denom);
			imm.surface_set_color(Color(base_col.r, base_col.g, base_col.b, alpha_a));
			imm.surface_add_vertex(pts[k] * pos_scale);
			imm.surface_set_color(Color(base_col.r, base_col.g, base_col.b, alpha_b));
			imm.surface_add_vertex(pts[k + 1] * pos_scale);

		if has_segments:
			imm.surface_end();
#endregion

#region HUD Status Updates
## Updates all HUD labels and the view-transfer button to reflect the active view
## mode, camera type, active scenario description, and selected perspective body.
func _update_hud() -> void:
	var is_2d: bool = _current_view_mode == ViewMode.VIEW_2D;

	if title_label != null:
		title_label.text = (
			"JR Orrery — 2D Orbital Mechanics Demo"
			if is_2d
			else "JR Orrery — 3D Orbital Mechanics Demo"
		);

	if overview_label != null and _current_scenario_index >= 0 and _current_scenario_index < scenario_descriptions.size():
		overview_label.text = scenario_descriptions[_current_scenario_index];

	if camera_status_label != null:
		camera_status_label.text = (
			"Active Camera: Camera2D (Top-Down X-Z Projection)"
			if is_2d
			else "Active Camera: Camera3D (3D Orbital Rig)"
		);

	if controls_label != null:
		controls_label.text = _get_2d_controls_text() if is_2d else _get_3d_controls_text();

	if switch_view_button != null:
		switch_view_button.text = (
			"Transfer to 3D View (Camera3D)  [Tab / V]"
			if is_2d
			else "Transfer to 2D View (Camera2D)  [Tab / V]"
		);

	_update_reference_status_label();

## Returns the formatted controls description string for [constant ViewMode.VIEW_2D].
func _get_2d_controls_text() -> String:
	return (
		"• Keys 1–5 (or Dropdown): Switch showcase scenario\n"
		+ "• Tab / V (or Button): Transfer to 3D View (Camera3D)\n"
		+ "• Click & Drag: Pan 2D camera across the orbital system\n"
		+ "• Mouse Wheel: Zoom 2D camera in and out around cursor\n"
		+ "• Double-Click Planet / Moon / Probe: Lock perspective to that body\n"
		+ "• Double-Click Central Star: Reset to root perspective\n"
		+ "• R Key: Cycle Inertial / Surface-Locked / Synodic (L1–L5) frame\n"
		+ "• H Key (or Hide/Instructions): Collapse or expand this panel"
	);

## Returns the formatted controls description string for [constant ViewMode.VIEW_3D].
func _get_3d_controls_text() -> String:
	return (
		"• Keys 1–5 (or Dropdown): Switch showcase scenario\n"
		+ "• Tab / V (or Button): Transfer to 2D View (Camera2D)\n"
		+ "• Right-Click & Drag: Orbit 3D camera around selected body\n"
		+ "• Mouse Wheel: Zoom 3D camera in and out\n"
		+ "• Double-Click Planet / Moon / Probe: Lock perspective to that body\n"
		+ "• Double-Click Central Star: Reset to root perspective\n"
		+ "• R Key: Cycle Inertial / Surface-Locked / Synodic (L1–L5) frame\n"
		+ "• H Key (or Hide/Instructions): Collapse or expand this panel"
	);

## Updates [member reference_status_label] with the active [member JrOrbitDriver.perspective_node]
## and active [enum JrOrbitDriver.PerspectiveRotationMode].
func _update_reference_status_label() -> void:
	if reference_status_label == null or orbit_driver == null:
		return;
	var mode_suffix: String = " [Inertial]";
	if orbit_driver.perspective_rotation_mode == JrOrbitDriver.PerspectiveRotationMode.BODY_SPIN:
		mode_suffix = " [Surface-Locked]";
	elif orbit_driver.perspective_rotation_mode == JrOrbitDriver.PerspectiveRotationMode.SYNODIC_ORBITAL:
		mode_suffix = " [Synodic L1–L5]";

	if orbit_driver.perspective_node == null or orbit_driver.perspective_node == orbit_driver:
		reference_status_label.text = "Active Reference: System Root (Heliocentric/Barycentric)" + mode_suffix;
	else:
		reference_status_label.text = "Active Reference: " + String(orbit_driver.perspective_node.name) + mode_suffix;
#endregion
