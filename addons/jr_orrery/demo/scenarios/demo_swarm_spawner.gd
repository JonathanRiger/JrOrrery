## Populates the 256-body GPU asteroid belt scenario at runtime and flattens the
## simulation hierarchy via [method JrOrbitDriver.collect_ballistic_nodes].
extends Node
class_name JrDemoSwarmSpawner

## Number of additional belt asteroids spawned at runtime alongside the 4 featured asteroids.
@export_range(0, 1024, 1)
var additional_asteroid_count: int = 252;

## Inner orbital radius of the spawned asteroid belt.
@export
var belt_inner_radius: float = 17.5;

## Outer orbital radius of the spawned asteroid belt.
@export
var belt_outer_radius: float = 28.5;

## Shared 2D position scale applied to spawned 2D asteroid followers.
@export
var position_scale_2d: float = 13.0;

## Shared mesh assigned to spawned 3D asteroid visual followers.
var _shared_mesh_3d: SphereMesh = null;

## Shared material assigned to spawned 3D asteroid visual followers.
var _shared_material_3d: StandardMaterial3D = null;

## Begins a batch update on [JrOrbitDriver] before child [method Node._ready] callbacks fire
## so the driver only evaluates orbits and GPU forecasts once after all belt asteroids are spawned.
func _enter_tree() -> void:
	if Engine.is_editor_hint() or additional_asteroid_count <= 0:
		return;
	var driver: JrOrbitDriver = get_node_or_null("JrOrbitDriver") as JrOrbitDriver;
	if driver != null:
		driver.begin_batch_updates();

## Spawns the asteroid belt nodes and calls [method JrOrbitDriver.collect_ballistic_nodes].
func _ready() -> void:
	if Engine.is_editor_hint() or additional_asteroid_count <= 0:
		return;

	var driver: JrOrbitDriver = get_node_or_null("JrOrbitDriver") as JrOrbitDriver;
	var star_3d: Node3D = get_node_or_null("View3D/CentralStar") as Node3D;
	var star_2d: Node2D = get_node_or_null("View2D/CentralStar") as Node2D;
	if driver == null or star_3d == null or star_2d == null:
		if driver != null:
			driver.end_batch_updates();
		return;

	_init_shared_3d_resources();
	var belt_group: JrOrreryElement = _find_or_create_belt_anchor(driver);
	var icon_tex: Texture2D = load("res://icon.svg") as Texture2D;
	var mu_star: float = maxf(1.0, driver.get_effective_parent_mu());
	var parent_pos: Vector3 = belt_group.get_focus_relative_position() if belt_group != driver else Vector3.ZERO;
	var parent_vel: Vector3 = belt_group.get_relative_velocity() if belt_group != driver else Vector3.ZERO;

	for i: int in range(additional_asteroid_count):
		var ball: JrBallisticOrbiter = _create_belt_ballistic(i, mu_star, parent_pos, parent_vel);
		ball.pinned_anchor = driver;
		belt_group.add_child(ball);
		_create_3d_follower(ball, star_3d);
		_create_2d_follower(ball, star_2d, icon_tex);

	driver.collect_ballistic_nodes();
	driver.end_batch_updates();

## Initializes the shared low-poly [SphereMesh] and [StandardMaterial3D] for belt asteroids.
func _init_shared_3d_resources() -> void:
	_shared_mesh_3d = SphereMesh.new();
	_shared_mesh_3d.radius = 0.14;
	_shared_mesh_3d.height = 0.28;
	_shared_mesh_3d.radial_segments = 8;
	_shared_mesh_3d.rings = 4;

	_shared_material_3d = StandardMaterial3D.new();
	_shared_material_3d.albedo_color = Color(0.76, 0.72, 0.66, 1.0);
	_shared_material_3d.roughness = 0.85;

## Returns [param driver] as the spawn parent (or a nested child orbiter for testing reparenting).
func _find_or_create_belt_anchor(driver: JrOrbitDriver) -> JrOrreryElement:
	var rocky: JrRailOrbiter = driver.get_node_or_null("InnerRockyPlanet") as JrRailOrbiter;
	if rocky != null:
		return rocky;
	return driver;

## Creates a deterministic belt [JrBallisticOrbiter] at index [param idx] around [param mu_star].
func _create_belt_ballistic(
	idx: int,
	mu_star: float,
	parent_pos: Vector3 = Vector3.ZERO,
	parent_vel: Vector3 = Vector3.ZERO
) -> JrBallisticOrbiter:
	var golden_angle: float = float(idx) * 2.399963229728653;
	var u_rad: float = fmod(float(idx) * 0.61803398875, 1.0);
	var radius: float = lerpf(belt_inner_radius, belt_outer_radius, u_rad);
	var ecc: float = 0.02 + 0.09 * fmod(float(idx) * 0.381966, 1.0);
	var inc: float = (fmod(float(idx) * 0.754877, 1.0) - 0.5) * 0.22;

	var pos_star: Vector3 = Vector3(
		radius * sin(golden_angle),
		radius * sin(inc) * cos(golden_angle * 1.7),
		-radius * cos(golden_angle)
	);
	var v_circ: float = sqrt(mu_star / radius) * (1.0 + 0.4 * ecc * cos(golden_angle * 2.0));
	var vel_star: Vector3 = Vector3(
		v_circ * cos(golden_angle),
		v_circ * 0.08 * sin(inc),
		v_circ * sin(golden_angle)
	);

	var ball: JrBallisticOrbiter = JrBallisticOrbiter.new();
	ball.name = "BeltAsteroid_%03d" % idx;
	ball.initial_position = pos_star - parent_pos;
	ball.initial_velocity = vel_star - parent_vel;
	ball.runtime_forecast_enabled = false;
	ball.preview_steps = 2;
	ball.preview_duration = 1.0;
	return ball;

## Creates a 3D visual follower for [param ball] under [param star_3d].
func _create_3d_follower(ball: JrBallisticOrbiter, star_3d: Node3D) -> void:
	var orb_3d: JrOrbiter3D = JrOrbiter3D.new();
	orb_3d.name = ball.name + "_3D";
	orb_3d.set_meta(&"swarm_background", true);
	orb_3d.draw_orbit_in_editor = false;
	orb_3d.show_editor_ghost = false;
	orb_3d.orbiter = ball;

	var mesh_inst: MeshInstance3D = MeshInstance3D.new();
	mesh_inst.mesh = _shared_mesh_3d;
	mesh_inst.material_override = _shared_material_3d;
	mesh_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	orb_3d.add_child(mesh_inst);
	star_3d.add_child(orb_3d);

## Creates a 2D visual follower for [param ball] under [param star_2d].
func _create_2d_follower(ball: JrBallisticOrbiter, star_2d: Node2D, icon_tex: Texture2D) -> void:
	var orb_2d: JrOrbiter2D = JrOrbiter2D.new();
	orb_2d.name = ball.name + "_2D";
	orb_2d.set_meta(&"swarm_background", true);
	orb_2d.draw_orbit_in_editor = false;
	orb_2d.show_editor_ghost = false;
	orb_2d.position_scale = position_scale_2d;
	orb_2d.orbiter = ball;

	if icon_tex != null:
		var sprite: Sprite2D = Sprite2D.new();
		sprite.texture = icon_tex;
		sprite.scale = Vector2(0.055, 0.055);
		sprite.modulate = Color(0.78, 0.74, 0.68, 0.85);
		orb_2d.add_child(sprite);
	star_2d.add_child(orb_2d);
