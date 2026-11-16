@tool
class_name JrBenchmarkSceneBuilder
extends RefCounted
## Deterministic scene and synthetic hierarchy builder for [JrOrreryBenchmark].
##
## Instantiates existing demo `.tscn` files or procedurally constructs multi-well planetary
## systems, ballistic swarms, forecasted trajectories, and deep [JrRailOrbiter] trees.

## Builds, attaches to [param parent_viewport], and configures the benchmark root node
## and its primary [JrOrbitDriver] according to [param config].
static func build_and_mount(
		parent_viewport: Window,
		config: JrBenchmarkConfig
) -> Dictionary:
	if config.kind == JrBenchmarkConfig.KIND_DEMO:
		return _mount_demo_scenario(parent_viewport, config);
	if config.kind == JrBenchmarkConfig.KIND_SYNTHETIC_RAILS:
		return _mount_synthetic_rails(parent_viewport, config);
	return _mount_synthetic_ballistics(parent_viewport, config);


## Instantiates a demo `.tscn` scene, attaches it to [param parent_viewport], disables automatic
## idle/physics stepping, and optionally strips follower subtrees.
static func _mount_demo_scenario(
		parent_viewport: Window,
		config: JrBenchmarkConfig
) -> Dictionary:
	var packed := load(config.scenario_path) as PackedScene;
	var root_node: Node = packed.instantiate();
	parent_viewport.add_child(root_node);

	var driver: JrOrbitDriver = _find_orbit_driver(root_node);
	driver.set_process(false);
	driver.set_physics_process(false);
	driver.auto_advance = true;
	driver.set_force_cpu_solver(config.force_cpu);

	if not config.attach_followers:
		_strip_follower_views(root_node);

	return {
		"root": root_node,
		"driver": driver,
	};


## Removes `View2D` and `View3D` subtrees from a mounted demo root to isolate core simulation
## cost from synchronous `JrOrbiter2D`/`JrOrbiter3D` follower callbacks.
static func _strip_follower_views(root_node: Node) -> void:
	var view_3d: Node = root_node.get_node_or_null("View3D");
	if view_3d != null:
		root_node.remove_child(view_3d);
		view_3d.free();
	var view_2d: Node = root_node.get_node_or_null("View2D");
	if view_2d != null:
		root_node.remove_child(view_2d);
		view_2d.free();


## Constructs a synthetic 8-well star/planet/moon system populated with ballistic bodies
## configured for either swarm stepping or trajectory forecast profiling.
static func _mount_synthetic_ballistics(
		parent_viewport: Window,
		config: JrBenchmarkConfig
) -> Dictionary:
	var root_node := Node.new();
	root_node.name = "SyntheticBallisticRoot";

	var driver := JrOrbitDriver.new();
	driver.name = "JrOrbitDriver";
	driver.mass = 1.0;
	driver.mass_unit = JrOrreryUnits.MassUnit.SOLAR_MASSES;
	driver.distance_unit = JrOrreryUnits.DistanceUnit.ASTRONOMICAL_UNITS;
	driver.time_unit = JrOrreryUnits.TimeUnit.YEARS;
	driver.time_scale = 0.25;
	driver.tidal_cull_threshold = config.cull_threshold;
	driver.enable_ballistic_forecasts = config.forecast_count > 0;
	driver.auto_advance = true;
	root_node.add_child(driver);

	driver.begin_batch_updates();
	var wells: Array[JrRailOrbiter] = _populate_planetary_wells(driver, config.rail_count);
	_populate_ballistic_bodies(driver, wells, config);

	if config.synodic_perspective and not wells.is_empty():
		driver.perspective_node = wells[1] if wells.size() > 1 else wells[0];
		driver.perspective_rotation_mode = JrOrbitDriver.PerspectiveRotationMode.SYNODIC_ORBITAL;

	parent_viewport.add_child(root_node);
	driver.set_process(false);
	driver.set_physics_process(false);
	driver.set_force_cpu_solver(config.force_cpu);
	driver.collect_ballistic_nodes();
	driver.end_batch_updates();

	return {
		"root": root_node,
		"driver": driver,
	};


## Constructs a multi-tier analytical [JrRailOrbiter] hierarchy (planets -> moons -> sub-moons)
## to measure pure Keplerian evaluation and LCA tree-walk scaling.
static func _mount_synthetic_rails(
		parent_viewport: Window,
		config: JrBenchmarkConfig
) -> Dictionary:
	var root_node := Node.new();
	root_node.name = "SyntheticRailRoot";

	var driver := JrOrbitDriver.new();
	driver.name = "JrOrbitDriver";
	driver.mass = 1.0;
	driver.mass_unit = JrOrreryUnits.MassUnit.SOLAR_MASSES;
	driver.distance_unit = JrOrreryUnits.DistanceUnit.ASTRONOMICAL_UNITS;
	driver.time_unit = JrOrreryUnits.TimeUnit.YEARS;
	driver.time_scale = 0.5;
	driver.enable_ballistic_forecasts = false;
	driver.auto_advance = true;
	root_node.add_child(driver);

	driver.begin_batch_updates();
	var deepest_leaf: JrRailOrbiter = _populate_hierarchical_rails(driver, config.rail_count);
	if config.deep_perspective and deepest_leaf != null:
		driver.perspective_node = deepest_leaf;
		driver.perspective_rotation_mode = JrOrbitDriver.PerspectiveRotationMode.SYNODIC_ORBITAL;

	parent_viewport.add_child(root_node);
	driver.set_process(false);
	driver.set_physics_process(false);
	driver.end_batch_updates();

	return {
		"root": root_node,
		"driver": driver,
	};


## Populates [param well_count] planetary and lunar [JrRailOrbiter] gravity wells under [param driver].
static func _populate_planetary_wells(
		driver: JrOrbitDriver,
		well_count: int
) -> Array[JrRailOrbiter]:
	var wells: Array[JrRailOrbiter] = [];
	var planet_count: int = maxi(1, mini(well_count, 6));
	for p: int in range(planet_count):
		var planet := JrRailOrbiter.new();
		planet.name = "Planet_%d" % p;
		planet.periapsis = 0.6 + float(p) * 0.85;
		planet.eccentricity = 0.015 + 0.02 * float(p % 4);
		planet.inclination = deg_to_rad(float(p) * 1.4);
		planet.longitude_of_the_ascending_node = deg_to_rad(float(p) * 47.0);
		planet.argument_of_periapsis = deg_to_rad(float(p) * 29.0);
		planet.mean_anomaly_at_epoch = deg_to_rad(float(p) * 63.0);
		planet.mass = 0.001 * float(1 + (p % 3));
		planet.mass_unit = JrOrreryUnits.MassUnit.SOLAR_MASSES;
		planet.sync_motion_to_gravity = true;
		driver.add_child(planet);
		wells.append(planet);

	var remaining: int = well_count - planet_count;
	for m: int in range(remaining):
		var host: JrRailOrbiter = wells[m % planet_count];
		var moon := JrRailOrbiter.new();
		moon.name = "Moon_%d" % m;
		moon.periapsis = 0.03 + 0.015 * float(m);
		moon.eccentricity = 0.01;
		moon.inclination = deg_to_rad(3.5);
		moon.mean_anomaly_at_epoch = deg_to_rad(float(m) * 110.0);
		moon.mass = 0.00002;
		moon.mass_unit = JrOrreryUnits.MassUnit.SOLAR_MASSES;
		moon.sync_motion_to_gravity = true;
		host.add_child(moon);
		wells.append(moon);

	return wells;


## Populates [param config.ballistic_count] deterministic [JrBallisticOrbiter] bodies using
## state vectors around [param driver] (or planetary [param wells] when unpinned).
static func _populate_ballistic_bodies(
		driver: JrOrbitDriver,
		wells: Array[JrRailOrbiter],
		config: JrBenchmarkConfig
) -> void:
	var mu_star: float = JrOrreryUnits.compute_mu(
			driver.mass,
			driver.mass_unit,
			driver.distance_unit,
			driver.time_unit
	);

	for i: int in range(config.ballistic_count):
		var body := JrBallisticOrbiter.new();
		body.name = "Ballistic_%d" % i;
		body.initial_state_mode = JrBallisticOrbiter.InitialStateMode.STATE_VECTORS;
		body.enable_tidal_forces = true;
		body.runtime_forecast_enabled = i < config.forecast_count;
		body.preview_steps = config.forecast_steps;

		var spawn_host: JrOrreryElement = driver;
		var host_mu: float = mu_star;
		var radius: float = lerpf(0.85, 4.2, fmod(float(i) * 0.61803398875, 1.0));
		body.preview_duration = 1.25;

		if not config.pin_anchors and not wells.is_empty() and (i % 8 == 0):
			var host_well: JrRailOrbiter = wells[i % wells.size()];
			spawn_host = host_well;
			host_mu = JrOrreryUnits.compute_mu(
					host_well.mass,
					host_well.mass_unit,
					driver.distance_unit,
					driver.time_unit
			);
			radius = 0.015 + 0.01 * fmod(float(i) * 0.29, 1.0);
			body.preview_duration = 0.18;

		var angle: float = float(i) * 2.399963229728653;
		var inc: float = (fmod(float(i) * 0.754877, 1.0) - 0.5) * 0.22;
		var ecc: float = 0.02 + 0.12 * fmod(float(i) * 0.381966, 1.0);
		var v_circ: float = sqrt(maxf(1e-9, host_mu) / radius) * (1.0 + 0.4 * ecc * cos(angle * 2.0));

		body.initial_position = Vector3(
				radius * sin(angle),
				radius * sin(inc) * cos(angle * 1.7),
				-radius * cos(angle)
		);
		body.initial_velocity = Vector3(
				v_circ * cos(angle),
				v_circ * 0.08 * sin(inc),
				v_circ * sin(angle)
		);

		if config.pin_anchors:
			body.pinned_anchor = driver;

		spawn_host.add_child(body);


## Populates a 3-tier planet -> moon -> sub-moon tree with [param total_orbiters] nodes and
## returns the deepest leaf orbiter.
static func _populate_hierarchical_rails(
		driver: JrOrbitDriver,
		total_orbiters: int
) -> JrRailOrbiter:
	var created: int = 0;
	var planet_idx: int = 0;
	var last_leaf: JrRailOrbiter = null;

	while created < total_orbiters:
		var planet: JrRailOrbiter = _make_rail_node(
				"Planet_%d" % planet_idx,
				0.5 + float(planet_idx) * 0.35,
				1e-3,
				planet_idx
		);
		driver.add_child(planet);
		created += 1;
		last_leaf = planet;
		planet_idx += 1;

		for m: int in range(3):
			if created >= total_orbiters:
				break;
			var moon: JrRailOrbiter = _make_rail_node(
					"Moon_%d_%d" % [planet_idx, m],
					0.02 + 0.01 * float(m),
					1e-5,
					created
			);
			planet.add_child(moon);
			created += 1;
			last_leaf = moon;

			for s: int in range(2):
				if created >= total_orbiters:
					break;
				var sub: JrRailOrbiter = _make_rail_node(
						"Sub_%d_%d_%d" % [planet_idx, m, s],
						0.002 + 0.001 * float(s),
						1e-7,
						created
				);
				moon.add_child(sub);
				created += 1;
				last_leaf = sub;

	return last_leaf;


## Creates a single configured [JrRailOrbiter] with deterministic Keplerian elements.
static func _make_rail_node(
		node_name: String,
		periapsis_dist: float,
		body_mass: float,
		seed_idx: int
) -> JrRailOrbiter:
	var orb := JrRailOrbiter.new();
	orb.name = node_name;
	orb.periapsis = periapsis_dist;
	orb.eccentricity = 0.01 + 0.08 * fmod(float(seed_idx) * 0.37, 1.0);
	orb.inclination = deg_to_rad(fmod(float(seed_idx) * 2.3, 18.0));
	orb.longitude_of_the_ascending_node = deg_to_rad(fmod(float(seed_idx) * 43.0, 360.0));
	orb.argument_of_periapsis = deg_to_rad(fmod(float(seed_idx) * 71.0, 360.0));
	orb.mean_anomaly_at_epoch = deg_to_rad(fmod(float(seed_idx) * 137.5, 360.0));
	orb.mass = body_mass;
	orb.mass_unit = JrOrreryUnits.MassUnit.SOLAR_MASSES;
	orb.sync_motion_to_gravity = true;
	return orb;


## Searches [param node] and its immediate children for the scenario's [JrOrbitDriver].
static func _find_orbit_driver(node: Node) -> JrOrbitDriver:
	if node is JrOrbitDriver:
		return node as JrOrbitDriver;
	var direct := node.get_node_or_null("JrOrbitDriver") as JrOrbitDriver;
	if direct != null:
		return direct;
	for child: Node in node.get_children():
		if child is JrOrbitDriver:
			return child as JrOrbitDriver;
	return null;
