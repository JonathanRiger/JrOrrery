extends SceneTree
## Automated CLI performance benchmark runner for `JrOrrery`.
##
## Executes deterministic warmup and timed simulation ticks across all 5 demo scenarios and
## synthetic scaling suites (ballistic swarms, Tier-1 on-rails vs active KDK, pinned vs unpinned
## Hill-sphere handoffs, trajectory forecasting in inertial vs synodic frames, and hierarchical
## `JrRailOrbiter` trees), printing a detailed microsecond telemetry breakdown to stdout.

## Demo scenario directory prefix.
const SCENARIO_DIR: String = "res://addons/jr_orrery/demo/scenarios/";

## Ensures the benchmark suite runs once on the first `SceneTree` iteration and exits cleanly.
var _executed: bool = false;


## Runs the full benchmark suite after `SceneTree::initialize()` has mounted `root`, then
## returns `true` to terminate the main loop cleanly with zero leaked objects.
func _process(_delta: float) -> bool:
	if _executed:
		return true;
	_executed = true;

	_minimize_window_if_visible();
	_print_environment_banner();
	var suite_filter: String = _parse_cli_filter("--suite=");
	var case_filter: String = _parse_cli_filter("--case=");
	if suite_filter.is_empty() and case_filter.is_empty():
		_run_observer_overhead_check();

	var selected_cases: Array[JrBenchmarkConfig] = _filter_benchmark_cases(
			_build_all_benchmark_cases(),
			suite_filter,
			case_filter
	);
	var total_steps: int = selected_cases.size();
	var current_suite: String = "";

	for i: int in range(total_steps):
		var cfg: JrBenchmarkConfig = selected_cases[i];
		if cfg.suite_name != current_suite:
			current_suite = cfg.suite_name;
			print("\n================================================================================");
			print("SUITE: %s" % current_suite);
			print("================================================================================");
		print("\n--- [Step %d of %d] %s ---" % [i + 1, total_steps, cfg.case_name]);
		var telemetry: JrOrreryTelemetry = _execute_single_case(cfg);
		_print_case_report(cfg, telemetry);

	print("\n================================================================================");
	print("BENCHMARK SUITE COMPLETE");
	print("================================================================================");
	quit(0);
	return true;


## Minimizes the OS window when running with `--display-driver windows` so D3D12 compute runs unobtrusively.
func _minimize_window_if_visible() -> void:
	if DisplayServer.get_name() != "headless":
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MINIMIZED);


## Extracts an optional CLI filter with [param prefix] (e.g. `--suite=` or `--case=`) after `--`.
func _parse_cli_filter(prefix: String) -> String:
	for arg: String in OS.get_cmdline_user_args():
		if arg.begins_with(prefix):
			return arg.trim_prefix(prefix).strip_edges().to_lower();
	return "";


## Filters [param all_cases] by [param suite_filter] and [param case_filter] when non-empty.
func _filter_benchmark_cases(
		all_cases: Array[JrBenchmarkConfig],
		suite_filter: String,
		case_filter: String
) -> Array[JrBenchmarkConfig]:
	if suite_filter.is_empty() and case_filter.is_empty():
		return all_cases;
	var filtered: Array[JrBenchmarkConfig] = [];
	for cfg: JrBenchmarkConfig in all_cases:
		var suite_ok: bool = suite_filter.is_empty() or cfg.suite_name.to_lower().contains(suite_filter);
		var case_ok: bool = case_filter.is_empty() or cfg.case_name.to_lower().contains(case_filter);
		if suite_ok and case_ok:
			filtered.append(cfg);
	return filtered;


## Prints hardware, OS, Godot version, and active rendering adapter metadata.
func _print_environment_banner() -> void:
	var ver: Dictionary = Engine.get_version_info();
	print("================================================================================");
	print("JR ORRERY PERFORMANCE & TELEMETRY BENCHMARK");
	print("================================================================================");
	print("Godot Version    : %s" % String(ver.get("string", "unknown")));
	print("OS / Platform    : %s (%s)" % [OS.get_name(), OS.get_version()]);
	print("CPU              : %s (%d logical cores)" % [OS.get_processor_name(), OS.get_processor_count()]);
	print("Display Driver   : %s" % DisplayServer.get_name());
	print("Rendering Driver : %s" % RenderingServer.get_current_rendering_driver_name());
	print("GPU Adapter      : %s (%s)" % [
		RenderingServer.get_video_adapter_name(),
		RenderingServer.get_video_adapter_vendor(),
	]);


## Measures wall-clock tick time with `telemetry == null` vs `telemetry != null` on a 1024-body swarm.
func _run_observer_overhead_check() -> void:
	var cfg: JrBenchmarkConfig = JrBenchmarkConfig.create_swarm(
			"Overhead Check (1024 bodies)",
			1024,
			false,
			true
	);
	var ticks: int = 40;
	var mounted_a: Dictionary = JrBenchmarkSceneBuilder.build_and_mount(root, cfg);
	var driver_a: JrOrbitDriver = mounted_a["driver"] as JrOrbitDriver;
	var node_a: Node = mounted_a["root"] as Node;
	driver_a.set_telemetry(null);
	_step_driver_n_times(driver_a, cfg.frame_delta, 6);
	var t_uninst_0: int = Time.get_ticks_usec();
	_step_driver_n_times(driver_a, cfg.frame_delta, ticks);
	var uninst_mean_usec: float = float(Time.get_ticks_usec() - t_uninst_0) / float(ticks);
	_unmount_and_free(node_a);

	var mounted_b: Dictionary = JrBenchmarkSceneBuilder.build_and_mount(root, cfg);
	var driver_b: JrOrbitDriver = mounted_b["driver"] as JrOrbitDriver;
	var node_b: Node = mounted_b["root"] as Node;
	var telemetry := JrOrreryTelemetry.new();
	driver_b.set_telemetry(telemetry);
	_step_driver_n_times(driver_b, cfg.frame_delta, 6);
	telemetry.reset();
	var t_inst_0: int = Time.get_ticks_usec();
	_step_driver_n_times(driver_b, cfg.frame_delta, ticks);
	var inst_mean_usec: float = float(Time.get_ticks_usec() - t_inst_0) / float(ticks);
	_unmount_and_free(node_b);

	var delta_usec: float = inst_mean_usec - uninst_mean_usec;
	print("--------------------------------------------------------------------------------");
	print("OBSERVER OVERHEAD CHECK (1024 GPU Ballistic Bodies, %d ticks):" % ticks);
	print("  Un-instrumented (telemetry == null) : %.2f us/tick" % uninst_mean_usec);
	print("  Instrumented    (telemetry != null) : %.2f us/tick (delta: %+.2f us/tick)" % [
		inst_mean_usec,
		delta_usec,
	]);


## Builds the complete ordered list of demo and synthetic benchmark configurations.
func _build_all_benchmark_cases() -> Array[JrBenchmarkConfig]:
	var list: Array[JrBenchmarkConfig] = [];
	_append_demo_cases(list);
	_append_swarm_cases(list);
	_append_forecast_cases(list);
	_append_rail_cases(list);
	return list;


## Appends all 5 demo scenarios (GPU, CPU, and follower-isolated comparison) to [param list].
func _append_demo_cases(list: Array[JrBenchmarkConfig]) -> void:
	list.append(JrBenchmarkConfig.create_demo("Planetary System [GPU]", SCENARIO_DIR + "demo_planetary_system.tscn", false));
	list.append(JrBenchmarkConfig.create_demo("Planetary System [CPU]", SCENARIO_DIR + "demo_planetary_system.tscn", true));
	list.append(JrBenchmarkConfig.create_demo("Real-Scale LEO [GPU]", SCENARIO_DIR + "demo_real_scale_leo.tscn", false));
	list.append(JrBenchmarkConfig.create_demo("Real-Scale LEO [CPU]", SCENARIO_DIR + "demo_real_scale_leo.tscn", true));
	list.append(JrBenchmarkConfig.create_demo("Lagrange Points (Synodic) [GPU]", SCENARIO_DIR + "demo_lagrange_points.tscn", false));
	list.append(JrBenchmarkConfig.create_demo("Lagrange Points (Synodic) [CPU]", SCENARIO_DIR + "demo_lagrange_points.tscn", true));
	list.append(JrBenchmarkConfig.create_demo("Binary Stars [GPU]", SCENARIO_DIR + "demo_binary_stars.tscn", false));
	list.append(JrBenchmarkConfig.create_demo("Binary Stars [CPU]", SCENARIO_DIR + "demo_binary_stars.tscn", true));
	list.append(JrBenchmarkConfig.create_demo("Asteroid Swarm 256 [GPU + Followers]", SCENARIO_DIR + "demo_asteroid_swarm.tscn", false, true));
	list.append(JrBenchmarkConfig.create_demo("Asteroid Swarm 256 [GPU, Core Only]", SCENARIO_DIR + "demo_asteroid_swarm.tscn", false, false));
	list.append(JrBenchmarkConfig.create_demo("Asteroid Swarm 256 [CPU + Followers]", SCENARIO_DIR + "demo_asteroid_swarm.tscn", true, true));


## Appends synthetic N-body swarm scaling cases (GPU vs CPU, Pinned vs Unpinned, Active vs On-Rails).
func _append_swarm_cases(list: Array[JrBenchmarkConfig]) -> void:
	for count: int in [256, 1024, 4096]:
		list.append(JrBenchmarkConfig.create_swarm("Swarm %d [GPU, Pinned, Active KDK]" % count, count, false, true));
		list.append(JrBenchmarkConfig.create_swarm("Swarm %d [GPU, Unpinned, Active KDK]" % count, count, false, false));
		list.append(JrBenchmarkConfig.create_swarm("Swarm %d [CPU, Pinned, Active KDK]" % count, count, true, true));

	for count: int in [1024, 4096]:
		var gpu_rails: JrBenchmarkConfig = JrBenchmarkConfig.create_swarm(
				"Swarm %d [GPU, Pinned, Tier-1 On-Rails]" % count,
				count,
				false,
				true
		);
		gpu_rails.cull_threshold = 10.0;
		list.append(gpu_rails);

		var cpu_rails: JrBenchmarkConfig = JrBenchmarkConfig.create_swarm(
				"Swarm %d [CPU, Pinned, Tier-1 On-Rails]" % count,
				count,
				true,
				true
		);
		cpu_rails.cull_threshold = 10.0;
		cpu_rails.measured_ticks = 12 if count >= 4096 else 25;
		list.append(cpu_rails);


## Appends synthetic trajectory forecast scaling cases (1..64 forecasted bodies, Inertial vs Synodic, GPU vs CPU).
func _append_forecast_cases(list: Array[JrBenchmarkConfig]) -> void:
	for count: int in [1, 4, 16, 64]:
		list.append(JrBenchmarkConfig.create_forecast("Forecast %d bodies [GPU, Inertial]" % count, count, false, false));
		list.append(JrBenchmarkConfig.create_forecast("Forecast %d bodies [GPU, Synodic]" % count, count, false, true));
	for count: int in [1, 4, 16]:
		list.append(JrBenchmarkConfig.create_forecast("Forecast %d bodies [CPU, Inertial]" % count, count, true, false));


## Appends synthetic analytical [JrRailOrbiter] hierarchy scaling cases (16..1024 bodies).
func _append_rail_cases(list: Array[JrBenchmarkConfig]) -> void:
	for count: int in [16, 64, 256, 1024]:
		list.append(JrBenchmarkConfig.create_rails("Rails %d bodies [Heliocentric]" % count, count, false));
		list.append(JrBenchmarkConfig.create_rails("Rails %d bodies [Deep Moon Synodic]" % count, count, true));


## Mounts [param cfg], executes warmup ticks, records [member JrBenchmarkConfig.measured_ticks],
## frees all scene resources, and returns the populated [JrOrreryTelemetry].
func _execute_single_case(cfg: JrBenchmarkConfig) -> JrOrreryTelemetry:
	var t_m0: int = Time.get_ticks_msec();
	var mounted: Dictionary = JrBenchmarkSceneBuilder.build_and_mount(root, cfg);
	var t_m1: int = Time.get_ticks_msec();
	var root_node: Node = mounted["root"] as Node;
	var driver: JrOrbitDriver = mounted["driver"] as JrOrbitDriver;
	var telemetry := JrOrreryTelemetry.new();

	driver.set_telemetry(telemetry);
	_step_driver_n_times(driver, cfg.frame_delta, cfg.warmup_ticks);
	var t_m2: int = Time.get_ticks_msec();
	telemetry.reset();
	_step_driver_n_times(driver, cfg.frame_delta, cfg.measured_ticks);
	var t_m3: int = Time.get_ticks_msec();
	driver.set_telemetry(null);

	_unmount_and_free(root_node);
	var t_m4: int = Time.get_ticks_msec();
	print("  Lifecycle: mount = %d ms | warmup = %d ms | measure = %d ms | free = %d ms" % [
		t_m1 - t_m0,
		t_m2 - t_m1,
		t_m3 - t_m2,
		t_m4 - t_m3,
	]);
	return telemetry;


## Steps [param driver] forward [param tick_count] times by [param delta].
func _step_driver_n_times(driver: JrOrbitDriver, delta: float, tick_count: int) -> void:
	for _i: int in range(tick_count):
		driver.step_time(delta);


## Detaches [param node] from `root` and frees it immediately so GPU solver RIDs release cleanly.
func _unmount_and_free(node: Node) -> void:
	if node == null:
		return;
	if node.get_parent() != null:
		node.get_parent().remove_child(node);
	node.free();


## Prints the header summary and non-zero phase breakdown table for a completed benchmark case.
func _print_case_report(cfg: JrBenchmarkConfig, telemetry: JrOrreryTelemetry) -> void:
	var all_stats: Dictionary[StringName, Dictionary] = telemetry.get_all_phase_stats();
	var total: Dictionary = all_stats[JrOrreryTelemetry.PHASE_UPDATE_ORBITS_TOTAL];
	var mean_us: float = float(total.get("mean_usec", 0.0));
	var p50_us: float = float(total.get("p50_usec", 0.0));
	var p95_us: float = float(total.get("p95_usec", 0.0));
	var max_us: int = int(total.get("max_usec", 0));

	var solver_mode: String = "GPU" if (telemetry.gpu_step_active or telemetry.gpu_forecast_active) else ("CPU-Forced" if cfg.force_cpu else "CPU/None");
	print("  Frames: %d | Mode: %s | Wells: %d | Rails: %d | Ballistics: %d (OnRails: %d, Unpinned: %d, Forecast: %d)" % [
		telemetry.get_frame_count(),
		solver_mode,
		telemetry.gravity_well_count,
		telemetry.rail_body_count,
		telemetry.ballistic_body_count,
		telemetry.on_rails_ballistic_count,
		telemetry.unpinned_ballistic_count,
		telemetry.forecast_body_count,
	]);
	print("  TOTAL update_orbits(): mean = %8.2f us (%.3f ms) | p50 = %8.2f us | p95 = %8.2f us | max = %d us" % [
		mean_us,
		mean_us / 1000.0,
		p50_us,
		p95_us,
		max_us,
	]);
	print("  %-34s %10s %10s %10s %8s" % ["Phase", "Mean (us)", "P50 (us)", "P95 (us)", "% Total"]);

	var accounted_sum_usec: int = 0;
	for phase: StringName in JrOrreryTelemetry.ORDERED_PHASES:
		if phase == JrOrreryTelemetry.PHASE_UPDATE_ORBITS_TOTAL:
			continue;
		var st: Dictionary = all_stats[phase];
		var phase_total: int = int(st.get("total_usec", 0));
		accounted_sum_usec += phase_total;
		if phase_total <= 0:
			continue;
		print("    %-32s %10.2f %10.2f %10.2f %7.1f%%" % [
			String(phase),
			float(st.get("mean_usec", 0.0)),
			float(st.get("p50_usec", 0.0)),
			float(st.get("p95_usec", 0.0)),
			float(st.get("pct_of_total", 0.0)),
		]);

	var total_sum_usec: int = int(total.get("total_usec", 0));
	var unmeasured_usec: int = maxi(0, total_sum_usec - accounted_sum_usec);
	if telemetry.get_frame_count() > 0 and unmeasured_usec > 0:
		var unmeasured_mean: float = float(unmeasured_usec) / float(telemetry.get_frame_count());
		var unmeasured_pct: float = (100.0 * float(unmeasured_usec) / float(total_sum_usec)) if total_sum_usec > 0 else 0.0;
		print("    %-32s %10.2f %10s %10s %7.1f%%" % [
			"(unmeasured_glue_and_timers)",
			unmeasured_mean,
			"-",
			"-",
			unmeasured_pct,
		]);
