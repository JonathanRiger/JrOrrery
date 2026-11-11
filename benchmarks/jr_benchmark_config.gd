@tool
class_name JrBenchmarkConfig
extends RefCounted
## Configuration parameters defining a single automated benchmark run in [JrOrreryBenchmark].
##
## Encapsulates scenario selection, synthetic hierarchy population counts, GPU/CPU solver
## selection, perspective settings, and warmup/measurement tick budgets so benchmark functions
## remain concise and strongly typed.

## Runs an existing `.tscn` demo scenario from `addons/jr_orrery/demo/scenarios/`.
const KIND_DEMO: int = 0;

## Builds and benchmarks a synthetic ballistic N-body swarm around an 8-well planetary system.
const KIND_SYNTHETIC_SWARM: int = 1;

## Builds and benchmarks synthetic ballistic trajectory forecasting at varying body counts.
const KIND_SYNTHETIC_FORECAST: int = 2;

## Builds and benchmarks a multi-tier analytical [JrRailOrbiter] planet/moon/sub-moon tree.
const KIND_SYNTHETIC_RAILS: int = 3;

## Logical benchmark suite heading used to group related cases in the CLI report.
var suite_name: String = "";

## Human-readable identifier for this specific benchmark configuration.
var case_name: String = "";

## Scenario construction mode (`KIND_DEMO`, `KIND_SYNTHETIC_SWARM`, etc.).
var kind: int = KIND_DEMO;

## Resource path to the `.tscn` file when [member kind] is [constant KIND_DEMO].
var scenario_path: String = "";

## Forces [JrBallisticComputeSolver] onto its CPU fallback path when [code]true[/code].
var force_cpu: bool = false;

## Number of unrecorded warmup simulation steps executed prior to measurement.
var warmup_ticks: int = 16;

## Number of timed simulation steps recorded into [JrOrreryTelemetry].
var measured_ticks: int = 120;

## Simulation frame delta passed to [method JrOrbitDriver.step_time] on each tick.
var frame_delta: float = 1.0 / 60.0;

## Number of [JrRailOrbiter] gravity wells to generate in synthetic benchmarks.
var rail_count: int = 8;

## Number of [JrBallisticOrbiter] bodies to generate in synthetic benchmarks.
var ballistic_count: int = 0;

## Number of [JrBallisticOrbiter] bodies with `runtime_forecast_enabled = true`.
var forecast_count: int = 0;

## Number of integration output segments per forecasted body (`preview_steps`).
var forecast_steps: int = 192;

## Pins synthetic ballistic bodies to the central star (`pinned_anchor = driver`) when `true`.
var pin_anchors: bool = false;

## Tidal culling threshold assigned to [member JrOrbitDriver.tidal_cull_threshold].
var cull_threshold: float = 1e-4;

## Selects a planet-centered `SYNODIC_ORBITAL` perspective frame when `true`.
var synodic_perspective: bool = false;

## Selects a deep leaf moon as `perspective_node` to stress LCA tree walks when `true`.
var deep_perspective: bool = false;

## Retains `View2D`/`View3D` visual follower nodes in demo scenarios when `true`.
var attach_followers: bool = true;


## Creates a benchmark configuration for an existing `.tscn` demo scenario.
static func create_demo(
		label: String,
		path: String,
		cpu_fallback: bool,
		with_followers: bool = true
) -> JrBenchmarkConfig:
	var cfg := JrBenchmarkConfig.new();
	cfg.suite_name = "Demo Scenarios";
	cfg.case_name = label;
	cfg.kind = KIND_DEMO;
	cfg.scenario_path = path;
	cfg.force_cpu = cpu_fallback;
	cfg.attach_followers = with_followers;
	cfg.warmup_ticks = 6;
	cfg.measured_ticks = 20 if cpu_fallback else 40;
	return cfg;


## Creates a synthetic ballistic swarm scaling configuration.
static func create_swarm(
		label: String,
		bodies: int,
		cpu_fallback: bool,
		pinned: bool
) -> JrBenchmarkConfig:
	var cfg := JrBenchmarkConfig.new();
	cfg.suite_name = "Synthetic Ballistic Swarm Scaling";
	cfg.case_name = label;
	cfg.kind = KIND_SYNTHETIC_SWARM;
	cfg.ballistic_count = bodies;
	cfg.force_cpu = cpu_fallback;
	cfg.pin_anchors = pinned;
	cfg.cull_threshold = 1e-9;
	cfg.forecast_count = 0;
	cfg.warmup_ticks = 8 if cpu_fallback else 16;
	cfg.measured_ticks = _choose_swarm_ticks(bodies, cpu_fallback);
	return cfg;


## Creates a synthetic ballistic trajectory forecast scaling configuration.
static func create_forecast(
		label: String,
		forecasts: int,
		cpu_fallback: bool,
		synodic: bool
) -> JrBenchmarkConfig:
	var cfg := JrBenchmarkConfig.new();
	cfg.suite_name = "Synthetic Forecast Scaling (192 steps/body, 8 wells)";
	cfg.case_name = label;
	cfg.kind = KIND_SYNTHETIC_FORECAST;
	cfg.ballistic_count = forecasts;
	cfg.forecast_count = forecasts;
	cfg.forecast_steps = 192;
	cfg.force_cpu = cpu_fallback;
	cfg.synodic_perspective = synodic;
	cfg.pin_anchors = false;
	cfg.cull_threshold = 1e-6;
	cfg.warmup_ticks = 6 if cpu_fallback else 12;
	cfg.measured_ticks = _choose_forecast_ticks(forecasts, cpu_fallback);
	return cfg;


## Creates a synthetic analytical [JrRailOrbiter] hierarchy scaling configuration.
static func create_rails(
		label: String,
		orbiters: int,
		deep_persp: bool
) -> JrBenchmarkConfig:
	var cfg := JrBenchmarkConfig.new();
	cfg.suite_name = "Synthetic Rail Hierarchy Scaling";
	cfg.case_name = label;
	cfg.kind = KIND_SYNTHETIC_RAILS;
	cfg.rail_count = orbiters;
	cfg.deep_perspective = deep_persp;
	cfg.warmup_ticks = 4;
	cfg.measured_ticks = 15 if orbiters >= 1024 else 30;
	return cfg;


## Chooses a fast, statistically stable tick count for swarm runs based on body count and device.
static func _choose_swarm_ticks(bodies: int, cpu_fallback: bool) -> int:
	if cpu_fallback:
		if bodies >= 4096:
			return 6;
		if bodies >= 1024:
			return 12;
		return 25;
	if bodies >= 4096:
		return 15;
	return 30;


## Chooses a fast, statistically stable tick count for forecast runs based on count and device.
static func _choose_forecast_ticks(forecasts: int, cpu_fallback: bool) -> int:
	if cpu_fallback:
		return 6 if forecasts >= 16 else 12;
	if forecasts >= 64:
		return 10;
	return 25;
