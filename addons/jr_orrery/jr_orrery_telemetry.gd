@tool
class_name JrOrreryTelemetry
extends RefCounted
## Lightweight microsecond telemetry collector for [JrOrbitDriver] and [JrBallisticComputeSolver].
##
## Captures per-frame phase durations in microseconds across orbital rail evaluation,
## GPU/CPU ballistic integration, buffer packing/readback, Hill-sphere handoffs,
## Lowest Common Ancestor (LCA) coordinate resolution, follower signal emission, and
## trajectory forecasting. Designed so that when [code]null[/code] on [JrOrbitDriver],
## zero timing calls occur during normal gameplay.

## Total wall-clock duration of [method JrOrbitDriver.update_orbits].
const PHASE_UPDATE_ORBITS_TOTAL: StringName = &"update_orbits_total";

## Duration of [method JrOrbitDriver._propagate_units_and_parent_mu].
const PHASE_UNITS_AND_MU: StringName = &"units_and_mu";

## Duration of Keplerian analytical evaluation across all tracked [JrRailOrbiter] nodes.
const PHASE_RAIL_EVAL: StringName = &"rail_eval";

## Duration of pending [JrBallisticOrbiter] state initialization and local position capture.
const PHASE_BALLISTIC_INIT: StringName = &"ballistic_init";

## Duration of pre-step anchor validation and pinned-anchor synchronization.
const PHASE_BALLISTIC_ANCHOR_PRE: StringName = &"ballistic_anchor_pre";

## Duration of gravity-well list construction, packing, and solver context configuration.
const PHASE_BALLISTIC_WELL_PACK: StringName = &"ballistic_well_pack";

## Duration of packing [JrBallisticOrbiter] states into CPU float/byte buffers for GPU stepping.
const PHASE_SOLVER_STEP_PACK: StringName = &"solver_step_pack";

## Duration of GPU SSBO capacity checks, buffer uploads, and compute command-list recording.
const PHASE_SOLVER_STEP_UPLOAD_DISPATCH: StringName = &"solver_step_upload_dispatch";

## Duration of [method RenderingDevice.submit] and [method RenderingDevice.sync] for stepping.
const PHASE_SOLVER_STEP_SUBMIT_SYNC: StringName = &"solver_step_submit_sync";

## Duration of [method RenderingDevice.buffer_get_data] readback after GPU stepping.
const PHASE_SOLVER_STEP_READBACK: StringName = &"solver_step_readback";

## Duration of unpacking integrated float buffers back into [JrBallisticOrbiter] nodes.
const PHASE_SOLVER_STEP_UNPACK: StringName = &"solver_step_unpack";

## Duration of CPU fallback stepping when GPU stepping is disabled or unavailable.
const PHASE_SOLVER_STEP_CPU: StringName = &"solver_step_cpu";

## Duration of post-step Hill-sphere entry/exit hysteresis checks for unpinned bodies.
const PHASE_BALLISTIC_ANCHOR_POST: StringName = &"ballistic_anchor_post";

## Duration of post-step LCA anchor-to-parent transform synchronization on ballistic bodies.
const PHASE_BALLISTIC_TRANSFORM_SYNC: StringName = &"ballistic_transform_sync";

## Duration of perspective basis and LCA relative transform math across driver, rails, and ballistics.
const PHASE_SYSTEM_LCA_MATH: StringName = &"system_lca_math";

## Duration of [method JrOrreryElement.set_system_transform] and synchronous follower signals.
const PHASE_TRANSFORM_SIGNAL_EMIT: StringName = &"transform_signal_emit";

## Duration of sampling moving [JrRailOrbiter] gravity wells into the forecast timeline buffer.
const PHASE_FORECAST_TIMELINE_SAMPLE: StringName = &"forecast_timeline_sample";

## Duration of packing forecast body inputs for GPU trajectory forecasting.
const PHASE_SOLVER_FORECAST_PACK: StringName = &"solver_forecast_pack";

## Duration of GPU forecast buffer uploads and compute command-list dispatch.
const PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH: StringName = &"solver_forecast_upload_dispatch";

## Duration of [method RenderingDevice.submit] and [method RenderingDevice.sync] for forecasting.
const PHASE_SOLVER_FORECAST_SUBMIT_SYNC: StringName = &"solver_forecast_submit_sync";

## Duration of GPU forecast output buffer readback and polyline unpacking.
const PHASE_SOLVER_FORECAST_READBACK_UNPACK: StringName = &"solver_forecast_readback_unpack";

## Duration of CPU fallback trajectory forecast integration.
const PHASE_SOLVER_FORECAST_CPU: StringName = &"solver_forecast_cpu";

## Duration of projecting anchor-relative forecast polylines into the active perspective frame.
const PHASE_FORECAST_PERSPECTIVE_PROJECT: StringName = &"forecast_perspective_project";

## Canonical execution-order list of all instrumented phase identifiers.
const ORDERED_PHASES: Array[StringName] = [
	PHASE_UPDATE_ORBITS_TOTAL,
	PHASE_UNITS_AND_MU,
	PHASE_RAIL_EVAL,
	PHASE_BALLISTIC_INIT,
	PHASE_BALLISTIC_ANCHOR_PRE,
	PHASE_BALLISTIC_WELL_PACK,
	PHASE_SOLVER_STEP_PACK,
	PHASE_SOLVER_STEP_UPLOAD_DISPATCH,
	PHASE_SOLVER_STEP_SUBMIT_SYNC,
	PHASE_SOLVER_STEP_READBACK,
	PHASE_SOLVER_STEP_UNPACK,
	PHASE_SOLVER_STEP_CPU,
	PHASE_BALLISTIC_ANCHOR_POST,
	PHASE_BALLISTIC_TRANSFORM_SYNC,
	PHASE_SYSTEM_LCA_MATH,
	PHASE_TRANSFORM_SIGNAL_EMIT,
	PHASE_FORECAST_TIMELINE_SAMPLE,
	PHASE_SOLVER_FORECAST_PACK,
	PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH,
	PHASE_SOLVER_FORECAST_SUBMIT_SYNC,
	PHASE_SOLVER_FORECAST_READBACK_UNPACK,
	PHASE_SOLVER_FORECAST_CPU,
	PHASE_FORECAST_PERSPECTIVE_PROJECT,
];

## Most recently recorded count of tracked [JrRailOrbiter] bodies.
var rail_body_count: int = 0;

## Most recently recorded count of tracked [JrBallisticOrbiter] bodies.
var ballistic_body_count: int = 0;

## Most recently recorded count of [JrBallisticOrbiter] bodies coasting in Tier-1 on-rails mode.
var on_rails_ballistic_count: int = 0;

## Most recently recorded count of unpinned [JrBallisticOrbiter] bodies subject to Hill handoffs.
var unpinned_ballistic_count: int = 0;

## Most recently recorded count of [JrBallisticOrbiter] bodies with active runtime forecasts.
var forecast_body_count: int = 0;

## Most recently recorded count of gravity-well elements in the active solver pass.
var gravity_well_count: int = 0;

## Whether the GPU compute stepping pipeline was active on the most recent frame.
var gpu_step_active: bool = false;

## Whether the GPU compute forecast pipeline was active on the most recent frame.
var gpu_forecast_active: bool = false;

## Number of completed frames recorded since initialization or the last [method reset].
var _frame_count: int = 0;

## Accumulator of microsecond durations per phase for the frame currently in progress.
var _current_frame_usec: Dictionary[StringName, int] = {};

## Per-frame microsecond sample arrays indexed by phase [StringName].
var _samples: Dictionary[StringName, PackedInt64Array] = {};


## Initializes empty sample buffers for all phases in [constant ORDERED_PHASES].
func _init() -> void:
	reset();


## Clears all recorded frame samples and resets entity counters to zero.
func reset() -> void:
	_frame_count = 0;
	_current_frame_usec.clear();
	_samples.clear();
	for phase: StringName in ORDERED_PHASES:
		_current_frame_usec[phase] = 0;
		_samples[phase] = PackedInt64Array();
	rail_body_count = 0;
	ballistic_body_count = 0;
	on_rails_ballistic_count = 0;
	unpinned_ballistic_count = 0;
	forecast_body_count = 0;
	gravity_well_count = 0;
	gpu_step_active = false;
	gpu_forecast_active = false;


## Zeroes the per-phase accumulators at the start of [method JrOrbitDriver.update_orbits].
func begin_frame() -> void:
	for phase: StringName in ORDERED_PHASES:
		_current_frame_usec[phase] = 0;


## Adds [param elapsed_usec] microseconds to [param phase] for the active frame.
func add_phase_usec(phase: StringName, elapsed_usec: int) -> void:
	var prev: int = _current_frame_usec.get(phase, 0);
	_current_frame_usec[phase] = prev + maxi(0, elapsed_usec);


## Finalizes the active frame with [param total_usec] and commits all phase accumulators.
func end_frame(total_usec: int) -> void:
	_current_frame_usec[PHASE_UPDATE_ORBITS_TOTAL] = maxi(0, total_usec);
	for phase: StringName in ORDERED_PHASES:
		var arr: PackedInt64Array = _samples[phase];
		arr.append(_current_frame_usec.get(phase, 0));
		_samples[phase] = arr;
	_frame_count += 1;


## Updates the latest body population counters for diagnostic reporting.
func record_body_counts(
		rail_count: int,
		ballistic_count: int,
		on_rails_count: int,
		unpinned_count: int
) -> void:
	rail_body_count = rail_count;
	ballistic_body_count = ballistic_count;
	on_rails_ballistic_count = on_rails_count;
	unpinned_ballistic_count = unpinned_count;


## Updates the latest solver well/forecast counts and GPU activation flags.
func record_solver_state(
		well_count: int,
		forecast_count: int,
		step_gpu: bool,
		forecast_gpu: bool
) -> void:
	gravity_well_count = well_count;
	forecast_body_count = forecast_count;
	gpu_step_active = step_gpu;
	gpu_forecast_active = forecast_gpu;


## Returns the total number of frames committed via [method end_frame].
func get_frame_count() -> int:
	return _frame_count;


## Returns a copy of the raw microsecond sample series for [param phase].
func get_phase_samples(phase: StringName) -> PackedInt64Array:
	if not _samples.has(phase):
		return PackedInt64Array();
	return _samples[phase].duplicate();


## Computes summary statistics (`mean_usec`, `p50_usec`, `p95_usec`, `min_usec`,
## `max_usec`, `total_usec`, and `pct_of_total`) for [param phase].
func get_phase_stats(phase: StringName) -> Dictionary:
	var raw: PackedInt64Array = _samples.get(phase, PackedInt64Array());
	var count: int = raw.size();
	if count == 0:
		return _empty_stats_dict();

	var sorted: Array[int] = _to_sorted_int_array(raw);
	var sum_usec: int = 0;
	for val: int in sorted:
		sum_usec += val;

	var total_sum_usec: int = _sum_samples(_samples.get(PHASE_UPDATE_ORBITS_TOTAL, PackedInt64Array()));
	var mean_val: float = float(sum_usec) / float(count);
	var p50_val: float = _percentile_from_sorted(sorted, 0.50);
	var p95_val: float = _percentile_from_sorted(sorted, 0.95);
	var pct: float = (100.0 * float(sum_usec) / float(total_sum_usec)) if total_sum_usec > 0 else 0.0;

	return {
		"mean_usec": mean_val,
		"p50_usec": p50_val,
		"p95_usec": p95_val,
		"min_usec": sorted[0],
		"max_usec": sorted[count - 1],
		"total_usec": sum_usec,
		"pct_of_total": pct,
	};


## Returns summary statistics dictionaries for all phases in [constant ORDERED_PHASES].
func get_all_phase_stats() -> Dictionary[StringName, Dictionary]:
	var result: Dictionary[StringName, Dictionary] = {};
	for phase: StringName in ORDERED_PHASES:
		result[phase] = get_phase_stats(phase);
	return result;


## Returns a zero-initialized statistics dictionary when no frames have been recorded.
static func _empty_stats_dict() -> Dictionary:
	return {
		"mean_usec": 0.0,
		"p50_usec": 0.0,
		"p95_usec": 0.0,
		"min_usec": 0,
		"max_usec": 0,
		"total_usec": 0,
		"pct_of_total": 0.0,
	};


## Copies [param raw] into a sorted ascending [Array] of [int] values.
static func _to_sorted_int_array(raw: PackedInt64Array) -> Array[int]:
	var arr: Array[int] = [];
	arr.resize(raw.size());
	for i: int in range(raw.size()):
		arr[i] = raw[i];
	arr.sort();
	return arr;


## Sums all microsecond values in [param raw].
static func _sum_samples(raw: PackedInt64Array) -> int:
	var total: int = 0;
	for i: int in range(raw.size()):
		total += raw[i];
	return total;


## Computes the linear-interpolated percentile [param q] (`0.0` to `1.0`) from [param sorted].
static func _percentile_from_sorted(sorted: Array[int], q: float) -> float:
	var n: int = sorted.size();
	if n == 0:
		return 0.0;
	if n == 1:
		return float(sorted[0]);
	var pos: float = clampf(q, 0.0, 1.0) * float(n - 1);
	var lo: int = int(floorf(pos));
	var hi: int = mini(lo + 1, n - 1);
	var frac: float = pos - float(lo);
	return lerpf(float(sorted[lo]), float(sorted[hi]), frac);
