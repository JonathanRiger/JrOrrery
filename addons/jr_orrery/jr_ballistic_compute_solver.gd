@tool
class_name JrBallisticComputeSolver
extends RefCounted
## Batched GPU compute-shader hybrid Universal-Variable Kepler & Battin f(q) tidal
## integrator and trajectory forecaster for [JrBallisticOrbiter] nodes.
##
## Manages a local [RenderingDevice], compiles [code]ballistic_gravity.glsl[/code]
## and [code]ballistic_forecast.glsl[/code], maintains dynamically sized [code]std430[/code]
## storage buffers (SSBOs), and executes anchor-relative Universal-Variable Kick-Drift-Kick
## with Lowest Common Ancestor (LCA) inter-well vector resolution, square-root-free
## tidal culling, and Tier-1 epoch-frozen analytical coasting. Automatically falls back
## to identical CPU integration when running in headless environments.

## Path to the per-frame stepping GLSL compute shader resource.
const SHADER_PATH: String = "res://addons/jr_orrery/shaders/ballistic_gravity.glsl"

## Path to the forward trajectory forecast GLSL compute shader resource.
const FORECAST_SHADER_PATH: String = "res://addons/jr_orrery/shaders/ballistic_forecast.glsl"

## Compute shader workgroup X dimension matching [code]local_size_x[/code] in GLSL.
const WORKGROUP_SIZE_X: int = 64

## Number of 32-bit floats per packed [code]BallisticBody[/code] or [code]ForecastBodyInput[/code]
## (4 x vec4 = 64 bytes).
const FLOATS_PER_BODY: int = 16

## Number of [Vector4] entries per packed [code]BallisticBody[/code] struct (4 x vec4 = 64 bytes).
const VEC4S_PER_BODY: int = 4;

## Byte stride per packed body struct (64 bytes).
const BYTES_PER_BODY: int = 64;

## Number of [Vector4] entries per packed [code]BallisticEpoch[/code] struct (2 x vec4 = 32 bytes).
const VEC4S_PER_EPOCH: int = 2;

## Byte stride per packed [code]BallisticEpoch[/code] struct (32 bytes).
const BYTES_PER_EPOCH: int = 32;

## Header byte size of [code]StepOutputBuffer[/code] (`6 x uint32 = 24 bytes = 2 x Vector3`).
const STEP_OUT_HEADER_BYTES: int = 24;

## Packed `(pos, vel)` output byte stride per body in [code]StepOutputBuffer[/code] (`24 bytes`).
const STEP_OUT_PV_BYTES_PER_BODY: int = 24;

## Sparse event record byte stride in [code]StepOutputBuffer[/code] (`2 x uint32 = 8 bytes`).
const STEP_OUT_EVENT_BYTES: int = 8;

## Maximum elapsed simulation seconds from [member _gpu_time_origin] before rebasing Tier-1 epochs.
const MAX_GPU_TIME_ORIGIN_SPAN: float = 65536.0;

## Number of 32-bit floats per packed [code]GravityWell[/code] struct (3 x vec4 = 48 bytes).
const FLOATS_PER_WELL: int = 12;

## Byte stride per packed [code]GravityWell[/code] struct (48 bytes).
const BYTES_PER_WELL: int = 48;

## Number of 32-bit floats per [code]vec4[/code] SSBO entry (16 bytes).
const FLOATS_PER_VEC4: int = 4

## Byte stride per [code]vec4[/code] SSBO element (16 bytes).
const BYTES_PER_VEC4: int = 16

## Number of 32-bit floats per timeline well sample (2 x vec4 = 32 bytes).
const FLOATS_PER_TIMELINE_WELL: int = 8

## Byte size of the 16-byte aligned forecast push constant block.
const PUSH_CONSTANT_BYTES: int = 16;

## Byte size of the 32-byte aligned stepping push constant block.
const STEP_PUSH_CONSTANT_BYTES: int = 32;

## Internal Kick-Drift-Kick substeps per forecast output segment (matches
## [code]ballistic_forecast.glsl[/code]).
const FORECAST_INTERNAL_SUBSTEPS: int = 2

## Hill-sphere multiplier for automatic anchor entry handoffs (~0.98 x r_Hill).
const HILL_ENTRY_MULTIPLIER: float = 0.98

## Hill-sphere multiplier for automatic anchor exit handoffs (~1.05 x r_Hill).
const HILL_EXIT_MULTIPLIER: float = 1.05;

## Minimum body count required to amortize D3D12/Vulkan local [RenderingDevice] submit+sync latency
## during per-frame stepping; smaller batches execute identically on the CPU without fence stalls.
const MIN_GPU_STEP_BODIES: int = 32;

## Minimum body count required to amortize local [RenderingDevice] submit+sync latency during
## forward trajectory forecasting regardless of step count.
const MIN_GPU_FORECAST_BODIES: int = 2;

## Minimum single-body `(preview_steps * well_count)` work product required to amortize GPU
## forecast dispatch when forecasting a single body.
const MIN_GPU_FORECAST_WORK: int = 1000;

## Shared ref-counted local [RenderingDevice] reused across active solver instances.
static var _shared_rd: RenderingDevice = null

## Shared compiled stepping compute shader [RID] on [member _shared_rd].
static var _shared_shader_rid := RID()

## Shared stepping compute pipeline [RID] on [member _shared_rd].
static var _shared_pipeline_rid := RID()

## Shared compiled forecast compute shader [RID] on [member _shared_rd].
static var _shared_forecast_shader_rid := RID()

## Shared forecast compute pipeline [RID] on [member _shared_rd].
static var _shared_forecast_pipeline_rid := RID()

## Number of active solver instances currently holding a reference to [member _shared_rd].
static var _shared_rd_ref_count: int = 0

## Local [RenderingDevice] used to execute compute dispatches off the main frame graph.
var _rd: RenderingDevice = null

## Whether this solver instance currently holds a reference to [member _shared_rd].
var _holds_shared_rd: bool = false

## Whether GPU initialization has already been attempted.
var _gpu_init_attempted: bool = false

## Whether the stepping GPU compute pipeline compiled successfully.
var _gpu_available: bool = false

## Whether the trajectory forecast GPU compute pipeline compiled successfully.
var _forecast_gpu_available: bool = false

## Compiled stepping compute shader [RID] on [member _rd].
var _shader_rid := RID()

## Stepping compute pipeline [RID] on [member _rd].
var _pipeline_rid := RID()

## Storage buffer [RID] for stepping Binding 0 ([code]BallisticBodyBuffer[/code]).
var _body_buffer_rid := RID();

## Storage buffer [RID] for stepping Binding 1 ([code]GravityWellBuffer[/code]).
var _well_buffer_rid := RID();

## Storage buffer [RID] for stepping Binding 2 ([code]BallisticEpochBuffer[/code]).
var _epoch_buffer_rid := RID();

## Storage buffer [RID] for stepping Binding 3 ([code]StepOutputBuffer[/code]).
var _output_buffer_rid := RID();

## Uniform set [RID] for stepping bindings 0, 1, 2, and 3.
var _uniform_set_rid := RID();

## Allocated body capacity of [member _body_buffer_rid].
var _body_capacity: int = 0

## Allocated gravity-well capacity of [member _well_buffer_rid].
var _well_capacity: int = 0

## Compiled forecast compute shader [RID] on [member _rd].
var _forecast_shader_rid := RID()

## Forecast compute pipeline [RID] on [member _rd].
var _forecast_pipeline_rid := RID()

## Storage buffer [RID] for forecast Binding 0 ([code]ForecastBodyInputBuffer[/code]).
var _forecast_body_buffer_rid := RID()

## Storage buffer [RID] for forecast Binding 1 ([code]WellTimelineBuffer[/code]).
var _forecast_timeline_buffer_rid := RID()

## Storage buffer [RID] for forecast Binding 2 ([code]ForecastPointOutputBuffer[/code]).
var _forecast_output_buffer_rid := RID()

## Uniform set [RID] for forecast bindings 0, 1, and 2.
var _forecast_uniform_set_rid := RID()

## Allocated body capacity of [member _forecast_body_buffer_rid].
var _forecast_body_capacity: int = 0

## Allocated vec4 entry capacity of [member _forecast_timeline_buffer_rid].
var _forecast_timeline_capacity: int = 0

## Allocated vec4 point capacity of [member _forecast_output_buffer_rid].
var _forecast_output_capacity: int = 0

## Current simulation timestamp supplied for the active stepping pass.
var _step_current_time: float = 0.0

## Squared tidal culling threshold supplied for the active stepping pass.
var _step_cull_threshold_sq: float = 1e-8

## Ordered array of gravity-well elements for the active stepping pass.
var _step_well_elements: Array[JrOrreryElement] = []

## Lookup mapping each gravity-well [JrOrreryElement] to its index in [member _step_well_elements].
var _step_well_indices: Dictionary[JrOrreryElement, int] = {}

## Optional telemetry sink receiving microsecond phase timings when non-null.
var _telemetry: JrOrreryTelemetry = null;

## When [code]true[/code], forces CPU integration and forecasting even if a GPU is available.
var _force_cpu: bool = false;

## Precomputed per-well analytical safe coasting radii for the active stepping pass.
var _step_safe_radii: PackedFloat32Array = PackedFloat32Array();

## Reusable staging buffer for packing body inputs into [member _body_buffer_rid].
var _packed_body_vec4s: PackedVector4Array = PackedVector4Array();

## Reusable staging buffer for packing Tier-1 epoch inputs into [member _epoch_buffer_rid].
var _packed_epoch_vec4s: PackedVector4Array = PackedVector4Array();

## Whether the most recent [method step_bodies] pass evaluated Hill-sphere handoffs on the GPU.
var _gpu_handoffs_evaluated: bool = false;

## Body indices in [member _ballistic_sequence] that triggered a GPU Hill-sphere handoff this step.
var _gpu_handoff_body_indices: PackedInt32Array = PackedInt32Array();

## Target gravity-well indices paired with [member _gpu_handoff_body_indices].
var _gpu_handoff_target_wells: PackedInt32Array = PackedInt32Array();

## Whether [member _body_buffer_rid] currently holds authoritative resident GPU state from the last step.
var _gpu_ssbo_valid: bool = false;

## Whether one or more tracked [JrBallisticOrbiter] nodes mutated state since the last GPU upload.
var _any_gpu_body_dirty: bool = true;

## Body count uploaded into the active resident [member _body_buffer_rid].
var _resident_body_count: int = 0;

## 64-bit simulation timestamp origin against which GPU `(time_hi, time_lo)` offsets are measured.
var _gpu_time_origin: float = 0.0;

## Ordered gravity-well reference list matching `anchor_idx` values in [member _body_buffer_rid].
var _resident_well_elements: Array[JrOrreryElement] = [];

## Reusable 64-byte staging slice for sparse single-body updates into [member _body_buffer_rid].
var _single_body_vec4s: PackedVector4Array = PackedVector4Array();

## Reusable 32-byte staging slice for sparse single-body updates into [member _epoch_buffer_rid].
var _single_epoch_vec4s: PackedVector4Array = PackedVector4Array();

## Reusable 2-float scratch array for exact IEEE-754 `float64 -> (hi, lo)` `binary32` splitting.
var _f32_split_scratch: PackedFloat32Array = PackedFloat32Array([0.0, 0.0]);

## Reusable 4-byte zero payload used to clear `step_out.words[0]` (`event_count`) before dispatch.
var _zero_u32_bytes: PackedByteArray = PackedByteArray([0, 0, 0, 0]);

## Sparse queue of [JrBallisticOrbiter] instances mutated since the previous GPU step.
var _dirty_gpu_bodies: Array[JrBallisticOrbiter] = [];

## Reusable scratch output position for allocation-free CPU Keplerian drift.
var _drift_r: Vector3 = Vector3.ZERO;

## Reusable scratch output velocity for allocation-free CPU Keplerian drift.
var _drift_v: Vector3 = Vector3.ZERO;

## Reusable scratch acceleration output for allocation-free CPU tidal evaluation.
var _pert_accel_out: Vector3 = Vector3.ZERO;

## Reusable scratch activity flag for allocation-free CPU tidal evaluation.
var _pert_active_out: bool = false;

## Cached timeline array reference during CPU forecast integration.
var _fc_timeline: PackedFloat32Array = PackedFloat32Array();

## Cached gravity-well count during CPU forecast integration.
var _fc_well_count: int = 0;

## Cached maximum segment index (`total_samples - 1`) during CPU forecast integration.
var _fc_max_seg: int = 0;

## Cached squared tidal culling threshold during CPU forecast integration.
var _fc_cull_sq: float = 0.0;

## Cached per-well gravitational parameters (`mu`) during CPU forecast integration.
var _fc_well_mu: PackedFloat32Array = PackedFloat32Array();

## Cached per-well parent indices during CPU forecast integration.
var _fc_well_parent: PackedInt32Array = PackedInt32Array();

## Cached per-well child well index lists during CPU forecast integration.
var _fc_well_children: Array[PackedInt32Array] = [];

## Scratch anchor-relative position during CPU forecast SOI handoff checks.
var _fc_cur_r: Vector3 = Vector3.ZERO;

## Scratch anchor-relative velocity during CPU forecast SOI handoff checks.
var _fc_cur_v: Vector3 = Vector3.ZERO;

## Scratch anchor index during CPU forecast SOI handoff checks.
var _fc_cur_anchor: int = 0;


## Propagates [param r0] and [param v0] analytically over [param dt] under central
## gravitational parameter [param mu] using dimensionless Universal-Variable iteration.
static func kepler_drift_cpu(r0: Vector3, v0: Vector3, mu: float, dt: float) -> Array[Vector3]:
	var r_scale: float = r0.length()
	if r_scale <= 1e-9 or mu <= 0.0 or absf(dt) <= 1e-12:
		return [r0 + v0 * dt, v0]

	var v_scale: float = sqrt(mu / r_scale)
	var inv_v_scale: float = 1.0 / v_scale
	var r_unit: Vector3 = r0 / r_scale
	var v_dim: Vector3 = v0 * inv_v_scale
	var dt_dim: float = dt * (v_scale / r_scale)

	var sigma0: float = r_unit.dot(v_dim)
	var alpha: float = 2.0 - v_dim.length_squared()

	if alpha > 1e-6:
		var period_dim: float = TAU / (alpha * sqrt(alpha))
		if absf(dt_dim) > period_dim:
			dt_dim = fmod(dt_dim, period_dim)
		if dt_dim > 0.5 * period_dim:
			dt_dim -= period_dim
		elif dt_dim < -0.5 * period_dim:
			dt_dim += period_dim

	var chi: float = (dt_dim * alpha) if alpha > 1e-6 else dt_dim
	chi = clampf(chi, -1e5, 1e5)

	for _iter: int in range(20):
		var chi2: float = chi * chi
		var z: float = alpha * chi2
		var cz: float = _stumpff_c(z)
		var sz: float = _stumpff_s(z)
		var chi3: float = chi2 * chi

		var f_val: float = sigma0 * chi2 * cz + (1.0 - alpha) * chi3 * sz + chi - dt_dim
		var df: float = sigma0 * chi * (1.0 - z * sz) + (1.0 - alpha) * chi2 * cz + 1.0
		var safe_df: float = maxf(1e-9, df) if df >= 0.0 else minf(-1e-9, df)
		var delta: float = f_val / safe_df
		chi -= delta
		if absf(delta) < 1e-10:
			break

	var chi2_f: float = chi * chi
	var z_f: float = alpha * chi2_f
	var cz_f: float = _stumpff_c(z_f)
	var sz_f: float = _stumpff_s(z_f)
	var chi3_f: float = chi2_f * chi

	var r_dim: float = maxf(
			1e-9,
			1.0 + sigma0 * chi * (1.0 - z_f * sz_f) + (1.0 - alpha) * chi2_f * cz_f
	)
	var f_lag: float = 1.0 - chi2_f * cz_f
	var g_lag: float = dt_dim - chi3_f * sz_f
	var df_lag: float = (chi * (z_f * sz_f - 1.0)) / r_dim
	var dg_lag: float = 1.0 - (chi2_f * cz_f) / r_dim

	var r1: Vector3 = (f_lag * r_unit + g_lag * v_dim) * r_scale
	var v1: Vector3 = (df_lag * r_unit + dg_lag * v_dim) * v_scale
	return [r1, v1]


## Computes the Hill sphere radius [code]q * (mu_body / (3 * mu_parent))^(1/3)[/code]
## for a body with periapsis [param periapsis_dist], gravity [param mu_body], and parent gravity
## [param mu_parent].
static func compute_hill_radius(periapsis_dist: float, mu_body: float, mu_parent: float) -> float:
	if periapsis_dist <= 0.0 or mu_body <= 0.0 or mu_parent <= 0.0:
		return 0.0
	return periapsis_dist * pow(mu_body / (3.0 * mu_parent), 1.0 / 3.0)


## Frees the shared pipelines, shaders, and local [RenderingDevice] when the last reference is
## released.
static func _dispose_shared_rd() -> void:
	if _shared_rd == null:
		return
	if _shared_pipeline_rid.is_valid():
		_shared_rd.free_rid(_shared_pipeline_rid)
		_shared_pipeline_rid = RID()
	if _shared_shader_rid.is_valid():
		_shared_rd.free_rid(_shared_shader_rid)
		_shared_shader_rid = RID()
	if _shared_forecast_pipeline_rid.is_valid():
		_shared_rd.free_rid(_shared_forecast_pipeline_rid)
		_shared_forecast_pipeline_rid = RID()
	if _shared_forecast_shader_rid.is_valid():
		_shared_rd.free_rid(_shared_forecast_shader_rid)
		_shared_forecast_shader_rid = RID()
	_shared_rd.free()
	_shared_rd = null


## Returns the smallest power of two greater than or equal to [param value] (minimum 8).
static func _next_power_of_two(value: int) -> int:
	var capacity: int = 8
	while capacity < value:
		capacity *= 2
	return capacity


## Evaluates Stumpff C(z) on the CPU.
static func _stumpff_c(z: float) -> float:
	if absf(z) < 1e-4:
		return 0.5 - z * (1.0 / 24.0 - z * (1.0 / 720.0))
	if z > 0.0:
		var sz: float = sqrt(z)
		return (1.0 - cos(sz)) / z
	var sz_neg: float = sqrt(-z)
	return (cosh(sz_neg) - 1.0) / (-z)


## Evaluates Stumpff S(z) on the CPU.
static func _stumpff_s(z: float) -> float:
	if absf(z) < 1e-4:
		return (1.0 / 6.0) - z * ((1.0 / 120.0) - z * (1.0 / 5040.0))
	if z > 0.0:
		var sz: float = sqrt(z)
		return (sz - sin(sz)) / (z * sz)
	var sz_neg: float = sqrt(-z)
	return (sinh(sz_neg) - sz_neg) / ((-z) * sz_neg)


## Evaluates thrust acceleration on the CPU for [param ext_accel] and [param is_orbital].
static func _evaluate_thrust_cpu(
		r_rel: Vector3,
		v_rel: Vector3,
		ext_accel: Vector3,
		is_orbital: bool
) -> Vector3:
	if ext_accel == Vector3.ZERO:
		return Vector3.ZERO
	if not is_orbital:
		return ext_accel
	var rtn: Basis = JrBallisticOrbiter.compute_orbital_rtn_basis(r_rel, v_rel)
	return rtn * ext_accel


## Attaches or detaches a [JrOrreryTelemetry] collector for microsecond solver phase profiling.
func set_telemetry(telemetry: JrOrreryTelemetry) -> void:
	_telemetry = telemetry;


## Returns the currently attached [JrOrreryTelemetry] collector, or [code]null[/code] if disabled.
func get_telemetry() -> JrOrreryTelemetry:
	return _telemetry;


## Enables or disables forced CPU fallback execution for stepping and trajectory forecasting.
func set_force_cpu(force_cpu: bool) -> void:
	_force_cpu = force_cpu;


## Returns [code]true[/code] when CPU fallback execution is explicitly forced.
func is_force_cpu() -> bool:
	return _force_cpu;


## Returns [code]true[/code] if the stepping GPU compute pipeline is active on a local
## [RenderingDevice].
func is_gpu_active() -> bool:
	if _force_cpu:
		return false;
	_ensure_gpu_initialized()
	return _gpu_available


## Returns [code]true[/code] if the trajectory forecast GPU compute pipeline is active.
func is_forecast_gpu_active() -> bool:
	if _force_cpu:
		return false;
	_ensure_gpu_initialized()
	return _forecast_gpu_available


## Configures the current simulation timestamp, tidal culling threshold, and ordered
## gravity-well element list prior to calling [method step_bodies].
func configure_step_context(
		current_time: float,
		cull_threshold: float,
		well_elements: Array[JrOrreryElement]
) -> void:
	_step_current_time = current_time;
	var clamped_cull: float = maxf(0.0, cull_threshold);
	_step_cull_threshold_sq = clamped_cull * clamped_cull;
	_step_well_elements = well_elements;
	_step_well_indices.clear();
	for i: int in range(well_elements.size()):
		var elem: JrOrreryElement = well_elements[i];
		if elem != null:
			elem._runtime_well_index = i;
			_step_well_indices[elem] = i;


## Releases all GPU storage buffers, uniform sets, and (when the last active solver
## is disposed) the shared pipelines, shaders, and local [RenderingDevice] to prevent RID leaks.
func dispose() -> void:
	if _rd == null:
		_gpu_available = false
		_forecast_gpu_available = false
		_gpu_init_attempted = false
		return

	_dispose_stepping_rids()
	_dispose_forecast_rids()
	_shader_rid = RID()
	_pipeline_rid = RID()
	_forecast_shader_rid = RID()
	_forecast_pipeline_rid = RID()

	if _holds_shared_rd:
		_holds_shared_rd = false
		_shared_rd_ref_count = maxi(0, _shared_rd_ref_count - 1)
		if _shared_rd_ref_count == 0 and _shared_rd != null:
			_dispose_shared_rd()

	_rd = null
	_body_capacity = 0
	_well_capacity = 0
	_forecast_body_capacity = 0
	_forecast_timeline_capacity = 0
	_forecast_output_capacity = 0
	_gpu_available = false
	_forecast_gpu_available = false
	_gpu_init_attempted = false


## Integrates all [param bodies] over [param dt] with up to [param substeps] adaptive
## Universal-Variable Kick-Drift-Kick substeps under the influence of [param packed_wells].
func step_bodies(
		bodies: Array[JrBallisticOrbiter],
		packed_wells: PackedFloat32Array,
		dt: float,
		substeps: int = 1
) -> void:
	_gpu_handoffs_evaluated = false;
	_gpu_handoff_body_indices.clear();
	_gpu_handoff_target_wells.clear();
	if bodies.is_empty() or is_zero_approx(dt):
		return;

	var clamped_substeps: int = maxi(1, substeps);
	if is_gpu_active() and bodies.size() >= MIN_GPU_STEP_BODIES:
		_step_bodies_gpu(bodies, packed_wells, dt, clamped_substeps);
	else:
		_gpu_ssbo_valid = false;
		_step_bodies_cpu(bodies, packed_wells, dt, clamped_substeps);


## Invalidates the resident GPU stepping SSBO so the next step performs a full body upload.
func invalidate_resident_step_buffer() -> void:
	_gpu_ssbo_valid = false;
	_any_gpu_body_dirty = true;
	_dirty_gpu_bodies.clear();


## Records [param body] for `O(1)` sparse 64-byte re-upload into the resident GPU stepping SSBO.
func record_dirty_gpu_body(body: JrBallisticOrbiter) -> void:
	_any_gpu_body_dirty = true;
	if body != null:
		_dirty_gpu_bodies.append(body);


## Returns `true` if the most recent [method step_bodies] pass evaluated Hill handoffs on the GPU.
func has_gpu_step_handoffs() -> bool:
	return _gpu_handoffs_evaluated;


## Returns the indices of bodies that crossed a Hill-sphere threshold during the last GPU step.
func get_gpu_handoff_body_indices() -> PackedInt32Array:
	return _gpu_handoff_body_indices;


## Returns the target gravity-well indices paired with [method get_gpu_handoff_body_indices].
func get_gpu_handoff_target_wells() -> PackedInt32Array:
	return _gpu_handoff_target_wells;


## Computes forward-integrated anchor-relative trajectory polylines ([code]xyz = r_rel, w =
## anchor_well_idx[/code])
## for all [JrBallisticOrbiter] nodes in [param config].
func compute_forecasts_anchor_relative(
		config: JrBallisticForecastConfig
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var results: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {};
	var valid_bodies: Array[JrBallisticOrbiter] = [];
	var max_duration: float = 0.0;

	for body: JrBallisticOrbiter in config.bodies:
		if body != null:
			valid_bodies.append(body);
			max_duration = maxf(max_duration, body.preview_duration);

	if valid_bodies.is_empty():
		return results;

	var segments: int = maxi(8, config.timeline_segments);
	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	var well_list: Array[JrOrreryElement] = _build_forecast_well_list(config);
	var timeline_floats: PackedFloat32Array = _sample_well_timeline(
			config,
			well_list,
			max_duration,
			segments
	);
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_FORECAST_TIMELINE_SAMPLE,
				Time.get_ticks_usec() - t0
		);
	var inv_timeline_dt: float = float(segments) / max_duration if max_duration > 0.0 else 0.0;

	if should_use_gpu_for_forecast(valid_bodies, well_list.size()):
		return _compute_forecasts_gpu(valid_bodies, well_list, timeline_floats, inv_timeline_dt);
	return _compute_forecasts_cpu(valid_bodies, well_list, timeline_floats, inv_timeline_dt);


## Returns `true` when [param bodies] and [param well_count] amortize GPU forecast dispatch.
func should_use_gpu_for_forecast(
		bodies: Array[JrBallisticOrbiter],
		well_count: int
) -> bool:
	if not is_forecast_gpu_active() or bodies.is_empty():
		return false;
	if bodies.size() >= MIN_GPU_FORECAST_BODIES:
		return true;
	var total_work: int = maxi(1, bodies[0].preview_steps) * maxi(1, well_count);
	return total_work >= MIN_GPU_FORECAST_WORK;


## Computes forward-integrated inertial trajectory polylines for all [JrBallisticOrbiter]
## nodes in [param config] starting from [member JrBallisticForecastConfig.start_time].
func compute_forecasts_inertial(
		config: JrBallisticForecastConfig
) -> Dictionary[JrBallisticOrbiter, PackedVector3Array]:
	var rel_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			compute_forecasts_anchor_relative(config)
	)
	var well_list: Array[JrOrreryElement] = _build_forecast_well_list(config)
	var inertial_results: Dictionary[JrBallisticOrbiter, PackedVector3Array] = {}

	for body: JrBallisticOrbiter in rel_map.keys():
		var rel_pts: PackedVector4Array = rel_map[body]
		var count: int = rel_pts.size()
		var out_pts: PackedVector3Array = []
		out_pts.resize(count)
		var step_dt: float = body.preview_duration / float(maxi(1, count - 1))
		for k: int in range(count):
			var pt4: Vector4 = rel_pts[k]
			var anchor_idx := int(roundf(pt4.w))
			var eval_t: float = config.start_time + float(k) * step_dt
			var anchor_origin: Vector3 = _evaluate_well_inertial_origin_at(
					anchor_idx,
					well_list,
					config,
					eval_t
			)
			out_pts[k] = anchor_origin + Vector3(pt4.x, pt4.y, pt4.z)
		inertial_results[body] = out_pts

	return inertial_results


## Returns the ordered array of gravity-well [JrOrreryElement] nodes for [param config]
## (index 0 = [member JrBallisticForecastConfig.driver_element], followed by valid
## [member JrBallisticForecastConfig.orbit_sequence]).
func get_forecast_well_elements(config: JrBallisticForecastConfig) -> Array[JrOrreryElement]:
	return _build_forecast_well_list(config)


## Evaluates a single-body forward-integrated N-body trajectory preview in inertial
## driver space using [param config], sampling moving [JrRailOrbiter] gravity wells.
func compute_preview_trajectory(config: JrBallisticPreviewConfig) -> PackedVector3Array:
	var step_count: int = maxi(1, config.steps)
	var total_points: int = step_count + 1
	var points: PackedVector3Array = []
	points.resize(total_points)

	var pos: Vector3 = config.initial_inertial_position
	var vel: Vector3 = config.initial_inertial_velocity
	points[0] = pos

	var step_dt: float = config.duration / float(step_count)
	if is_zero_approx(step_dt):
		for i: int in range(1, total_points):
			points[i] = pos
		return points

	var wells_start: PackedVector4Array = _sample_system_wells_at_time(config, config.start_time)
	var accel: Vector3 = config.external_acceleration + _sum_wells_acceleration(pos, wells_start)

	for step_idx: int in range(step_count):
		var half_vel: Vector3 = vel + 0.5 * step_dt * accel
		pos += step_dt * half_vel

		var next_time: float = config.start_time + float(step_idx + 1) * step_dt
		var wells_end: PackedVector4Array = _sample_system_wells_at_time(config, next_time)
		var next_accel: Vector3 = config.external_acceleration + _sum_wells_acceleration(
				pos,
				wells_end
		)
		vel = half_vel + 0.5 * step_dt * next_accel
		accel = next_accel
		points[step_idx + 1] = pos

	return points


## Lazily initializes or attaches to the shared local [RenderingDevice] and compiled compute
## pipelines.
func _ensure_gpu_initialized() -> void:
	if _gpu_init_attempted:
		return
	_gpu_init_attempted = true

	if RenderingServer.get_current_rendering_driver_name() == "dummy":
		_gpu_available = false
		_forecast_gpu_available = false
		return

	if _shared_rd == null:
		_shared_rd = RenderingServer.create_local_rendering_device()
		if _shared_rd == null:
			_gpu_available = false
			_forecast_gpu_available = false
			return
		_rd = _shared_rd
		_init_stepping_pipeline()
		_init_forecast_pipeline()
		_shared_shader_rid = _shader_rid
		_shared_pipeline_rid = _pipeline_rid
		_shared_forecast_shader_rid = _forecast_shader_rid
		_shared_forecast_pipeline_rid = _forecast_pipeline_rid
		if not _shared_pipeline_rid.is_valid() and not _shared_forecast_pipeline_rid.is_valid():
			_dispose_shared_rd()
			_rd = null
			return

	_rd = _shared_rd
	_shader_rid = _shared_shader_rid
	_pipeline_rid = _shared_pipeline_rid
	_forecast_shader_rid = _shared_forecast_shader_rid
	_forecast_pipeline_rid = _shared_forecast_pipeline_rid
	_gpu_available = _pipeline_rid.is_valid()
	_forecast_gpu_available = _forecast_pipeline_rid.is_valid()
	_shared_rd_ref_count += 1
	_holds_shared_rd = true


## Compiles [constant SHADER_PATH] and creates [member _pipeline_rid].
func _init_stepping_pipeline() -> void:
	var spirv: RDShaderSPIRV = _load_or_compile_spirv(SHADER_PATH)
	if (
			spirv == null
			or not spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE).is_empty()
	):
		return

	_shader_rid = _rd.shader_create_from_spirv(spirv)
	if not _shader_rid.is_valid():
		return

	_pipeline_rid = _rd.compute_pipeline_create(_shader_rid)
	_gpu_available = _pipeline_rid.is_valid()


## Compiles [constant FORECAST_SHADER_PATH] and creates [member _forecast_pipeline_rid].
func _init_forecast_pipeline() -> void:
	var spirv: RDShaderSPIRV = _load_or_compile_spirv(FORECAST_SHADER_PATH)
	if (
			spirv == null
			or not spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE).is_empty()
	):
		return

	_forecast_shader_rid = _rd.shader_create_from_spirv(spirv)
	if not _forecast_shader_rid.is_valid():
		return

	_forecast_pipeline_rid = _rd.compute_pipeline_create(_forecast_shader_rid)
	_forecast_gpu_available = _forecast_pipeline_rid.is_valid()


## Compiles raw GLSL source from [param path] on [member _rd], or falls back to pre-imported
## SPIR-V.
func _load_or_compile_spirv(path: String) -> RDShaderSPIRV:
	if FileAccess.file_exists(path):
		var raw_text: String = FileAccess.get_file_as_string(path)
		var clean_glsl: String = raw_text.replace("#[compute]", "")
		var shader_source := RDShaderSource.new()
		shader_source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
		shader_source.source_compute = clean_glsl
		var compiled: RDShaderSPIRV = _rd.shader_compile_spirv_from_source(shader_source)
		if (
				compiled != null
				and compiled.get_stage_compile_error(
						RenderingDevice.SHADER_STAGE_COMPUTE
				).is_empty()
		):
			return compiled

	if ResourceLoader.exists(path):
		var loaded: Resource = ResourceLoader.load(path)
		if loaded is RDShaderFile:
			var file_spirv: RDShaderSPIRV = (loaded as RDShaderFile).get_spirv()
			if (
					file_spirv != null
					and file_spirv.get_stage_compile_error(
							RenderingDevice.SHADER_STAGE_COMPUTE
					).is_empty()
			):
				return file_spirv

	return null


## Frees all per-instance stepping buffers and uniform set RIDs on [member _rd].
func _dispose_stepping_rids() -> void:
	_gpu_ssbo_valid = false;
	_free_uniform_set();
	if _body_buffer_rid.is_valid():
		_rd.free_rid(_body_buffer_rid);
		_body_buffer_rid = RID();
	if _well_buffer_rid.is_valid():
		_rd.free_rid(_well_buffer_rid);
		_well_buffer_rid = RID();
	if _epoch_buffer_rid.is_valid():
		_rd.free_rid(_epoch_buffer_rid);
		_epoch_buffer_rid = RID();
	if _output_buffer_rid.is_valid():
		_rd.free_rid(_output_buffer_rid);
		_output_buffer_rid = RID();


## Frees all per-instance forecast buffers and uniform set RIDs on [member _rd].
func _dispose_forecast_rids() -> void:
	_free_forecast_uniform_set()
	if _forecast_body_buffer_rid.is_valid():
		_rd.free_rid(_forecast_body_buffer_rid)
		_forecast_body_buffer_rid = RID()
	if _forecast_timeline_buffer_rid.is_valid():
		_rd.free_rid(_forecast_timeline_buffer_rid)
		_forecast_timeline_buffer_rid = RID()
	if _forecast_output_buffer_rid.is_valid():
		_rd.free_rid(_forecast_output_buffer_rid)
		_forecast_output_buffer_rid = RID()


## Ensures stepping SSBOs and uniform set have capacity for [param required_bodies] and
## [param required_wells].
func _ensure_buffer_capacities(required_bodies: int, required_wells: int) -> void:
	var target_bodies: int = maxi(1, required_bodies);
	var target_wells: int = maxi(1, required_wells);
	var buffers_recreated: bool = false;

	if (
			target_bodies > _body_capacity
			or not _body_buffer_rid.is_valid()
			or not _epoch_buffer_rid.is_valid()
			or not _output_buffer_rid.is_valid()
	):
		_free_uniform_set();
		if _body_buffer_rid.is_valid():
			_rd.free_rid(_body_buffer_rid);
		if _epoch_buffer_rid.is_valid():
			_rd.free_rid(_epoch_buffer_rid);
		if _output_buffer_rid.is_valid():
			_rd.free_rid(_output_buffer_rid);
		_body_capacity = _next_power_of_two(target_bodies);
		_body_buffer_rid = _rd.storage_buffer_create(_body_capacity * BYTES_PER_BODY);
		_epoch_buffer_rid = _rd.storage_buffer_create(_body_capacity * BYTES_PER_EPOCH);
		var out_bytes: int = (
				STEP_OUT_HEADER_BYTES
				+ _body_capacity * (STEP_OUT_PV_BYTES_PER_BODY + STEP_OUT_EVENT_BYTES)
		);
		_output_buffer_rid = _rd.storage_buffer_create(out_bytes);
		_gpu_ssbo_valid = false;
		buffers_recreated = true;

	if target_wells > _well_capacity or not _well_buffer_rid.is_valid():
		_free_uniform_set();
		if _well_buffer_rid.is_valid():
			_rd.free_rid(_well_buffer_rid);
		_well_capacity = _next_power_of_two(target_wells);
		_well_buffer_rid = _rd.storage_buffer_create(_well_capacity * BYTES_PER_WELL);
		buffers_recreated = true;

	if buffers_recreated or not _uniform_set_rid.is_valid():
		_rebuild_uniform_set();


## Rebuilds [member _uniform_set_rid] binding Bindings 0 (bodies), 1 (wells), 2 (epochs), and 3 (output).
func _rebuild_uniform_set() -> void:
	_free_uniform_set();

	var body_uniform := RDUniform.new();
	body_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	body_uniform.binding = 0;
	body_uniform.add_id(_body_buffer_rid);

	var well_uniform := RDUniform.new();
	well_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	well_uniform.binding = 1;
	well_uniform.add_id(_well_buffer_rid);

	var epoch_uniform := RDUniform.new();
	epoch_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	epoch_uniform.binding = 2;
	epoch_uniform.add_id(_epoch_buffer_rid);

	var out_uniform := RDUniform.new();
	out_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	out_uniform.binding = 3;
	out_uniform.add_id(_output_buffer_rid);

	_uniform_set_rid = _rd.uniform_set_create(
			[body_uniform, well_uniform, epoch_uniform, out_uniform],
			_shader_rid,
			0
	);


## Safely frees [member _uniform_set_rid] if still valid on [member _rd].
func _free_uniform_set() -> void:
	if _rd != null and _uniform_set_rid.is_valid():
		_rd.free_rid(_uniform_set_rid)
	_uniform_set_rid = RID()


## Uploads body and gravity-well SSBOs, dispatches [code]ballistic_gravity.glsl[/code],
## synchronizes, and writes integrated anchor-relative positions and velocities back to
## [param bodies].
func _step_bodies_gpu(
		bodies: Array[JrBallisticOrbiter],
		packed_wells: PackedFloat32Array,
		dt: float,
		substeps: int
) -> void:
	if _telemetry != null:
		_step_bodies_gpu_instrumented(bodies, packed_wells, dt, substeps);
		return;

	var body_count: int = bodies.size();
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL;
	_ensure_buffer_capacities(body_count, well_count);
	_sync_or_upload_bodies_ssbo(bodies, packed_wells, dt);

	if well_count > 0:
		for w: int in range(well_count):
			packed_wells[w * FLOATS_PER_WELL + 11] = _step_safe_radii[w];
		var well_bytes: PackedByteArray = packed_wells.to_byte_array();
		_rd.buffer_update(_well_buffer_rid, 0, well_bytes.size(), well_bytes);

	_rd.buffer_update(_output_buffer_rid, 0, 4, _zero_u32_bytes);
	var push_bytes: PackedByteArray = _build_push_constants(
			body_count,
			well_count,
			substeps,
			dt
	);
	var groups_x := int(ceili(float(body_count) / float(WORKGROUP_SIZE_X)));

	var compute_list: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(compute_list, _pipeline_rid);
	_rd.compute_list_bind_uniform_set(compute_list, _uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(compute_list, push_bytes, STEP_PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(compute_list, groups_x, 1, 1);
	_rd.compute_list_end();

	_rd.submit();
	_rd.sync();

	var pv_read_bytes: int = STEP_OUT_HEADER_BYTES + body_count * STEP_OUT_PV_BYTES_PER_BODY;
	var head_bytes: PackedByteArray = _rd.buffer_get_data(_output_buffer_rid, 0, pv_read_bytes);
	_unpack_bodies_buffer(bodies, head_bytes, packed_wells);


## Executes [method _step_bodies_gpu] while recording microsecond sub-phase timings into
## [member _telemetry].
func _step_bodies_gpu_instrumented(
		bodies: Array[JrBallisticOrbiter],
		packed_wells: PackedFloat32Array,
		dt: float,
		substeps: int
) -> void:
	var body_count: int = bodies.size();
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL;
	_ensure_buffer_capacities(body_count, well_count);

	var t0: int = Time.get_ticks_usec();
	_sync_or_upload_bodies_ssbo(bodies, packed_wells, dt);
	var t1: int = Time.get_ticks_usec();

	if well_count > 0:
		for w: int in range(well_count):
			packed_wells[w * FLOATS_PER_WELL + 11] = _step_safe_radii[w];
		var well_bytes: PackedByteArray = packed_wells.to_byte_array();
		_rd.buffer_update(_well_buffer_rid, 0, well_bytes.size(), well_bytes);

	_rd.buffer_update(_output_buffer_rid, 0, 4, _zero_u32_bytes);
	var push_bytes: PackedByteArray = _build_push_constants(
			body_count,
			well_count,
			substeps,
			dt
	);
	var groups_x: int = int(ceili(float(body_count) / float(WORKGROUP_SIZE_X)));

	var compute_list: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(compute_list, _pipeline_rid);
	_rd.compute_list_bind_uniform_set(compute_list, _uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(compute_list, push_bytes, STEP_PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(compute_list, groups_x, 1, 1);
	_rd.compute_list_end();
	var t2: int = Time.get_ticks_usec();

	_rd.submit();
	_rd.sync();
	var t3: int = Time.get_ticks_usec();

	var pv_read_bytes: int = STEP_OUT_HEADER_BYTES + body_count * STEP_OUT_PV_BYTES_PER_BODY;
	var head_bytes: PackedByteArray = _rd.buffer_get_data(_output_buffer_rid, 0, pv_read_bytes);
	var t4: int = Time.get_ticks_usec();

	_unpack_bodies_buffer(bodies, head_bytes, packed_wells);
	var t5: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_STEP_PACK, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_STEP_UPLOAD_DISPATCH, t2 - t1);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_STEP_SUBMIT_SYNC, t3 - t2);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_STEP_READBACK, t4 - t3);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_STEP_UNPACK, t5 - t4);


## Synchronizes [member _body_buffer_rid] and [member _epoch_buffer_rid] via zero-copy persistence
## when resident state is clean, sparse `O(K)` patches when few bodies changed, or bulk upload on
## topology/time-origin changes.
func _sync_or_upload_bodies_ssbo(
		bodies: Array[JrBallisticOrbiter],
		packed_wells: PackedFloat32Array,
		dt: float
) -> void:
	_precompute_step_safe_radii(packed_wells);
	var body_count: int = bodies.size();
	var locked_ecliptic: bool = body_count > 0 and bodies[0].is_locked_to_ecliptic();

	if (
			not _gpu_ssbo_valid
			or locked_ecliptic
			or body_count != _resident_body_count
			or absf(_step_current_time - _gpu_time_origin) > MAX_GPU_TIME_ORIGIN_SPAN
			or not _check_resident_wells_match()
	):
		_gpu_time_origin = _step_current_time;
		_upload_bulk_bodies_and_epochs(bodies, dt);
		_resident_body_count = body_count;
		_resident_well_elements = _step_well_elements.duplicate();
		_gpu_ssbo_valid = not locked_ecliptic;
		_any_gpu_body_dirty = false;
		_dirty_gpu_bodies.clear();
		return;

	if not _any_gpu_body_dirty:
		return;

	if _dirty_gpu_bodies.size() > 64:
		_gpu_time_origin = _step_current_time;
		_upload_bulk_bodies_and_epochs(bodies, dt);
		_any_gpu_body_dirty = false;
		_dirty_gpu_bodies.clear();
		return;

	if _upload_queued_dirty_bodies_fast(bodies, dt):
		_any_gpu_body_dirty = false;
		return;

	_gpu_time_origin = _step_current_time;
	_upload_bulk_bodies_and_epochs(bodies, dt);
	_any_gpu_body_dirty = false;
	_dirty_gpu_bodies.clear();


## Uploads only the queued [member _dirty_gpu_bodies] in `O(K)` via their cached sequence index.
func _upload_queued_dirty_bodies_fast(bodies: Array[JrBallisticOrbiter], dt: float) -> bool:
	var body_count: int = bodies.size();
	for q: int in range(_dirty_gpu_bodies.size()):
		var candidate: JrBallisticOrbiter = _dirty_gpu_bodies[q];
		if not is_instance_valid(candidate):
			continue;
		var idx: int = candidate._ballistic_seq_index;
		if idx < 0 or idx >= body_count or bodies[idx] != candidate:
			return false;

	if _single_body_vec4s.size() != VEC4S_PER_BODY:
		_single_body_vec4s.resize(VEC4S_PER_BODY);
	if _single_epoch_vec4s.size() != VEC4S_PER_EPOCH:
		_single_epoch_vec4s.resize(VEC4S_PER_EPOCH);
	var single_v4s: PackedVector4Array = _single_body_vec4s;
	var single_ep_v4s: PackedVector4Array = _single_epoch_vec4s;
	_single_body_vec4s = PackedVector4Array();
	_single_epoch_vec4s = PackedVector4Array();
	var curr_time: float = _step_current_time;

	for q: int in range(_dirty_gpu_bodies.size()):
		var body: JrBallisticOrbiter = _dirty_gpu_bodies[q];
		if not is_instance_valid(body) or not body._gpu_step_dirty:
			continue;
		var idx: int = body._ballistic_seq_index;
		_write_body_vec4s(body, single_v4s, 0, dt);
		body._gpu_step_dirty = false;
		var slice_bytes: PackedByteArray = single_v4s.to_byte_array();
		_rd.buffer_update(_body_buffer_rid, idx * BYTES_PER_BODY, BYTES_PER_BODY, slice_bytes);
		if body._is_on_rails:
			_write_epoch_vec4s(body, single_ep_v4s, 0, curr_time);
			var ep_bytes: PackedByteArray = single_ep_v4s.to_byte_array();
			_rd.buffer_update(_epoch_buffer_rid, idx * BYTES_PER_EPOCH, BYTES_PER_EPOCH, ep_bytes);

	_single_body_vec4s = single_v4s;
	_single_epoch_vec4s = single_ep_v4s;
	_dirty_gpu_bodies.clear();
	return true;


## Returns `true` if [member _step_well_elements] matches [member _resident_well_elements] identically.
func _check_resident_wells_match() -> bool:
	var w_count: int = _step_well_elements.size();
	if w_count != _resident_well_elements.size():
		return false;
	for w: int in range(w_count):
		if _step_well_elements[w] != _resident_well_elements[w]:
			return false;
	return true;


## Writes the 4 [Vector4] fields of [param body] into [param dst] at [param base].
func _write_body_vec4s(
		body: JrBallisticOrbiter,
		dst: PackedVector4Array,
		base: int,
		eff_dt: float
) -> void:
	var anchor_elem: JrOrreryElement = body._anchor_element;
	var anchor_idx: int = (
			anchor_elem._runtime_well_index
			if (anchor_elem != null and anchor_elem._runtime_well_index >= 0)
			else 0
	);
	var frame_pin_w: float = body._solver_frame_pin_w;
	var thrust_v4 := Vector4(0.0, 0.0, 0.0, frame_pin_w);
	if body._has_external_thrust:
		var ext: Vector3 = body.get_scaled_external_acceleration();
		thrust_v4 = Vector4(ext.x, ext.y, ext.z, frame_pin_w);

	var pos: Vector3 = body._anchor_relative_position;
	var vel: Vector3 = body._anchor_relative_velocity;
	dst[base] = Vector4(pos.x, pos.y, pos.z, float(anchor_idx));
	dst[base + 1] = Vector4(vel.x, vel.y, vel.z, eff_dt);
	dst[base + 2] = thrust_v4;
	if body._is_on_rails:
		dst[base + 3] = Vector4(1.0, body._solver_flags_y, body._epoch_safe_max_radius, -1.0);
	else:
		dst[base + 3] = Vector4(0.0, body._solver_flags_y, 0.0, -1.0);


## Writes the 2 [Vector4] Tier-1 epoch fields of [param body] into [param dst] at [param base].
func _write_epoch_vec4s(
		body: JrBallisticOrbiter,
		dst: PackedVector4Array,
		base: int,
		curr_time: float
) -> void:
	var wrapped_dt: float = body.get_wrapped_epoch_elapsed_time(curr_time);
	var e_rel: float = (curr_time - _gpu_time_origin) - wrapped_dt;
	var ep_pair: Vector2 = _split_float64_to_f32_pair(e_rel);
	var r_pos: Vector3 = body._epoch_anchor_pos;
	var r_vel: Vector3 = body._epoch_anchor_vel;
	dst[base] = Vector4(r_pos.x, r_pos.y, r_pos.z, ep_pair.x);
	dst[base + 1] = Vector4(r_vel.x, r_vel.y, r_vel.z, ep_pair.y);


## Packs and uploads [member _body_buffer_rid] and [member _epoch_buffer_rid] for all [param bodies].
func _upload_bulk_bodies_and_epochs(bodies: Array[JrBallisticOrbiter], dt: float) -> void:
	var count: int = bodies.size();
	var req_body_v4s: int = count * VEC4S_PER_BODY;
	var req_ep_v4s: int = count * VEC4S_PER_EPOCH;
	var vec4s: PackedVector4Array = _packed_body_vec4s;
	var ep_vec4s: PackedVector4Array = _packed_epoch_vec4s;
	_packed_body_vec4s = PackedVector4Array();
	_packed_epoch_vec4s = PackedVector4Array();
	if vec4s.size() != req_body_v4s:
		vec4s.resize(req_body_v4s);
	if ep_vec4s.size() != req_ep_v4s:
		ep_vec4s.resize(req_ep_v4s);

	var curr_time: float = _step_current_time;
	var any_rails: bool = false;

	for i: int in range(count):
		var body: JrBallisticOrbiter = bodies[i];
		body._gpu_step_dirty = false;
		var base: int = i * VEC4S_PER_BODY;
		_write_body_vec4s(body, vec4s, base, dt);
		if body._is_on_rails:
			any_rails = true;
			_write_epoch_vec4s(body, ep_vec4s, i * VEC4S_PER_EPOCH, curr_time);

	var body_bytes: PackedByteArray = vec4s.to_byte_array();
	_rd.buffer_update(_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if any_rails and count > 0:
		var ep_bytes: PackedByteArray = ep_vec4s.to_byte_array();
		_rd.buffer_update(_epoch_buffer_rid, 0, ep_bytes.size(), ep_bytes);

	_packed_body_vec4s = vec4s;
	_packed_epoch_vec4s = ep_vec4s;


## Unpacks integrated anchor-relative `(pos, vel)` vectors from [param head_bytes] and applies
## sparse GPU rail/SOI transition events when `event_count > 0`.
func _unpack_bodies_buffer(
		bodies: Array[JrBallisticOrbiter],
		head_bytes: PackedByteArray,
		packed_wells: PackedFloat32Array
) -> void:
	var count: int = bodies.size();
	var locked_ecliptic: bool = count > 0 and bodies[0].is_locked_to_ecliptic();
	_gpu_handoffs_evaluated = not locked_ecliptic;
	_gpu_handoff_body_indices.clear();
	_gpu_handoff_target_wells.clear();

	var pv_vecs: PackedVector3Array = head_bytes.to_vector3_array();
	if locked_ecliptic:
		var pv_ec: int = 2;
		for i: int in range(count):
			bodies[i].apply_integrated_anchor_state(pv_vecs[pv_ec], pv_vecs[pv_ec + 1]);
			pv_ec += 2;
	else:
		var pv_idx: int = 2;
		for i: int in range(count):
			var body: JrBallisticOrbiter = bodies[i];
			body._anchor_relative_position = pv_vecs[pv_idx];
			body._anchor_relative_velocity = pv_vecs[pv_idx + 1];
			pv_idx += 2;

	var event_count: int = mini(head_bytes.decode_u32(0), count);
	if event_count > 0:
		_apply_gpu_step_events(bodies, event_count, packed_wells, locked_ecliptic);


## Reads and applies the `event_count` sparse rail-transition / Hill-sphere handoff records from
## [member _output_buffer_rid].
func _apply_gpu_step_events(
		bodies: Array[JrBallisticOrbiter],
		event_count: int,
		packed_wells: PackedFloat32Array,
		locked_ecliptic: bool
) -> void:
	var count: int = bodies.size();
	var ev_offset: int = STEP_OUT_HEADER_BYTES + count * STEP_OUT_PV_BYTES_PER_BODY;
	var ev_bytes: PackedByteArray = _rd.buffer_get_data(
			_output_buffer_rid,
			ev_offset,
			event_count * STEP_OUT_EVENT_BYTES
	);
	var ev_words: PackedInt32Array = ev_bytes.to_int32_array();
	var safe_radii: PackedFloat32Array = _step_safe_radii;
	var safe_count: int = safe_radii.size();
	var curr_time: float = _step_current_time;

	var handoff_bodies: PackedInt32Array = _gpu_handoff_body_indices;
	var handoff_wells: PackedInt32Array = _gpu_handoff_target_wells;
	_gpu_handoff_body_indices = PackedInt32Array();
	_gpu_handoff_target_wells = PackedInt32Array();

	for e: int in range(event_count):
		var b_idx: int = ev_words[e * 2];
		if b_idx < 0 or b_idx >= count:
			continue;
		var packed_ev: int = ev_words[e * 2 + 1];
		var next_on_rails: bool = (packed_ev & 1) != 0;
		var well_plus_one: int = packed_ev >> 1;
		var body: JrBallisticOrbiter = bodies[b_idx];

		if not locked_ecliptic and well_plus_one > 0:
			handoff_bodies.append(b_idx);
			handoff_wells.append(well_plus_one - 1);

		if next_on_rails != body._is_on_rails:
			if next_on_rails:
				var anchor_elem: JrOrreryElement = body._anchor_element;
				var anchor_idx: int = (
						anchor_elem._runtime_well_index
						if (anchor_elem != null and anchor_elem._runtime_well_index >= 0)
						else 0
				);
				var mu_anchor: float = _get_well_mu(anchor_idx, packed_wells);
				var safe_r: float = (
						safe_radii[anchor_idx]
						if (anchor_idx >= 0 and anchor_idx < safe_count)
						else _compute_safe_max_radius_for_anchor(anchor_idx, packed_wells)
				);
				body.sync_on_rails_epoch_from_gpu(curr_time, mu_anchor, safe_r);
			else:
				body._is_on_rails = false;

	_gpu_handoff_body_indices = handoff_bodies;
	_gpu_handoff_target_wells = handoff_wells;


## Splits a 64-bit [param val] into an exact IEEE-754 `binary32` `(hi, lo)` pair.
func _split_float64_to_f32_pair(val: float) -> Vector2:
	if _f32_split_scratch.size() != 2:
		_f32_split_scratch.resize(2);
	_f32_split_scratch[0] = val;
	var hi: float = _f32_split_scratch[0];
	_f32_split_scratch[1] = val - hi;
	return Vector2(hi, _f32_split_scratch[1]);


## Precomputes [member _step_safe_radii] once per stepping pass across all gravity wells.
func _precompute_step_safe_radii(packed_wells: PackedFloat32Array) -> void:
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL;
	if _step_safe_radii.size() != well_count:
		_step_safe_radii.resize(well_count);
	for w: int in range(well_count):
		_step_safe_radii[w] = _compute_safe_max_radius_for_anchor(w, packed_wells);


## Encodes the 32-byte push constant payload for [code]ballistic_gravity.glsl[/code].
func _build_push_constants(
		body_count: int,
		well_count: int,
		substeps: int,
		step_dt: float
) -> PackedByteArray:
	var time_pair: Vector2 = _split_float64_to_f32_pair(_step_current_time - _gpu_time_origin);
	var events_word_offset: int = 6 + body_count * 6;
	var bytes: PackedByteArray = [];
	bytes.resize(STEP_PUSH_CONSTANT_BYTES);
	bytes.encode_float(0, _step_cull_threshold_sq);
	bytes.encode_u32(4, body_count);
	bytes.encode_u32(8, well_count);
	bytes.encode_u32(12, substeps);
	bytes.encode_float(16, step_dt);
	bytes.encode_float(20, time_pair.x);
	bytes.encode_float(24, time_pair.y);
	bytes.encode_u32(28, events_word_offset);
	return bytes;


## Returns the gravitational parameter mu of well [param well_idx] in [param packed_wells].
func _get_well_mu(well_idx: int, packed_wells: PackedFloat32Array) -> float:
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL
	if well_idx < 0 or well_idx >= well_count:
		return 0.0
	return packed_wells[well_idx * FLOATS_PER_WELL + 3]


## Computes the analytical maximum radius around [param anchor_idx] below which all
## third-body tides remain below [member _step_cull_threshold_sq].
func _compute_safe_max_radius_for_anchor(
		anchor_idx: int,
		packed_wells: PackedFloat32Array
) -> float:
	var mu_anchor: float = _get_well_mu(anchor_idx, packed_wells)
	if mu_anchor <= 0.0:
		return 0.0
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL
	var cull_eps: float = sqrt(_step_cull_threshold_sq)
	if cull_eps <= 0.0:
		return 0.0

	var min_safe_r: float = INF
	for k: int in range(well_count):
		if k == anchor_idx:
			continue
		var mu_k: float = packed_wells[k * FLOATS_PER_WELL + 3]
		if mu_k <= 0.0:
			continue
		var d_vec: Vector3 = _compute_well_difference_cpu(anchor_idx, k, packed_wells, 1.0)
		var d_len: float = d_vec.length()
		if d_len <= 0.0:
			continue
		var wake_r: float = d_len * pow((cull_eps * mu_anchor) / (2.0 * mu_k), 1.0 / 3.0)
		min_safe_r = minf(min_safe_r, wake_r)

	return 0.0 if is_inf(min_safe_r) else min_safe_r


## Executes identical Hybrid Universal-Variable Kick-Drift-Kick stepping on the CPU.
func _step_bodies_cpu(
		bodies: Array[JrBallisticOrbiter],
		packed_wells: PackedFloat32Array,
		dt: float,
		substeps: int
) -> void:
	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	_precompute_step_safe_radii(packed_wells);
	for body: JrBallisticOrbiter in bodies:
		_step_single_body_cpu(body, packed_wells, dt, substeps);
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_SOLVER_STEP_CPU,
				Time.get_ticks_usec() - t0
		);


## Integrates a single [param body] over [param dt] on the CPU matching
## [code]ballistic_gravity.glsl[/code].
func _step_single_body_cpu(
		body: JrBallisticOrbiter,
		packed_wells: PackedFloat32Array,
		dt: float,
		max_substeps: int
) -> void:
	var anchor_idx: int = _step_well_indices.get(body.get_anchor_element(), 0);
	var mu_anchor: float = _get_well_mu(anchor_idx, packed_wells);
	var ext_accel: Vector3 = body.get_scaled_external_acceleration();
	var has_thrust: bool = ext_accel != Vector3.ZERO;

	if body.is_on_rails() and not has_thrust and mu_anchor > 0.0:
		_step_on_rails_body_cpu(body, anchor_idx, mu_anchor, packed_wells);
		return;

	var is_orbital_thrust: bool = body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL;
	var enable_tidal: bool = body.enable_tidal_forces;
	var r_rel: Vector3 = body.get_anchor_relative_position();
	var v_rel: Vector3 = body.get_anchor_relative_velocity();
	var r_len: float = maxf(1e-6, r_rel.length());
	var approx_dtheta: float = (v_rel.length() * absf(dt)) / r_len;
	var step_count: int = clampi(int(ceili(approx_dtheta / 0.12)), 1, maxi(1, max_substeps));
	if has_thrust or mu_anchor <= 0.0:
		step_count = maxi(step_count, mini(maxi(1, max_substeps), 4));

	var sub_dt: float = dt / float(step_count);
	var inv_steps: float = 1.0 / float(step_count);
	var any_perturbation_ever: bool = has_thrust or (mu_anchor <= 0.0);

	_evaluate_perturbing_into(r_rel, anchor_idx, packed_wells, 0.0 if enable_tidal else -1.0);
	if _pert_active_out:
		any_perturbation_ever = true;
	var total_pert: Vector3 = (
			_pert_accel_out + _evaluate_thrust_cpu(r_rel, v_rel, ext_accel, is_orbital_thrust)
	);

	for s: int in range(step_count):
		v_rel += 0.5 * sub_dt * total_pert;
		_kepler_drift_into(r_rel, v_rel, mu_anchor, sub_dt);
		r_rel = _drift_r;
		v_rel = _drift_v;

		var alpha_end: float = (float(s + 1) * inv_steps) if enable_tidal else -1.0;
		_evaluate_perturbing_into(r_rel, anchor_idx, packed_wells, alpha_end);
		if _pert_active_out:
			any_perturbation_ever = true;
		total_pert = (
				_pert_accel_out + _evaluate_thrust_cpu(r_rel, v_rel, ext_accel, is_orbital_thrust)
		);
		v_rel += 0.5 * sub_dt * total_pert;

	body.apply_integrated_anchor_state(r_rel, v_rel);
	if not any_perturbation_ever and mu_anchor > 0.0:
		var safe_r: float = (
				_step_safe_radii[anchor_idx]
				if (anchor_idx >= 0 and anchor_idx < _step_safe_radii.size())
				else _compute_safe_max_radius_for_anchor(anchor_idx, packed_wells)
		);
		body.enter_on_rails_epoch(_step_current_time, mu_anchor, safe_r);
	else:
		body.exit_on_rails_epoch();


## Evaluates a Tier-1 epoch-frozen coasting [param body] on the CPU and wakes it if tides exceed
## threshold.
func _step_on_rails_body_cpu(
		body: JrBallisticOrbiter,
		anchor_idx: int,
		mu_anchor: float,
		packed_wells: PackedFloat32Array
) -> void:
	var elapsed_dt: float = body.get_wrapped_epoch_elapsed_time(_step_current_time);
	_kepler_drift_into(
			body.get_epoch_anchor_position(),
			body.get_epoch_anchor_velocity(),
			mu_anchor,
			elapsed_dt
	);
	var r_new: Vector3 = _drift_r;
	var v_new: Vector3 = _drift_v;
	var wake_up: bool = false;
	if body.enable_tidal_forces:
		var safe_r: float = body.get_epoch_safe_max_radius();
		if safe_r > 0.0 and r_new.length_squared() > safe_r * safe_r:
			wake_up = true;
		else:
			_evaluate_perturbing_into(r_new, anchor_idx, packed_wells, 1.0);
			wake_up = _pert_active_out;
	body.apply_integrated_anchor_state(r_new, v_new);
	if wake_up:
		body.exit_on_rails_epoch();


## Propagates [param r0] and [param v0] into [member _drift_r] and [member _drift_v] without heap
## allocations.
func _kepler_drift_into(r0: Vector3, v0: Vector3, mu: float, dt: float) -> void:
	var r_scale: float = r0.length();
	if r_scale <= 1e-9 or mu <= 0.0 or absf(dt) <= 1e-12:
		_drift_r = r0 + v0 * dt;
		_drift_v = v0;
		return;

	var v_scale: float = sqrt(mu / r_scale);
	var r_unit: Vector3 = r0 / r_scale;
	var v_dim: Vector3 = v0 / v_scale;
	var dt_dim: float = dt * (v_scale / r_scale);
	var sigma0: float = r_unit.dot(v_dim);
	var alpha: float = 2.0 - v_dim.length_squared();

	if alpha > 1e-6:
		var period_dim: float = TAU / (alpha * sqrt(alpha));
		if absf(dt_dim) > period_dim:
			dt_dim = fmod(dt_dim, period_dim);
		if dt_dim > 0.5 * period_dim:
			dt_dim -= period_dim;
		elif dt_dim < -0.5 * period_dim:
			dt_dim += period_dim;

	var chi: float = _solve_universal_chi(sigma0, alpha, dt_dim);
	var chi2_f: float = chi * chi;
	var z_f: float = alpha * chi2_f;
	var cz_f: float = _stumpff_c(z_f);
	var sz_f: float = _stumpff_s(z_f);
	var chi3_f: float = chi2_f * chi;

	var r_dim: float = maxf(
			1e-9,
			1.0 + sigma0 * chi * (1.0 - z_f * sz_f) + (1.0 - alpha) * chi2_f * cz_f
	);
	var f_lag: float = 1.0 - chi2_f * cz_f;
	var g_lag: float = dt_dim - chi3_f * sz_f;
	var df_lag: float = (chi * (z_f * sz_f - 1.0)) / r_dim;
	var dg_lag: float = 1.0 - (chi2_f * cz_f) / r_dim;

	_drift_r = (f_lag * r_unit + g_lag * v_dim) * r_scale;
	_drift_v = (df_lag * r_unit + dg_lag * v_dim) * v_scale;


## Solves the dimensionless universal anomaly [code]chi[/code] via Newton-Raphson iteration.
static func _solve_universal_chi(sigma0: float, alpha: float, dt_dim: float) -> float:
	var chi: float = clampf((dt_dim * alpha) if alpha > 1e-6 else dt_dim, -1e5, 1e5);
	for _iter: int in range(20):
		var chi2: float = chi * chi;
		var z: float = alpha * chi2;
		var cz: float = _stumpff_c(z);
		var sz: float = _stumpff_s(z);
		var chi3: float = chi2 * chi;

		var f_val: float = sigma0 * chi2 * cz + (1.0 - alpha) * chi3 * sz + chi - dt_dim;
		var df: float = sigma0 * chi * (1.0 - z * sz) + (1.0 - alpha) * chi2 * cz + 1.0;
		var safe_df: float = maxf(1e-9, df) if df >= 0.0 else minf(-1e-9, df);
		var delta: float = f_val / safe_df;
		chi -= delta;
		if absf(delta) < 1e-10:
			break;
	return chi;


## Evaluates Battin's f(q) third-body tidal acceleration into [member _pert_accel_out] and
## [member _pert_active_out] without allocating a [Dictionary].
func _evaluate_perturbing_into(
		r_rel: Vector3,
		anchor_idx: int,
		packed_wells: PackedFloat32Array,
		alpha_or_neg: float
) -> void:
	_pert_accel_out = Vector3.ZERO;
	_pert_active_out = false;
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL;
	if well_count == 0:
		return;

	var enable_tidal: bool = alpha_or_neg >= 0.0;
	var mu_anchor: float = _get_well_mu(anchor_idx, packed_wells);
	if mu_anchor > 0.0 and not enable_tidal:
		return;

	var alpha: float = maxf(0.0, alpha_or_neg);
	var r_rel_sq: float = r_rel.length_squared();
	var total_accel := Vector3.ZERO;
	var any_active: bool = false;

	for k: int in range(well_count):
		if k == anchor_idx:
			continue;
		var mu_k: float = packed_wells[k * FLOATS_PER_WELL + 3];
		if mu_k <= 0.0:
			continue;

		var d_vec: Vector3 = _compute_well_difference_cpu(anchor_idx, k, packed_wells, alpha);
		var d_sq: float = d_vec.length_squared();
		if d_sq <= 1e-12:
			continue;

		if mu_anchor <= 0.0:
			if not enable_tidal and not _is_well_descendant_cpu(k, anchor_idx, packed_wells):
				continue;
			any_active = true;
			var rho_direct: Vector3 = r_rel + d_vec;
			var rho_direct_sq: float = rho_direct.length_squared();
			if rho_direct_sq > 1e-12:
				total_accel -= rho_direct * ((mu_k / rho_direct_sq) / sqrt(rho_direct_sq));
			continue;

		var r_over_d_sq: float = r_rel_sq / d_sq;
		var r_over_d_6: float = r_over_d_sq * r_over_d_sq * r_over_d_sq;
		var mu_ratio: float = mu_k / mu_anchor;
		if 4.0 * (mu_ratio * mu_ratio) * r_over_d_6 < _step_cull_threshold_sq:
			continue;

		any_active = true;
		var rho: Vector3 = r_rel + d_vec;
		var rho_sq: float = rho.length_squared();
		if rho_sq <= 1e-12:
			continue;

		var rho_factor: float = (mu_k / rho_sq) / sqrt(rho_sq);
		var q: float = r_rel.dot(r_rel + 2.0 * d_vec) / d_sq;
		if absf(q) <= 0.5 and (1.0 + q) > 0.0:
			var one_plus_q: float = 1.0 + q;
			var fq: float = q * (3.0 + 3.0 * q + q * q) / (1.0 + one_plus_q * sqrt(one_plus_q));
			total_accel -= rho_factor * (r_rel - fq * d_vec);
		else:
			var d_factor: float = (mu_k / d_sq) / sqrt(d_sq);
			total_accel -= (rho * rho_factor - d_vec * d_factor);

	_pert_accel_out = total_accel;
	_pert_active_out = any_active;


## Computes [code]r_well_a - r_well_b[/code] at interpolation fraction [param alpha]
## via a two-pointer Lowest Common Ancestor (LCA) walk on [param packed_wells].
func _compute_well_difference_cpu(
		well_a: int,
		well_b: int,
		packed_wells: PackedFloat32Array,
		alpha: float
) -> Vector3:
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL
	var curr_a: int = well_a
	var curr_b: int = well_b
	var diff := Vector3.ZERO
	var guard: int = 0

	while curr_a != curr_b and guard < 32:
		guard += 1
		if curr_a > curr_b:
			if curr_a < 0 or curr_a >= well_count:
				break
			var base_a: int = curr_a * FLOATS_PER_WELL
			var start_a := Vector3(
					packed_wells[base_a],
					packed_wells[base_a + 1],
					packed_wells[base_a + 2]
			)
			var end_a := Vector3(
					packed_wells[base_a + 4],
					packed_wells[base_a + 5],
					packed_wells[base_a + 6]
			)
			diff += start_a.lerp(end_a, alpha)
			curr_a = int(roundf(packed_wells[base_a + 7]))
		else:
			if curr_b < 0 or curr_b >= well_count:
				break
			var base_b: int = curr_b * FLOATS_PER_WELL
			var start_b := Vector3(
					packed_wells[base_b],
					packed_wells[base_b + 1],
					packed_wells[base_b + 2]
			)
			var end_b := Vector3(
					packed_wells[base_b + 4],
					packed_wells[base_b + 5],
					packed_wells[base_b + 6]
			)
			diff -= start_b.lerp(end_b, alpha)
			curr_b = int(roundf(packed_wells[base_b + 7]))

	return diff


## Returns [code]true[/code] if [param well_candidate] is a descendant of [param ancestor_well].
func _is_well_descendant_cpu(
		well_candidate: int,
		ancestor_well: int,
		packed_wells: PackedFloat32Array
) -> bool:
	if ancestor_well < 0 or well_candidate <= ancestor_well:
		return false
	@warning_ignore("integer_division")
	var well_count: int = packed_wells.size() / FLOATS_PER_WELL
	var curr: int = well_candidate
	var guard: int = 0
	while curr > ancestor_well and guard < 32:
		guard += 1
		if curr >= well_count:
			return false
		curr = int(roundf(packed_wells[curr * FLOATS_PER_WELL + 7]))
		if curr == ancestor_well:
			return true
	return false


## Ensures forecast SSBOs and uniform set have sufficient capacity.
func _ensure_forecast_capacities(
		req_bodies: int,
		req_timeline_entries: int,
		req_output_points: int
) -> void:
	var target_bodies: int = maxi(1, req_bodies)
	var target_timeline: int = maxi(2, req_timeline_entries)
	var target_output: int = maxi(1, req_output_points)
	var recreated: bool = false

	if target_bodies > _forecast_body_capacity or not _forecast_body_buffer_rid.is_valid():
		_free_forecast_uniform_set()
		if _forecast_body_buffer_rid.is_valid():
			_rd.free_rid(_forecast_body_buffer_rid)
		_forecast_body_capacity = _next_power_of_two(target_bodies)
		_forecast_body_buffer_rid = _rd.storage_buffer_create(
				_forecast_body_capacity * BYTES_PER_BODY
		)
		recreated = true

	if (
			target_timeline > _forecast_timeline_capacity
			or not _forecast_timeline_buffer_rid.is_valid()
	):
		_free_forecast_uniform_set()
		if _forecast_timeline_buffer_rid.is_valid():
			_rd.free_rid(_forecast_timeline_buffer_rid)
		_forecast_timeline_capacity = _next_power_of_two(target_timeline)
		_forecast_timeline_buffer_rid = _rd.storage_buffer_create(
				_forecast_timeline_capacity * BYTES_PER_VEC4
		)
		recreated = true

	if target_output > _forecast_output_capacity or not _forecast_output_buffer_rid.is_valid():
		_free_forecast_uniform_set()
		if _forecast_output_buffer_rid.is_valid():
			_rd.free_rid(_forecast_output_buffer_rid)
		_forecast_output_capacity = _next_power_of_two(target_output)
		_forecast_output_buffer_rid = _rd.storage_buffer_create(
				_forecast_output_capacity * BYTES_PER_VEC4
		)
		recreated = true

	if recreated or not _forecast_uniform_set_rid.is_valid():
		_rebuild_forecast_uniform_set()


## Rebuilds [member _forecast_uniform_set_rid] binding forecast Bindings 0, 1, and 2.
func _rebuild_forecast_uniform_set() -> void:
	_free_forecast_uniform_set()

	var u_bodies := RDUniform.new()
	u_bodies.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_bodies.binding = 0
	u_bodies.add_id(_forecast_body_buffer_rid)

	var u_timeline := RDUniform.new()
	u_timeline.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_timeline.binding = 1
	u_timeline.add_id(_forecast_timeline_buffer_rid)

	var u_output := RDUniform.new()
	u_output.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_output.binding = 2
	u_output.add_id(_forecast_output_buffer_rid)

	_forecast_uniform_set_rid = _rd.uniform_set_create(
			[u_bodies, u_timeline, u_output],
			_forecast_shader_rid,
			0
	)


## Safely frees [member _forecast_uniform_set_rid] if still valid on [member _rd].
func _free_forecast_uniform_set() -> void:
	if _rd != null and _forecast_uniform_set_rid.is_valid():
		_rd.free_rid(_forecast_uniform_set_rid)
	_forecast_uniform_set_rid = RID()


## Dispatches [code]ballistic_forecast.glsl[/code] on [member _rd] to integrate future
## anchor-relative trajectories for all [param bodies] in parallel.
func _compute_forecasts_gpu(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		timeline_floats: PackedFloat32Array,
		inv_timeline_dt: float
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	if _telemetry != null:
		return _compute_forecasts_gpu_instrumented(
				bodies,
				well_list,
				timeline_floats,
				inv_timeline_dt
		);

	var body_count: int = bodies.size()
	var well_count: int = well_list.size()
	var offsets: PackedInt32Array = _compute_forecast_output_offsets(bodies)
	var total_points: int = offsets[body_count]
	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_floats.size() / FLOATS_PER_VEC4
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	)

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_points)

	var body_bytes: PackedByteArray = (
			_pack_forecast_bodies_buffer(bodies, well_list, offsets).to_byte_array()
	)
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes)

	if not timeline_floats.is_empty():
		var tl_bytes: PackedByteArray = timeline_floats.to_byte_array()
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, tl_bytes.size(), tl_bytes)

	var push_bytes: PackedByteArray = _build_forecast_push_constants(
			inv_timeline_dt,
			body_count,
			well_count,
			timeline_segments
	)
	var groups_x := int(ceili(float(body_count) / float(WORKGROUP_SIZE_X)))

	var compute_list: int = _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(compute_list, _forecast_pipeline_rid)
	_rd.compute_list_bind_uniform_set(compute_list, _forecast_uniform_set_rid, 0)
	_rd.compute_list_set_push_constant(compute_list, push_bytes, PUSH_CONSTANT_BYTES)
	_rd.compute_list_dispatch(compute_list, groups_x, 1, 1)
	_rd.compute_list_end()

	_rd.submit()
	_rd.sync()

	var out_bytes: PackedByteArray = _rd.buffer_get_data(
			_forecast_output_buffer_rid,
			0,
			total_points * BYTES_PER_VEC4
	)
	return _unpack_forecast_output(bodies, offsets, out_bytes.to_float32_array())


## Executes [method _compute_forecasts_gpu] while recording microsecond sub-phase timings into
## [member _telemetry].
func _compute_forecasts_gpu_instrumented(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		timeline_floats: PackedFloat32Array,
		inv_timeline_dt: float
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var body_count: int = bodies.size();
	var well_count: int = well_list.size();

	var t0: int = Time.get_ticks_usec();
	var offsets: PackedInt32Array = _compute_forecast_output_offsets(bodies);
	var total_points: int = offsets[body_count];
	var body_bytes: PackedByteArray = (
			_pack_forecast_bodies_buffer(bodies, well_list, offsets).to_byte_array()
	);
	var t1: int = Time.get_ticks_usec();

	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_floats.size() / FLOATS_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_points);
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if not timeline_floats.is_empty():
		var tl_bytes: PackedByteArray = timeline_floats.to_byte_array();
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, tl_bytes.size(), tl_bytes);

	var push_bytes: PackedByteArray = _build_forecast_push_constants(
			inv_timeline_dt,
			body_count,
			well_count,
			timeline_segments
	);
	var groups_x: int = int(ceili(float(body_count) / float(WORKGROUP_SIZE_X)));

	var compute_list: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(compute_list, _forecast_pipeline_rid);
	_rd.compute_list_bind_uniform_set(compute_list, _forecast_uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(compute_list, push_bytes, PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(compute_list, groups_x, 1, 1);
	_rd.compute_list_end();
	var t2: int = Time.get_ticks_usec();

	_rd.submit();
	_rd.sync();
	var t3: int = Time.get_ticks_usec();

	var out_bytes: PackedByteArray = _rd.buffer_get_data(
			_forecast_output_buffer_rid,
			0,
			total_points * BYTES_PER_VEC4
	);
	var unpacked: Dictionary[JrBallisticOrbiter, PackedVector4Array] = _unpack_forecast_output(
			bodies,
			offsets,
			out_bytes.to_float32_array()
	);
	var t4: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_PACK, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH, t2 - t1);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_SUBMIT_SYNC, t3 - t2);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_READBACK_UNPACK, t4 - t3);
	return unpacked;


## Computes prefix-sum output start offsets (and total point count at index
## [code]bodies.size()[/code]).
func _compute_forecast_output_offsets(bodies: Array[JrBallisticOrbiter]) -> PackedInt32Array:
	var count: int = bodies.size()
	var offsets: PackedInt32Array = []
	offsets.resize(count + 1)
	var running_total: int = 0
	for i: int in range(count):
		offsets[i] = running_total
		running_total += maxi(1, bodies[i].preview_steps) + 1
	offsets[count] = running_total
	return offsets


## Packs anchor-relative initial states, durations, step counts, thrust vectors,
## and anchor metadata for [code]ForecastBodyInputBuffer[/code].
func _pack_forecast_bodies_buffer(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		offsets: PackedInt32Array
) -> PackedFloat32Array:
	var count: int = bodies.size()
	var floats: PackedFloat32Array = []
	floats.resize(count * FLOATS_PER_BODY)

	var well_map: Dictionary[JrOrreryElement, int] = {}
	for w: int in range(well_list.size()):
		if well_list[w] != null:
			well_map[well_list[w]] = w

	for i: int in range(count):
		var body: JrBallisticOrbiter = bodies[i]
		var pos: Vector3 = body.get_anchor_relative_position()
		var vel: Vector3 = body.get_anchor_relative_velocity()
		var ext: Vector3 = body.get_scaled_external_acceleration()
		var anchor_idx: int = well_map.get(body.get_anchor_element(), 0)
		var frame_val: float = (
				2.0
				if body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL
				else 0.0
		)
		var base: int = i * FLOATS_PER_BODY

		floats[base] = pos.x
		floats[base + 1] = pos.y
		floats[base + 2] = pos.z
		floats[base + 3] = maxf(0.0, body.preview_duration)

		floats[base + 4] = vel.x
		floats[base + 5] = vel.y
		floats[base + 6] = vel.z
		floats[base + 7] = float(maxi(1, body.preview_steps))

		floats[base + 8] = ext.x
		floats[base + 9] = ext.y
		floats[base + 10] = ext.z
		floats[base + 11] = float(offsets[i])

		floats[base + 12] = float(anchor_idx)
		floats[base + 13] = frame_val
		floats[base + 14] = 1.0 if body.enable_tidal_forces else 0.0
		floats[base + 15] = 1.0 if body.pinned_anchor != null else 0.0

	return floats


## Unpacks flattened [code]vec4[/code] output points into per-body [PackedVector4Array] slices.
func _unpack_forecast_output(
		bodies: Array[JrBallisticOrbiter],
		offsets: PackedInt32Array,
		out_floats: PackedFloat32Array
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var results: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {}
	for i: int in range(bodies.size()):
		var start_pt: int = offsets[i]
		var pt_count: int = offsets[i + 1] - start_pt
		var pts: PackedVector4Array = []
		pts.resize(pt_count)
		for k: int in range(pt_count):
			var f_base: int = (start_pt + k) * FLOATS_PER_VEC4
			pts[k] = Vector4(
					out_floats[f_base],
					out_floats[f_base + 1],
					out_floats[f_base + 2],
					out_floats[f_base + 3]
			)
		results[bodies[i]] = pts
	return results


## Encodes the 16-byte push constant payload for [code]ballistic_forecast.glsl[/code].
func _build_forecast_push_constants(
		inv_timeline_dt: float,
		body_count: int,
		well_count: int,
		timeline_segments: int
) -> PackedByteArray:
	var bytes: PackedByteArray = []
	bytes.resize(PUSH_CONSTANT_BYTES)
	bytes.encode_float(0, inv_timeline_dt)
	bytes.encode_u32(4, body_count)
	bytes.encode_u32(8, well_count)
	bytes.encode_u32(12, timeline_segments)
	return bytes


## Executes identical trajectory forecast integration on the CPU without heap allocations inside
## substep loops.
func _compute_forecasts_cpu(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		timeline_floats: PackedFloat32Array,
		inv_timeline_dt: float
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	var results: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {};
	var well_count: int = well_list.size();
	var well_map: Dictionary[JrOrreryElement, int] = {};
	for w: int in range(well_count):
		if well_list[w] != null:
			well_map[well_list[w]] = w;

	_prepare_cpu_forecast_tables(well_count, timeline_floats);
	for body: JrBallisticOrbiter in bodies:
		var anchor_idx: int = well_map.get(body.get_anchor_element(), 0);
		results[body] = _integrate_single_body_forecast_cpu(body, anchor_idx, inv_timeline_dt);

	_fc_timeline = PackedFloat32Array();
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_SOLVER_FORECAST_CPU,
				Time.get_ticks_usec() - t0
		);
	return results;


## Precomputes integer parent topology, child lists, and gravitational parameters for CPU forecasts.
func _prepare_cpu_forecast_tables(well_count: int, timeline_floats: PackedFloat32Array) -> void:
	_fc_timeline = timeline_floats;
	_fc_well_count = well_count;
	if well_count <= 0 or timeline_floats.is_empty():
		_fc_max_seg = 0;
		_fc_cull_sq = 0.0;
		_fc_well_mu.clear();
		_fc_well_parent.clear();
		_fc_well_children.clear();
		return;

	@warning_ignore("integer_division")
	var total_samples: int = (timeline_floats.size() / FLOATS_PER_TIMELINE_WELL) / well_count;
	_fc_max_seg = maxi(0, total_samples - 1);
	_fc_cull_sq = timeline_floats[7];
	_fc_well_mu.resize(well_count);
	_fc_well_parent.resize(well_count);
	_fc_well_children.resize(well_count);

	for w: int in range(well_count):
		var base: int = w * FLOATS_PER_TIMELINE_WELL;
		_fc_well_mu[w] = timeline_floats[base + 3];
		_fc_well_parent[w] = int(roundf(timeline_floats[base + 6]));
		_fc_well_children[w] = PackedInt32Array();

	for w: int in range(1, well_count):
		var p_idx: int = _fc_well_parent[w];
		if p_idx >= 0 and p_idx < well_count and _fc_well_mu[w] > 0.0:
			_fc_well_children[p_idx].append(w);


## Integrates forward forecast points for a single [param body] starting at [param anchor_idx].
func _integrate_single_body_forecast_cpu(
		body: JrBallisticOrbiter,
		anchor_idx: int,
		inv_timeline_dt: float
) -> PackedVector4Array:
	var steps: int = maxi(1, body.preview_steps);
	var pts: PackedVector4Array = [];
	pts.resize(steps + 1);

	var r_rel: Vector3 = body.get_anchor_relative_position();
	var v_rel: Vector3 = body.get_anchor_relative_velocity();
	pts[0] = Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));

	var step_dt: float = maxf(0.0, body.preview_duration) / float(steps);
	if is_zero_approx(step_dt):
		for k: int in range(1, steps + 1):
			pts[k] = pts[0];
		return pts;

	var ext: Vector3 = body.get_scaled_external_acceleration();
	var is_orbital: bool = body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL;
	var enable_tidal: bool = body.enable_tidal_forces;
	var is_pinned: bool = body.pinned_anchor != null;
	var sub_dt: float = step_dt / float(FORECAST_INTERNAL_SUBSTEPS);
	var elapsed: float = 0.0;
	var mu_anchor: float = (
			_fc_well_mu[anchor_idx]
			if (anchor_idx >= 0 and anchor_idx < _fc_well_count)
			else 0.0
	);

	var sample_coord0: float = 0.0 if enable_tidal else -1.0;
	var pert_accel: Vector3 = (
			_compute_fc_perturbing_fast(r_rel, anchor_idx, sample_coord0)
			+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
	);

	for k: int in range(steps):
		for _s: int in range(FORECAST_INTERNAL_SUBSTEPS):
			v_rel += 0.5 * sub_dt * pert_accel;
			_kepler_drift_into(r_rel, v_rel, mu_anchor, sub_dt);
			r_rel = _drift_r;
			v_rel = _drift_v;
			elapsed += sub_dt;

			var sample_coord: float = (elapsed * inv_timeline_dt) if enable_tidal else -1.0;
			pert_accel = (
					_compute_fc_perturbing_fast(r_rel, anchor_idx, sample_coord)
					+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
			);
			v_rel += 0.5 * sub_dt * pert_accel;

		if not is_pinned and _fc_well_count > 0:
			_fc_cur_r = r_rel;
			_fc_cur_v = v_rel;
			_fc_cur_anchor = anchor_idx;
			if _maybe_switch_fc_anchor_fast(elapsed * inv_timeline_dt, inv_timeline_dt):
				r_rel = _fc_cur_r;
				v_rel = _fc_cur_v;
				anchor_idx = _fc_cur_anchor;
				mu_anchor = _fc_well_mu[anchor_idx];
				var sc: float = (elapsed * inv_timeline_dt) if enable_tidal else -1.0;
				pert_accel = (
						_compute_fc_perturbing_fast(r_rel, anchor_idx, sc)
						+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
				);

		pts[k + 1] = Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));

	return pts;


## Evaluates interpolated timeline perturbing gravity on the CPU using precomputed tables.
func _compute_fc_perturbing_fast(
		r_rel: Vector3,
		anchor_idx: int,
		sample_coord_or_neg: float
) -> Vector3:
	if _fc_well_count <= 0 or _fc_timeline.is_empty():
		return Vector3.ZERO;

	var enable_tidal: bool = sample_coord_or_neg >= 0.0;
	var mu_anchor: float = (
			_fc_well_mu[anchor_idx]
			if (anchor_idx >= 0 and anchor_idx < _fc_well_count)
			else 0.0
	);
	if mu_anchor > 0.0 and not enable_tidal:
		return Vector3.ZERO;

	var clamped_coord: float = clampf(maxf(0.0, sample_coord_or_neg), 0.0, float(_fc_max_seg));
	var idx_a: int = mini(int(floorf(clamped_coord)), _fc_max_seg);
	var frac: float = clamped_coord - float(idx_a);
	var r_rel_sq: float = r_rel.length_squared();
	var total_accel := Vector3.ZERO;

	for k: int in range(_fc_well_count):
		if k == anchor_idx:
			continue;
		var mu_k: float = _fc_well_mu[k];
		if mu_k <= 0.0:
			continue;

		var d_vec: Vector3 = _compute_fc_lca_diff_fast(anchor_idx, k, idx_a, frac);
		var d_sq: float = d_vec.length_squared();
		if d_sq <= 1e-12:
			continue;

		if mu_anchor <= 0.0:
			if not enable_tidal and not _is_fc_well_descendant_fast(k, anchor_idx):
				continue;
			var rho_d: Vector3 = r_rel + d_vec;
			var rho_d_sq: float = rho_d.length_squared();
			if rho_d_sq > 1e-12:
				total_accel -= rho_d * ((mu_k / rho_d_sq) / sqrt(rho_d_sq));
			continue;

		var r_over_d_sq: float = r_rel_sq / d_sq;
		var r_over_d_6: float = r_over_d_sq * r_over_d_sq * r_over_d_sq;
		var mu_ratio: float = mu_k / mu_anchor;
		if 4.0 * (mu_ratio * mu_ratio) * r_over_d_6 < _fc_cull_sq:
			continue;

		var rho: Vector3 = r_rel + d_vec;
		var rho_sq: float = rho.length_squared();
		if rho_sq <= 1e-12:
			continue;

		var rho_factor: float = (mu_k / rho_sq) / sqrt(rho_sq);
		var q: float = r_rel.dot(r_rel + 2.0 * d_vec) / d_sq;
		if absf(q) <= 0.5 and (1.0 + q) > 0.0:
			var one_plus_q: float = 1.0 + q;
			var fq: float = q * (3.0 + 3.0 * q + q * q) / (1.0 + one_plus_q * sqrt(one_plus_q));
			total_accel -= rho_factor * (r_rel - fq * d_vec);
		else:
			var d_factor: float = (mu_k / d_sq) / sqrt(d_sq);
			total_accel -= (rho * rho_factor - d_vec * d_factor);

	return total_accel;


## Computes LCA inter-well position difference at timeline segment [param idx_a] + [param frac].
func _compute_fc_lca_diff_fast(well_a: int, well_b: int, idx_a: int, frac: float) -> Vector3:
	var idx_b: int = mini(idx_a + 1, _fc_max_seg);
	var curr_a: int = well_a;
	var curr_b: int = well_b;
	var diff := Vector3.ZERO;
	var guard: int = 0;
	while curr_a != curr_b and guard < 32:
		guard += 1;
		if curr_a > curr_b:
			if curr_a < 0 or curr_a >= _fc_well_count:
				break;
			diff += _sample_fc_well_pos(curr_a, idx_a, idx_b, frac);
			curr_a = _fc_well_parent[curr_a];
		else:
			if curr_b < 0 or curr_b >= _fc_well_count:
				break;
			diff -= _sample_fc_well_pos(curr_b, idx_a, idx_b, frac);
			curr_b = _fc_well_parent[curr_b];
	return diff;


## Samples interpolated local position of [param well_idx] between [param idx_a] and [param idx_b].
func _sample_fc_well_pos(well_idx: int, idx_a: int, idx_b: int, frac: float) -> Vector3:
	var base_a: int = (idx_a * _fc_well_count + well_idx) * FLOATS_PER_TIMELINE_WELL;
	var pos_a := Vector3(
			_fc_timeline[base_a],
			_fc_timeline[base_a + 1],
			_fc_timeline[base_a + 2]
	);
	if idx_a == idx_b or frac <= 0.0:
		return pos_a;
	var base_b: int = (idx_b * _fc_well_count + well_idx) * FLOATS_PER_TIMELINE_WELL;
	var pos_b := Vector3(
			_fc_timeline[base_b],
			_fc_timeline[base_b + 1],
			_fc_timeline[base_b + 2]
	);
	return pos_a.lerp(pos_b, frac);


## Returns [code]true[/code] when [param candidate] is a descendant of [param ancestor].
func _is_fc_well_descendant_fast(candidate: int, ancestor: int) -> bool:
	if ancestor < 0 or candidate <= ancestor:
		return false;
	var curr: int = candidate;
	var guard: int = 0;
	while curr > ancestor and guard < 32:
		guard += 1;
		if curr >= _fc_well_count:
			return false;
		curr = _fc_well_parent[curr];
		if curr == ancestor:
			return true;
	return false;


## Evaluates mid-forecast Hill-sphere anchor handoffs into [member _fc_cur_r], [member _fc_cur_v],
## and [member _fc_cur_anchor], returning [code]true[/code] if the anchor changed.
func _maybe_switch_fc_anchor_fast(sample_coord: float, inv_timeline_dt: float) -> bool:
	var start_anchor: int = _fc_cur_anchor;
	if start_anchor < 0 or start_anchor >= _fc_well_count:
		return false;

	var clamped: float = clampf(sample_coord, 0.0, float(_fc_max_seg));
	var idx_a: int = mini(int(floorf(clamped)), _fc_max_seg);
	var idx_b: int = mini(idx_a + 1, _fc_max_seg);
	var frac: float = clamped - float(idx_a);

	for _depth: int in range(4):
		if _fc_cur_anchor <= 0 or _fc_cur_anchor >= _fc_well_count:
			break;
		var meta_base: int = (
				(idx_a * _fc_well_count + _fc_cur_anchor) * FLOATS_PER_TIMELINE_WELL + 4
		);
		var exit_r: float = _fc_timeline[meta_base + 1];
		var parent_idx: int = _fc_well_parent[_fc_cur_anchor];
		if parent_idx >= 0 and exit_r > 0.0 and _fc_cur_r.length_squared() > exit_r * exit_r:
			_fc_cur_r += _sample_fc_well_pos(_fc_cur_anchor, idx_a, idx_b, frac);
			if idx_a != idx_b:
				var p_a: Vector3 = _sample_fc_well_pos(_fc_cur_anchor, idx_a, idx_a, 0.0);
				var p_b: Vector3 = _sample_fc_well_pos(_fc_cur_anchor, idx_b, idx_b, 0.0);
				_fc_cur_v += (p_b - p_a) * inv_timeline_dt;
			_fc_cur_anchor = parent_idx;
		else:
			break;

	for _depth: int in range(4):
		if _fc_cur_anchor < 0 or _fc_cur_anchor >= _fc_well_count:
			break;
		var children: PackedInt32Array = _fc_well_children[_fc_cur_anchor];
		if children.is_empty():
			break;
		var entered_child: bool = false;
		for w: int in children:
			var w_meta: int = (idx_a * _fc_well_count + w) * FLOATS_PER_TIMELINE_WELL + 4;
			var entry_r: float = _fc_timeline[w_meta];
			if entry_r <= 0.0:
				continue;
			var child_pos: Vector3 = _sample_fc_well_pos(w, idx_a, idx_b, frac);
			var diff_to_child: Vector3 = _fc_cur_r - child_pos;
			if diff_to_child.length_squared() < entry_r * entry_r:
				_fc_cur_r = diff_to_child;
				if idx_a != idx_b:
					var c_a: Vector3 = _sample_fc_well_pos(w, idx_a, idx_a, 0.0);
					var c_b: Vector3 = _sample_fc_well_pos(w, idx_b, idx_b, 0.0);
					_fc_cur_v -= (c_b - c_a) * inv_timeline_dt;
				_fc_cur_anchor = w;
				entered_child = true;
				break;
		if not entered_child:
			break;

	return _fc_cur_anchor != start_anchor;


## Builds the ordered list of all gravity-well elements (root driver at index 0,
## followed by all valid [JrRailOrbiter] nodes in parent-before-child order).
func _build_forecast_well_list(config: JrBallisticForecastConfig) -> Array[JrOrreryElement]:
	var wells: Array[JrOrreryElement] = [];
	wells.append(config.driver_element);
	for orbiter: JrRailOrbiter in config.orbit_sequence:
		if orbiter != null and orbiter.is_orbit_valid():
			wells.append(orbiter);
	return wells;


## Evaluates the heliocentric origin of well [param well_idx] at [param eval_time].
func _evaluate_well_inertial_origin_at(
		well_idx: int,
		well_list: Array[JrOrreryElement],
		config: JrBallisticForecastConfig,
		eval_time: float
) -> Vector3:
	if well_idx <= 0 or well_idx >= well_list.size():
		return Vector3.ZERO;
	var origin := Vector3.ZERO;
	var curr: JrOrreryElement = well_list[well_idx];
	while curr is JrRailOrbiter:
		var orb := curr as JrRailOrbiter;
		if orb.is_orbit_valid():
			origin += orb.get_relative_position(eval_time, config.reference_plane);
		curr = config.orbiter_parents.get(orb, null);
	return origin;


## Pre-samples parent-relative positions, gravitational parameters, parent well indices,
## and Hill-sphere switch radii across [code]segments + 1[/code] uniform timestamps.
func _sample_well_timeline(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement],
		max_duration: float,
		segments: int
) -> PackedFloat32Array:
	var well_count: int = well_list.size();
	if well_count == 0:
		return PackedFloat32Array();

	var well_map: Dictionary[JrOrreryElement, int] = {};
	for w: int in range(well_count):
		if well_list[w] != null:
			well_map[well_list[w]] = w;

	var well_mu := PackedFloat32Array();
	var well_parent_float := PackedFloat32Array();
	var well_entry_ratio := PackedFloat32Array();
	var well_exit_ratio := PackedFloat32Array();
	well_mu.resize(well_count);
	well_parent_float.resize(well_count);
	well_entry_ratio.resize(well_count);
	well_exit_ratio.resize(well_count);

	var eff_mu_cache: Dictionary[JrOrreryElement, float] = {};
	for w: int in range(1, well_count):
		var orbiter := well_list[w] as JrRailOrbiter;
		var mu: float = orbiter.get_gravity_strength();
		var parent_elem: JrOrreryElement = config.orbiter_parents.get(orbiter, null);
		var parent_idx: int = well_map.get(parent_elem, 0);
		var parent_mu: float = config.driver_gravity_strength;
		if parent_elem != null:
			if not eff_mu_cache.has(parent_elem):
				eff_mu_cache[parent_elem] = parent_elem.get_effective_parent_mu();
			parent_mu = eff_mu_cache[parent_elem];
		var cube_root_ratio: float = (
				pow(mu / (3.0 * parent_mu), 1.0 / 3.0)
				if (mu > 0.0 and parent_mu > 0.0)
				else 0.0
		);
		well_mu[w] = mu;
		well_parent_float[w] = float(parent_idx);
		well_entry_ratio[w] = cube_root_ratio * HILL_ENTRY_MULTIPLIER;
		well_exit_ratio[w] = cube_root_ratio * HILL_EXIT_MULTIPLIER;

	var cull_sq: float = config.tidal_cull_threshold * config.tidal_cull_threshold;
	var floats := PackedFloat32Array();
	floats.resize((segments + 1) * well_count * FLOATS_PER_TIMELINE_WELL);
	var step_time: float = max_duration / float(segments);
	var write_idx: int = 0;

	for seg_idx: int in range(segments + 1):
		var eval_time: float = config.start_time + step_time * float(seg_idx);

		floats[write_idx] = 0.0;
		floats[write_idx + 1] = 0.0;
		floats[write_idx + 2] = 0.0;
		floats[write_idx + 3] = config.driver_gravity_strength;
		floats[write_idx + 4] = 0.0;
		floats[write_idx + 5] = 0.0;
		floats[write_idx + 6] = -1.0;
		floats[write_idx + 7] = cull_sq;
		write_idx += FLOATS_PER_TIMELINE_WELL;

		for w: int in range(1, well_count):
			var orbiter := well_list[w] as JrRailOrbiter;
			var local_pos: Vector3 = orbiter.get_relative_position(
					eval_time,
					config.reference_plane
			);
			var e_ratio: float = well_entry_ratio[w];
			var dist: float = local_pos.length() if e_ratio > 0.0 else 0.0;

			floats[write_idx] = local_pos.x;
			floats[write_idx + 1] = local_pos.y;
			floats[write_idx + 2] = local_pos.z;
			floats[write_idx + 3] = well_mu[w];
			floats[write_idx + 4] = dist * e_ratio;
			floats[write_idx + 5] = dist * well_exit_ratio[w];
			floats[write_idx + 6] = well_parent_float[w];
			floats[write_idx + 7] = cull_sq;
			write_idx += FLOATS_PER_TIMELINE_WELL;

	return floats;


## Samples all active gravity wells in [param config] at timestamp [param eval_time].
func _sample_system_wells_at_time(
		config: JrBallisticPreviewConfig,
		eval_time: float
) -> PackedVector4Array:
	var wells: PackedVector4Array = []
	if config.driver_gravity_strength > 0.0:
		wells.append(Vector4(0.0, 0.0, 0.0, config.driver_gravity_strength))

	if config.orbit_sequence.is_empty():
		return wells

	var orbiter_origins: Dictionary[JrRailOrbiter, Vector3] = {}
	for orbiter: JrRailOrbiter in config.orbit_sequence:
		if orbiter == null or not orbiter.is_orbit_valid():
			continue
		var parent_elem: JrOrreryElement = config.orbiter_parents.get(orbiter, null)
		var parent_origin := Vector3.ZERO
		if parent_elem is JrRailOrbiter:
			parent_origin = orbiter_origins.get(parent_elem as JrRailOrbiter, Vector3.ZERO)
		var inertial_pos: Vector3 = parent_origin + orbiter.get_relative_position(
				eval_time,
				config.reference_plane
		)
		orbiter_origins[orbiter] = inertial_pos

		var mu: float = orbiter.get_gravity_strength()
		if mu > 0.0:
			wells.append(Vector4(inertial_pos.x, inertial_pos.y, inertial_pos.z, mu))

	return wells


## Sums inverse-square gravitational acceleration at [param body_pos] across [param wells].
func _sum_wells_acceleration(body_pos: Vector3, wells: PackedVector4Array) -> Vector3:
	var total_accel := Vector3.ZERO
	for well: Vector4 in wells:
		var diff := Vector3(well.x - body_pos.x, well.y - body_pos.y, well.z - body_pos.z)
		var dist_sq: float = diff.length_squared()
		if dist_sq > 0.0:
			var inv_dist: float = 1.0 / sqrt(dist_sq)
			total_accel += diff * ((well.w / dist_sq) * inv_dist)
	return total_accel
