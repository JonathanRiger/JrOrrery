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
const FORECAST_SHADER_PATH: String = "res://addons/jr_orrery/shaders/ballistic_forecast.glsl";

## Path to the GPU periodic cubic B-spline baker shader for [JrRailOrbiter] mean-to-true anomaly lookup.
const RAIL_SPLINE_SHADER_PATH: String = "res://addons/jr_orrery/shaders/rail_spline_bake.glsl";

## Path to the parallel GPU Cubic Hermite trajectory-tape sampler compute shader.
const TAPE_SAMPLE_SHADER_PATH: String = "res://addons/jr_orrery/shaders/ballistic_tape_sample.glsl";

## Maximum knot capacity per body slot in the resident GPU trajectory-tape atlas (`264` knots).
const GPU_TAPE_SLOT_KNOTS: int = 264;

## Stride in `vec4` entries per body slot in `TapeAtlasBuffer` (`264 pos + 264 vel = 528 vec4s`).
const GPU_TAPE_SLOT_VEC4S: int = 528;

## Byte offset of `vel_flags[0]` inside a body's `TapeAtlasBuffer` slot (`264 * 16 = 4224 bytes`).
const GPU_TAPE_VEL_OFFSET_BYTES: int = GPU_TAPE_SLOT_KNOTS * 16;

## Byte stride per body slot in `TapeAtlasBuffer` (`528 * 16 = 8448 bytes`).
const GPU_TAPE_SLOT_BYTES: int = GPU_TAPE_SLOT_VEC4S * 16;

## Minimum tracked body count at which `TapeAtlasBuffer` residence and fused GPU bake+sample are enabled.
const MIN_GPU_TAPE_ATLAS_BODIES: int = 1024;

## Minimum tracked body count at which standalone GPU trajectory-tape sampling amortizes `submit + sync`.
const MIN_GPU_TAPE_SAMPLE_BODIES: int = 2048;

## Packed position byte stride per body in `TapeSampleOutputBuffer` (`3 x float32 = 12 bytes`).
const TAPE_SAMPLE_POS_BYTES_PER_BODY: int = 12;

## Number of `vec4` entries per sparse event in `TapeSampleOutputBuffer` (`5 x vec4 = 80 bytes`).
const TAPE_SAMPLE_EVENT_VEC4S: int = 5;

## Byte stride per sparse event in `TapeSampleOutputBuffer` (`5 x 16 = 80 bytes`).
const TAPE_SAMPLE_EVENT_BYTES: int = 80;

## Number of sparse event slots speculatively included in the primary single-call GPU readback (`32` events = `2560 bytes`).
const INLINE_TAPE_EVENT_CAP: int = 32;

## Event kind emitted by `ballistic_tape_sample.glsl` when a same-SOI segment switches anchor well.
const TAPE_EVENT_WELL_SWITCH: int = 1;

## Event kind emitted by `ballistic_tape_sample.glsl` with inline `(pa, pb, va, vb)` for a cross-SOI segment.
const TAPE_EVENT_CROSS_SOI: int = 2;

## Event kind emitted by `ballistic_tape_sample.glsl` when `knot_count < 2` (delegating to CPU-resident tape).
const TAPE_EVENT_CPU_FALLBACK: int = 3;

## Byte stride of a single 256-segment baked B-spline table (`256 * sizeof(vec4) = 4096 bytes`).
const BYTES_PER_RAIL_SPLINE: int = JrRailOrbiter.NU_SPLINE_SEGMENTS * BYTES_PER_VEC4;

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
static var _shared_forecast_pipeline_rid := RID();

## Shared compiled rail B-spline baker shader [RID] on [member _shared_rd].
static var _shared_rail_spline_shader_rid := RID();

## Shared rail B-spline baker compute pipeline [RID] on [member _shared_rd].
static var _shared_rail_spline_pipeline_rid := RID();

## Shared compiled trajectory-tape sampler shader [RID] on [member _shared_rd].
static var _shared_tape_sample_shader_rid := RID();

## Shared trajectory-tape sampler compute pipeline [RID] on [member _shared_rd].
static var _shared_tape_sample_pipeline_rid := RID();

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
var _forecast_gpu_available: bool = false;

## Whether the rail B-spline baker GPU compute pipeline compiled successfully.
var _rail_spline_gpu_available: bool = false;

## Whether the trajectory-tape sampler GPU compute pipeline compiled successfully.
var _tape_sample_gpu_available: bool = false;

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
var _forecast_output_capacity: int = 0;

## Compiled rail B-spline baker shader [RID] on [member _rd].
var _rail_spline_shader_rid := RID();

## Rail B-spline baker compute pipeline [RID] on [member _rd].
var _rail_spline_pipeline_rid := RID();

## Storage buffer [RID] for rail spline baker Binding 0 (`EccentricityInputBuffer`).
var _rail_spline_in_rid := RID();

## Storage buffer [RID] for rail spline baker Binding 1 (`SplineSegmentOutputBuffer`).
var _rail_spline_out_rid := RID();

## Uniform set [RID] for rail spline baker bindings 0 and 1.
var _rail_spline_uniform_set_rid := RID();

## Allocated curve capacity of [member _rail_spline_in_rid] and [member _rail_spline_out_rid].
var _rail_spline_capacity: int = 0;

## Cache of baked 256-segment B-spline tables keyed by exact quantized orbital eccentricity.
var _baked_ecc_cache: Dictionary[float, PackedVector4Array] = {};

## Compiled trajectory-tape sampler shader [RID] on [member _rd].
var _tape_sample_shader_rid := RID();

## Trajectory-tape sampler compute pipeline [RID] on [member _rd].
var _tape_sample_pipeline_rid := RID();

## Storage buffer [RID] for resident GPU trajectory-tape knot slots (`TapeAtlasBuffer` & forecast binding 3).
var _tape_atlas_buffer_rid := RID();

## Storage buffer [RID] for per-body GPU trajectory-tape metadata (`TapeMetaBuffer`, 32 bytes/body).
var _tape_meta_buffer_rid := RID();

## Storage buffer [RID] for GPU trajectory-tape sampled positions & sparse events (`TapeSampleOutputBuffer`).
var _tape_sample_out_rid := RID();

## Uniform set [RID] for `ballistic_tape_sample.glsl` bindings 0, 1, and 2.
var _tape_sample_uniform_set_rid := RID();

## Uniform set [RID] for `ballistic_forecast.glsl` writing compact tails directly into [member _tape_sample_out_rid].
var _fused_forecast_uniform_set_rid := RID();

## Allocated body slot capacity of [member _tape_atlas_buffer_rid] and [member _tape_meta_buffer_rid].
var _tape_atlas_capacity: int = 0;

## Allocated byte capacity of [member _tape_sample_out_rid].
var _tape_sample_out_byte_capacity: int = 0;

## Knot offset added to `offsets[i]` when packing `ForecastBodyInputBuffer` (`> 0` during fused single-readback passes).
var _job_out_base_knot_offset: int = 0;

## CPU mirror of `TapeMetaBuffer` (`2 x Vector4` per body) relative to [member _tape_gpu_time_origin].
var _tape_meta_vec4s: PackedVector4Array = PackedVector4Array();

## Active driver body count mirrored in [member _tape_meta_vec4s].
var _tape_meta_body_count: int = 0;

## Whether [member _tape_atlas_buffer_rid] and [member _tape_meta_buffer_rid] hold valid synchronized data.
var _tape_atlas_valid: bool = false;

## Whether [member _tape_meta_vec4s] has pending edits not yet uploaded to [member _tape_meta_buffer_rid].
var _tape_meta_gpu_dirty: bool = false;

## Lowest slot index in [member _tape_meta_vec4s] dirtied since the last GPU upload.
var _tape_meta_dirty_min_slot: int = 0;

## Highest slot index in [member _tape_meta_vec4s] dirtied since the last GPU upload.
var _tape_meta_dirty_max_slot: int = -1;

## Most recent GPU-sampled position buffer (`pos_vecs[seq_idx + 2]`) from [method _readback_unified_sample_and_bake_gpu].
var _last_sampled_pos_vecs: PackedVector3Array = PackedVector3Array();

## Cached gravity-well element list from the most recent [method _sample_well_timeline_bytes] call.
var _tl_cached_wells: Array[JrOrreryElement] = [];

## Cached root gravitational parameter (`mu`) for [member _tl_cached_wells].
var _tl_cached_driver_mu: float = -1.0;

## Cached [JrReferenceDirection] instance for [member _tl_cached_wells].
var _tl_cached_ref_plane: JrReferenceDirection = null;

## Cached per-well `binary_partner != null` flags (`0` or `1`) for [member _tl_cached_wells].
var _tl_cached_has_partner: PackedByteArray = PackedByteArray();

## Cached per-well gravitational parameters (`mu`) in 32-bit float for timeline buffer packing.
var _tl_well_mu: PackedFloat32Array = PackedFloat32Array();

## Cached per-well gravitational parameters (`mu`) in 64-bit float for exact cache validation.
var _tl_well_mu_f64: PackedFloat64Array = PackedFloat64Array();

## Cached per-well parent indices stored as floats for `well_samples`.
var _tl_well_parent_float: PackedFloat32Array = PackedFloat32Array();

## Cached per-well `HILL_ENTRY_MULTIPLIER * (mu / (3 * parent_mu))^(1/3)` factors.
var _tl_well_entry_ratio: PackedFloat32Array = PackedFloat32Array();

## Cached per-well `HILL_EXIT_MULTIPLIER * (mu / (3 * parent_mu))^(1/3)` factors.
var _tl_well_exit_ratio: PackedFloat32Array = PackedFloat32Array();

## Cached per-well fast-position eligibility flags (`1` when `sample_local_position_fast` is valid).
var _tl_use_fast_pos: PackedByteArray = PackedByteArray();

## Reusable staging buffer for [method _sample_well_timeline_bytes].
var _tl_vec4s: PackedVector4Array = PackedVector4Array();

## Cached per-well Keplerian parameter signature detecting runtime orbital element mutations.
var _tl_well_orbital_sig: PackedFloat64Array = PackedFloat64Array();

## Cached maximum `mean_motion` across active [JrRailOrbiter] wells in [member _tl_cached_wells].
var _tl_max_mean_n: float = 0.0;

## Whether [member _tl_grid_v4s] currently holds a valid power-of-two grid-aligned well timeline.
var _tl_grid_valid: bool = false;

## Power-of-two simulation step interval (`2^k`) of [member _tl_grid_v4s].
var _tl_grid_step_dt: float = 0.0;

## Squared tidal cull threshold stored in [member _tl_grid_v4s].
var _tl_grid_cull_sq: float = -1.0;

## Inclusive start integer grid index (`t = idx * _tl_grid_step_dt`) of [member _tl_grid_v4s].
var _tl_grid_start_idx: int = 0;

## Inclusive end integer grid index (`t = idx * _tl_grid_step_dt`) of [member _tl_grid_v4s].
var _tl_grid_end_idx: int = -1;

## Sliding-window cache of pre-sampled `well_samples` rows across `[ _tl_grid_start_idx .. _tl_grid_end_idx ]`.
var _tl_grid_v4s: PackedVector4Array = PackedVector4Array();

## Pre-serialized byte view of [member _tl_grid_v4s] for `O(1)` byte-range slicing on cache hits.
var _tl_grid_bytes: PackedByteArray = PackedByteArray();

## Whether [member _forecast_timeline_buffer_rid] needs to be updated before the next GPU forecast/bake dispatch.
var _tl_gpu_dirty: bool = true;

## Exact `1.0 / step_dt` computed by the most recent [method _sample_tape_timeline_cached] call.
var _tl_last_inv_dt: float = 0.0;

## Exact 64-bit evaluation timestamp passed to [method _execute_fused_bake_and_sample_gpu].
var _fused_eval_time: float = 0.0;

## 64-bit simulation timestamp origin against which `t0_rel_hi / t0_rel_lo` in `TapeMetaBuffer` are measured.
var _tape_gpu_time_origin: float = 0.0;

## Per-job flag (`1` when `ballistic_forecast.glsl` wrote the full tape directly into `TapeAtlasBuffer`).
var _job_wrote_atlas_flags: PackedByteArray = PackedByteArray();

## Per-job flag (`1` when `ballistic_forecast.glsl` wrote only a 2-`vec4` tail summary into `ForecastPointOutputBuffer`).
var _job_compact_out_flags: PackedByteArray = PackedByteArray();

## Cached `WeakRef` to `self` supplied to GPU-atlas-resident [JrBallisticTrajectoryTape] instances.
var _self_weak_ref: WeakRef = null;

## Whether the most recent [method bake_and_sample_tapes_gpu] call fused sampling into its single GPU submission.
var _last_fused_bake_sampled: bool = false;

## Whether a GPU tape sample or fused bake+sample submission is currently executing asynchronously.
var _pending_gpu_sample_sync: bool = false;

## Pending [JrBallisticForecastConfig] for an in-flight fused GPU bake+sample pass (`null` for standalone sample).
var _pending_gpu_sample_config: JrBallisticForecastConfig = null;

## Pending knot offset array for an in-flight fused GPU bake+sample pass.
var _pending_gpu_sample_offsets: PackedInt32Array = PackedInt32Array();

## Pending `tail_base_knot` offset in `TapeSampleOutputBuffer` for an in-flight GPU pass.
var _pending_gpu_sample_tail_base: int = 0;

## Whether `_tape_meta_vec4s` was pre-updated before dispatch during a fused GPU bake+sample pass.
var _meta_precommitted_in_fused: bool = false;

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
		_shared_rd.free_rid(_shared_forecast_shader_rid);
		_shared_forecast_shader_rid = RID();
	if _shared_rail_spline_pipeline_rid.is_valid():
		_shared_rd.free_rid(_shared_rail_spline_pipeline_rid);
		_shared_rail_spline_pipeline_rid = RID();
	if _shared_rail_spline_shader_rid.is_valid():
		_shared_rd.free_rid(_shared_rail_spline_shader_rid);
		_shared_rail_spline_shader_rid = RID();
	if _shared_tape_sample_pipeline_rid.is_valid():
		_shared_rd.free_rid(_shared_tape_sample_pipeline_rid);
		_shared_tape_sample_pipeline_rid = RID();
	if _shared_tape_sample_shader_rid.is_valid():
		_shared_rd.free_rid(_shared_tape_sample_shader_rid);
		_shared_tape_sample_shader_rid = RID();
	_shared_rd.free();
	_shared_rd = null;


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
	_ensure_gpu_initialized();
	return _forecast_gpu_available;


## Returns `true` if the GPU periodic cubic B-spline baker pipeline is active.
func is_rail_spline_gpu_active() -> bool:
	if _force_cpu:
		return false;
	_ensure_gpu_initialized();
	return _rail_spline_gpu_available;


## Returns `true` if the GPU Cubic Hermite trajectory-tape sampler pipeline is active.
func is_tape_sample_gpu_active() -> bool:
	if _force_cpu:
		return false;
	_ensure_gpu_initialized();
	return _tape_sample_gpu_available;


## Marks the resident GPU trajectory-tape atlas unsynchronized (e.g., when body sequence indices change).
func invalidate_tape_atlas() -> void:
	_tape_atlas_valid = false;
	_tape_meta_gpu_dirty = false;
	_tape_meta_dirty_min_slot = 0;
	_tape_meta_dirty_max_slot = -1;
	_tape_meta_body_count = 0;
	_tl_grid_valid = false;
	_tl_grid_bytes = PackedByteArray();
	_tl_gpu_dirty = true;


## Returns `true` when the VRAM `TapeAtlasBuffer` currently holds valid synchronized slot data.
func has_valid_tape_atlas() -> bool:
	return _tape_atlas_valid and _tape_atlas_buffer_rid.is_valid();


## Returns `true` when the most recent [method bake_and_sample_tapes_gpu] call executed a fused GPU sample pass.
func did_last_bake_fuse_sample() -> bool:
	return _last_fused_bake_sampled;


## Returns `true` when an asynchronous GPU tape sample or fused bake+sample submission is awaiting `_rd.sync()`.
func has_pending_gpu_sample_sync() -> bool:
	return _pending_gpu_sample_sync;


## Waits for an in-flight GPU tape sample or fused bake+sample submission, unpacks compact tails, and returns sparse events.
func sync_and_readback_pending_gpu_sample(bodies: Array[JrBallisticOrbiter]) -> PackedVector4Array:
	if not _pending_gpu_sample_sync:
		return PackedVector4Array();
	_pending_gpu_sample_sync = false;
	var cfg: JrBallisticForecastConfig = _pending_gpu_sample_config;
	var offs: PackedInt32Array = _pending_gpu_sample_offsets;
	var tail_base: int = _pending_gpu_sample_tail_base;
	_pending_gpu_sample_config = null;
	_pending_gpu_sample_offsets = PackedInt32Array();

	var rec_tel: bool = _telemetry != null;
	var t0: int = Time.get_ticks_usec() if rec_tel else 0;
	_rd.sync();
	var t1: int = Time.get_ticks_usec() if rec_tel else 0;
	if rec_tel:
		if cfg != null:
			_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_SUBMIT_SYNC, t1 - t0);
		else:
			_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_POST, t1 - t0);
	return _readback_unified_sample_and_bake_gpu(bodies, cfg, offs, tail_base);


## Returns the native `PackedVector3Array` (`pos_vecs[seq_idx + 2]`) read back by the latest GPU tape sample pass.
func get_last_sampled_pos_vecs() -> PackedVector3Array:
	return _last_sampled_pos_vecs;


## Updates `expected_well_idx` (`m1.w`) for [param slot_idx] in [member _tape_meta_vec4s] after an SOI adoption.
func update_body_expected_well_in_atlas(slot_idx: int, well_idx: int) -> void:
	var meta_idx: int = slot_idx * 2 + 1;
	if meta_idx >= 0 and meta_idx < _tape_meta_vec4s.size():
		var m1: Vector4 = _tape_meta_vec4s[meta_idx];
		if int(roundf(m1.w)) != well_idx:
			m1.w = float(well_idx);
			_tape_meta_vec4s[meta_idx] = m1;
			_mark_tape_meta_slot_dirty(slot_idx);


## Clears `loop_period` (`m1.z`) for [param slot_idx] in [member _tape_meta_vec4s] if a speculated periodic loop was perturbed.
func clear_body_periodic_loop_in_atlas(slot_idx: int) -> void:
	var meta_idx: int = slot_idx * 2 + 1;
	if meta_idx >= 0 and meta_idx < _tape_meta_vec4s.size():
		var m1: Vector4 = _tape_meta_vec4s[meta_idx];
		if m1.z != 0.0:
			m1.z = 0.0;
			_tape_meta_vec4s[meta_idx] = m1;
			_mark_tape_meta_slot_dirty(slot_idx);


## Downloads a single body's `pos_anchor` and `vel_flags` arrays from Knot-Major SoA VRAM `TapeAtlasBuffer` on demand.
func materialize_tape_from_gpu_atlas(tape: JrBallisticTrajectoryTape) -> void:
	if tape == null or tape.cpu_knots_resident:
		return;
	if _pending_gpu_sample_sync and _rd != null:
		_rd.sync();
		_pending_gpu_sample_sync = false;
		_pending_gpu_sample_config = null;
		_pending_gpu_sample_offsets = PackedInt32Array();
	var slot_idx: int = tape.atlas_slot_index;
	var k_count: int = tape.knot_count;
	var stride: int = _tape_atlas_capacity;
	if (
			_rd == null
			or not _tape_atlas_buffer_rid.is_valid()
			or slot_idx < 0
			or slot_idx >= stride
			or k_count <= 0
			or k_count > GPU_TAPE_SLOT_KNOTS
	):
		tape.cpu_knots_resident = true;
		return;

	var span_vec4s: int = (k_count - 1) * stride + slot_idx + 1;
	var span_bytes: int = span_vec4s * BYTES_PER_VEC4;
	var vel_base_bytes: int = GPU_TAPE_SLOT_KNOTS * stride * BYTES_PER_VEC4;
	var pos_v4: PackedVector4Array = _rd.buffer_get_data(_tape_atlas_buffer_rid, 0, span_bytes).to_vector4_array();
	var vel_v4: PackedVector4Array = _rd.buffer_get_data(
			_tape_atlas_buffer_rid,
			vel_base_bytes,
			span_bytes
	).to_vector4_array();
	var pos_out := PackedVector4Array();
	var vel_out := PackedVector4Array();
	pos_out.resize(k_count);
	vel_out.resize(k_count);
	for k: int in range(k_count):
		var idx: int = k * stride + slot_idx;
		pos_out[k] = pos_v4[idx];
		vel_out[k] = vel_v4[idx];
	tape.pos_anchor = pos_out;
	tape.vel_flags = vel_out;
	tape.cpu_knots_resident = true;


## Materializes all non-CPU-resident tapes in [param tapes] from Knot-Major SoA VRAM `TapeAtlasBuffer` in one readback.
func materialize_all_tapes_from_gpu_atlas(tapes: Array[JrBallisticTrajectoryTape]) -> void:
	if _rd == null or not _tape_atlas_buffer_rid.is_valid() or tapes.is_empty():
		return;
	if _pending_gpu_sample_sync:
		_rd.sync();
		_pending_gpu_sample_sync = false;
		_pending_gpu_sample_config = null;
		_pending_gpu_sample_offsets = PackedInt32Array();

	var max_slot: int = -1;
	for i: int in range(tapes.size()):
		var t: JrBallisticTrajectoryTape = tapes[i];
		if t != null and t.is_valid and not t.cpu_knots_resident and t.atlas_slot_index >= 0:
			max_slot = maxi(max_slot, t.atlas_slot_index);

	var stride: int = _tape_atlas_capacity;
	if max_slot < 0 or max_slot >= stride:
		return;

	var all_v4: PackedVector4Array = _rd.buffer_get_data(_tape_atlas_buffer_rid).to_vector4_array();
	var vel_row_base: int = GPU_TAPE_SLOT_KNOTS * stride;

	for i: int in range(tapes.size()):
		var t: JrBallisticTrajectoryTape = tapes[i];
		if t == null or not t.is_valid or t.cpu_knots_resident:
			continue;
		var s_idx: int = t.atlas_slot_index;
		var k_count: int = t.knot_count;
		if s_idx >= 0 and s_idx <= max_slot and k_count > 0 and k_count <= GPU_TAPE_SLOT_KNOTS:
			var pos_out := PackedVector4Array();
			var vel_out := PackedVector4Array();
			pos_out.resize(k_count);
			vel_out.resize(k_count);
			for k: int in range(k_count):
				var off: int = k * stride + s_idx;
				pos_out[k] = all_v4[off];
				vel_out[k] = all_v4[vel_row_base + off];
			t.pos_anchor = pos_out;
			t.vel_flags = vel_out;
		t.cpu_knots_resident = true;


## Bakes 256-segment periodic cubic B-spline lookup tables (`M -> nu(M) - M`) for all
## [JrRailOrbiter] nodes in [param orbiters] requiring a bake, deduplicating shared
## eccentricities and dispatching [constant RAIL_SPLINE_SHADER_PATH] on the GPU.
func bake_rail_splines(orbiters: Array[JrRailOrbiter]) -> void:
	if orbiters.is_empty():
		return;

	var pending_orbiters: Array[JrRailOrbiter] = [];
	var unique_ecc_keys: Array[float] = [];
	var unique_eccs: PackedFloat32Array = PackedFloat32Array();
	var missing_ecc_map: Dictionary[float, int] = {};

	for i: int in range(orbiters.size()):
		var orbiter: JrRailOrbiter = orbiters[i];
		if orbiter == null or not is_instance_valid(orbiter):
			continue;
		if not orbiter.needs_nu_spline_bake():
			orbiter._in_dirty_spline_queue = false;
			continue;
		var ecc_key: float = orbiter.eccentricity;
		if _baked_ecc_cache.has(ecc_key):
			orbiter.apply_baked_nu_spline(ecc_key, _baked_ecc_cache[ecc_key]);
			continue;
		pending_orbiters.append(orbiter);
		if not missing_ecc_map.has(ecc_key):
			missing_ecc_map[ecc_key] = unique_ecc_keys.size();
			unique_ecc_keys.append(ecc_key);
			unique_eccs.append(ecc_key);

	if unique_eccs.is_empty():
		return;

	var all_segments: PackedVector4Array = (
			_bake_unique_splines_gpu(unique_eccs)
			if is_rail_spline_gpu_active()
			else _bake_unique_splines_cpu(unique_eccs)
	);
	var seg_stride: int = JrRailOrbiter.NU_SPLINE_SEGMENTS;
	for c: int in range(unique_ecc_keys.size()):
		var start_idx: int = c * seg_stride;
		_baked_ecc_cache[unique_ecc_keys[c]] = all_segments.slice(start_idx, start_idx + seg_stride);

	for p: int in range(pending_orbiters.size()):
		var orb: JrRailOrbiter = pending_orbiters[p];
		var ecc_k: float = orb.eccentricity;
		if _baked_ecc_cache.has(ecc_k):
			orb.apply_baked_nu_spline(ecc_k, _baked_ecc_cache[ecc_k]);
		else:
			orb._in_dirty_spline_queue = false;


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
		_gpu_available = false;
		_forecast_gpu_available = false;
		_rail_spline_gpu_available = false;
		_tape_sample_gpu_available = false;
		_gpu_init_attempted = false;
		return;

	if _pending_gpu_sample_sync:
		_rd.sync();
		_pending_gpu_sample_sync = false;
		_pending_gpu_sample_config = null;
		_pending_gpu_sample_offsets = PackedInt32Array();

	_dispose_stepping_rids();
	_dispose_forecast_rids();
	_dispose_tape_sample_rids();
	_dispose_rail_spline_rids();
	_shader_rid = RID();
	_pipeline_rid = RID();
	_forecast_shader_rid = RID();
	_forecast_pipeline_rid = RID();
	_rail_spline_shader_rid = RID();
	_rail_spline_pipeline_rid = RID();
	_tape_sample_shader_rid = RID();
	_tape_sample_pipeline_rid = RID();

	if _holds_shared_rd:
		_holds_shared_rd = false;
		_shared_rd_ref_count = maxi(0, _shared_rd_ref_count - 1);
		if _shared_rd_ref_count == 0 and _shared_rd != null:
			_dispose_shared_rd();

	_rd = null;
	_body_capacity = 0;
	_well_capacity = 0;
	_forecast_body_capacity = 0;
	_forecast_timeline_capacity = 0;
	_forecast_output_capacity = 0;
	_rail_spline_capacity = 0;
	_tape_atlas_capacity = 0;
	_tape_sample_out_byte_capacity = 0;
	_tape_atlas_valid = false;
	_tape_meta_gpu_dirty = false;
	_tape_meta_dirty_min_slot = 0;
	_tape_meta_dirty_max_slot = -1;
	_tape_meta_body_count = 0;
	_last_sampled_pos_vecs = PackedVector3Array();
	_tl_cached_wells.clear();
	_tl_cached_driver_mu = -1.0;
	_tl_cached_ref_plane = null;
	_tl_grid_valid = false;
	_tl_grid_v4s = PackedVector4Array();
	_tl_grid_bytes = PackedByteArray();
	_gpu_available = false;
	_forecast_gpu_available = false;
	_rail_spline_gpu_available = false;
	_tape_sample_gpu_available = false;
	_gpu_init_attempted = false;


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

	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	var well_list: Array[JrOrreryElement] = _resolve_forecast_well_list(config);
	var segments: int = _compute_adaptive_timeline_segments(config, well_list, max_duration);
	var timeline_bytes: PackedByteArray = _sample_well_timeline_bytes(
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
	var inv_timeline_dt: float = float(segments) / max_duration if max_duration > 1e-12 else 0.0;

	if should_use_gpu_for_forecast(valid_bodies, well_list.size()):
		return _compute_forecasts_gpu(valid_bodies, well_list, timeline_bytes, inv_timeline_dt);
	return _compute_forecasts_cpu(
			valid_bodies,
			well_list,
			timeline_bytes.to_float32_array(),
			inv_timeline_dt
	);


## Bakes or extends rolling Cubic Hermite [JrBallisticTrajectoryTape] horizons in parallel
## for all [JrBallisticOrbiter] jobs described in [param config].
func bake_trajectory_tapes(config: JrBallisticForecastConfig) -> void:
	_last_fused_bake_sampled = false;
	if config == null or config.bodies.is_empty():
		return;

	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	var well_list: Array[JrOrreryElement] = _resolve_forecast_well_list(config);
	var timeline_bytes: PackedByteArray;
	var inv_timeline_dt: float;
	if (
			well_list.size() > 1
			and config.job_start_times.size() == config.bodies.size()
			and config.timeline_segments <= 0
	):
		timeline_bytes = _sample_tape_timeline_cached(config, well_list);
		inv_timeline_dt = _tl_last_inv_dt;
	else:
		var max_duration: float = _resolve_tape_batch_duration(config);
		var segments: int = _compute_adaptive_timeline_segments(config, well_list, max_duration);
		timeline_bytes = _sample_well_timeline_bytes(
				config,
				well_list,
				max_duration,
				segments
		);
		inv_timeline_dt = float(segments) / max_duration if max_duration > 1e-12 else 0.0;
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_FORECAST_TIMELINE_SAMPLE,
				Time.get_ticks_usec() - t0
		);

	var body_count: int = config.bodies.size();
	var total_knots: int = _compute_full_tape_knot_sum(config);
	var bake_work: int = _compute_tape_bake_work(config, well_list.size());

	if should_use_gpu_for_tape_bake(total_knots, body_count, bake_work):
		_bake_tapes_gpu(config, well_list.size(), timeline_bytes, inv_timeline_dt);
	else:
		_bake_tapes_cpu(config, well_list.size(), timeline_bytes.to_float32_array(), inv_timeline_dt);


## Executes trajectory-tape baking and parallel GPU Cubic Hermite sampling in a single
## compute command list, single GPU fence wait, and single readback when all jobs write to `TapeAtlasBuffer`.
func bake_and_sample_tapes_gpu(
		config: JrBallisticForecastConfig,
		all_bodies: Array[JrBallisticOrbiter],
		eval_time: float
) -> PackedVector4Array:
	_last_fused_bake_sampled = false;
	if config == null or config.bodies.is_empty():
		return PackedVector4Array();

	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	var well_list: Array[JrOrreryElement] = _resolve_forecast_well_list(config);
	var timeline_bytes: PackedByteArray;
	var inv_timeline_dt: float;
	if (
			well_list.size() > 1
			and config.job_start_times.size() == config.bodies.size()
			and config.timeline_segments <= 0
	):
		timeline_bytes = _sample_tape_timeline_cached(config, well_list);
		inv_timeline_dt = _tl_last_inv_dt;
	else:
		var max_duration: float = _resolve_tape_batch_duration(config);
		var segments: int = _compute_adaptive_timeline_segments(config, well_list, max_duration);
		timeline_bytes = _sample_well_timeline_bytes(
				config,
				well_list,
				max_duration,
				segments
		);
		inv_timeline_dt = float(segments) / max_duration if max_duration > 1e-12 else 0.0;
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_FORECAST_TIMELINE_SAMPLE,
				Time.get_ticks_usec() - t0
		);

	var well_count: int = well_list.size();
	if (
			is_forecast_gpu_active()
			and is_tape_sample_gpu_active()
			and all_bodies.size() >= MIN_GPU_TAPE_ATLAS_BODIES
			and _can_all_jobs_precommit_atlas(config)
	):
		_fused_eval_time = eval_time;
		return _execute_fused_bake_and_sample_gpu(
				config,
				all_bodies,
				timeline_bytes,
				Vector3(inv_timeline_dt, float(well_count), eval_time - config.start_time)
		);

	var body_count: int = config.bodies.size();
	var total_knots: int = _compute_full_tape_knot_sum(config);
	var bake_work: int = _compute_tape_bake_work(config, well_count);
	if should_use_gpu_for_tape_bake(total_knots, body_count, bake_work):
		_bake_tapes_gpu(config, well_count, timeline_bytes, inv_timeline_dt);
	else:
		_bake_tapes_cpu(config, well_count, timeline_bytes.to_float32_array(), inv_timeline_dt);
	return PackedVector4Array();


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


## Returns `true` when [param total_knots], [param body_count], and [param total_substep_work]
## amortize local [RenderingDevice] fence submission for rolling trajectory-tape baking.
func should_use_gpu_for_tape_bake(
		total_knots: int,
		body_count: int,
		total_substep_work: int
) -> bool:
	if not is_forecast_gpu_active() or body_count <= 0 or total_knots <= 0:
		return false;
	if body_count >= 4 and total_knots >= 128:
		return true;
	if body_count >= 4 and total_substep_work >= 1536:
		return true;
	return total_substep_work >= 8192 or total_knots >= 2048;


## Computes total `(steps * substeps * well_count)` integration work across all jobs in [param config].
func _compute_tape_bake_work(config: JrBallisticForecastConfig, well_count: int) -> int:
	var count: int = config.bodies.size();
	var has_steps: bool = not config.job_step_counts.is_empty();
	var has_subs: bool = not config.job_substeps.is_empty();
	var total_substeps: int = 0;
	for i: int in range(count):
		var steps: int = (
				maxi(1, config.job_step_counts[i])
				if has_steps
				else maxi(1, config.bodies[i].preview_steps)
		);
		var subs: int = (
				maxi(1, config.job_substeps[i])
				if has_subs
				else FORECAST_INTERNAL_SUBSTEPS
		);
		total_substeps += steps * subs;
	return total_substeps * maxi(1, well_count);


## Computes the sum of `steps + 1` knots across all jobs in [param config].
func _compute_full_tape_knot_sum(config: JrBallisticForecastConfig) -> int:
	var count: int = config.bodies.size();
	var has_steps: bool = not config.job_step_counts.is_empty();
	var sum_knots: int = 0;
	for i: int in range(count):
		var steps: int = (
				maxi(1, config.job_step_counts[i])
				if has_steps
				else maxi(1, config.bodies[i].preview_steps)
		);
		sum_knots += steps + 1;
	return sum_knots;


## Computes forward-integrated inertial trajectory polylines for all [JrBallisticOrbiter]
## nodes in [param config] starting from [member JrBallisticForecastConfig.start_time].
func compute_forecasts_inertial(
		config: JrBallisticForecastConfig
) -> Dictionary[JrBallisticOrbiter, PackedVector3Array]:
	var rel_map: Dictionary[JrBallisticOrbiter, PackedVector4Array] = (
			compute_forecasts_anchor_relative(config)
	)
	var well_list: Array[JrOrreryElement] = _resolve_forecast_well_list(config)
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
	return _resolve_forecast_well_list(config);


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
		return;
	_gpu_init_attempted = true;

	if RenderingServer.get_current_rendering_driver_name() == "dummy":
		_gpu_available = false;
		_forecast_gpu_available = false;
		_rail_spline_gpu_available = false;
		_tape_sample_gpu_available = false;
		return;

	if _shared_rd == null:
		_shared_rd = RenderingServer.create_local_rendering_device();
		if _shared_rd == null:
			_gpu_available = false;
			_forecast_gpu_available = false;
			_rail_spline_gpu_available = false;
			_tape_sample_gpu_available = false;
			return;
		_rd = _shared_rd;
		_init_stepping_pipeline();
		_init_forecast_pipeline();
		_init_rail_spline_pipeline();
		_init_tape_sample_pipeline();
		_shared_shader_rid = _shader_rid;
		_shared_pipeline_rid = _pipeline_rid;
		_shared_forecast_shader_rid = _forecast_shader_rid;
		_shared_forecast_pipeline_rid = _forecast_pipeline_rid;
		_shared_rail_spline_shader_rid = _rail_spline_shader_rid;
		_shared_rail_spline_pipeline_rid = _rail_spline_pipeline_rid;
		_shared_tape_sample_shader_rid = _tape_sample_shader_rid;
		_shared_tape_sample_pipeline_rid = _tape_sample_pipeline_rid;
		if (
				not _shared_pipeline_rid.is_valid()
				and not _shared_forecast_pipeline_rid.is_valid()
				and not _shared_rail_spline_pipeline_rid.is_valid()
				and not _shared_tape_sample_pipeline_rid.is_valid()
		):
			_dispose_shared_rd();
			_rd = null;
			return;

	_rd = _shared_rd;
	_shader_rid = _shared_shader_rid;
	_pipeline_rid = _shared_pipeline_rid;
	_forecast_shader_rid = _shared_forecast_shader_rid;
	_forecast_pipeline_rid = _shared_forecast_pipeline_rid;
	_rail_spline_shader_rid = _shared_rail_spline_shader_rid;
	_rail_spline_pipeline_rid = _shared_rail_spline_pipeline_rid;
	_tape_sample_shader_rid = _shared_tape_sample_shader_rid;
	_tape_sample_pipeline_rid = _shared_tape_sample_pipeline_rid;
	_gpu_available = _pipeline_rid.is_valid();
	_forecast_gpu_available = _forecast_pipeline_rid.is_valid();
	_rail_spline_gpu_available = _rail_spline_pipeline_rid.is_valid();
	_tape_sample_gpu_available = _tape_sample_pipeline_rid.is_valid();
	_shared_rd_ref_count += 1;
	_holds_shared_rd = true;


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

	_forecast_pipeline_rid = _rd.compute_pipeline_create(_forecast_shader_rid);
	_forecast_gpu_available = _forecast_pipeline_rid.is_valid();


## Compiles [constant RAIL_SPLINE_SHADER_PATH] and creates [member _rail_spline_pipeline_rid].
func _init_rail_spline_pipeline() -> void:
	var spirv: RDShaderSPIRV = _load_or_compile_spirv(RAIL_SPLINE_SHADER_PATH);
	if (
			spirv == null
			or not spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE).is_empty()
	):
		return;

	_rail_spline_shader_rid = _rd.shader_create_from_spirv(spirv);
	if not _rail_spline_shader_rid.is_valid():
		return;

	_rail_spline_pipeline_rid = _rd.compute_pipeline_create(_rail_spline_shader_rid);
	_rail_spline_gpu_available = _rail_spline_pipeline_rid.is_valid();


## Compiles [constant TAPE_SAMPLE_SHADER_PATH] and creates [member _tape_sample_pipeline_rid].
func _init_tape_sample_pipeline() -> void:
	var spirv: RDShaderSPIRV = _load_or_compile_spirv(TAPE_SAMPLE_SHADER_PATH);
	if (
			spirv == null
			or not spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE).is_empty()
	):
		return;

	_tape_sample_shader_rid = _rd.shader_create_from_spirv(spirv);
	if not _tape_sample_shader_rid.is_valid():
		return;

	_tape_sample_pipeline_rid = _rd.compute_pipeline_create(_tape_sample_shader_rid);
	_tape_sample_gpu_available = _tape_sample_pipeline_rid.is_valid();


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
	_tl_gpu_dirty = true;
	if _forecast_output_buffer_rid.is_valid():
		_rd.free_rid(_forecast_output_buffer_rid);
		_forecast_output_buffer_rid = RID();


## Frees all per-instance rail spline baker buffers and uniform set RIDs on [member _rd].
func _dispose_rail_spline_rids() -> void:
	if _rail_spline_uniform_set_rid.is_valid() and _rd.uniform_set_is_valid(_rail_spline_uniform_set_rid):
		_rd.free_rid(_rail_spline_uniform_set_rid);
	_rail_spline_uniform_set_rid = RID();
	if _rail_spline_in_rid.is_valid():
		_rd.free_rid(_rail_spline_in_rid);
		_rail_spline_in_rid = RID();
	if _rail_spline_out_rid.is_valid():
		_rd.free_rid(_rail_spline_out_rid);
		_rail_spline_out_rid = RID();


## Frees all per-instance GPU trajectory-tape atlas and sampler buffers and uniform set RIDs.
func _dispose_tape_sample_rids() -> void:
	_tape_atlas_valid = false;
	_tape_meta_gpu_dirty = false;
	_tape_sample_out_byte_capacity = 0;
	_free_forecast_uniform_set();
	if _tape_sample_uniform_set_rid.is_valid() and _rd.uniform_set_is_valid(_tape_sample_uniform_set_rid):
		_rd.free_rid(_tape_sample_uniform_set_rid);
	_tape_sample_uniform_set_rid = RID();
	if _tape_atlas_buffer_rid.is_valid():
		_rd.free_rid(_tape_atlas_buffer_rid);
		_tape_atlas_buffer_rid = RID();
	if _tape_meta_buffer_rid.is_valid():
		_rd.free_rid(_tape_meta_buffer_rid);
		_tape_meta_buffer_rid = RID();
	if _tape_sample_out_rid.is_valid():
		_rd.free_rid(_tape_sample_out_rid);
		_tape_sample_out_rid = RID();


## Ensures GPU trajectory-tape atlas, metadata, and output SSBOs have capacity for [param required_slots].
func _ensure_tape_atlas_capacities(required_slots: int) -> void:
	var target: int = maxi(1, required_slots);
	if (
			target <= _tape_atlas_capacity
			and _tape_atlas_buffer_rid.is_valid()
			and _tape_meta_buffer_rid.is_valid()
			and _tape_sample_out_rid.is_valid()
			and (not _tape_sample_gpu_available or _tape_sample_uniform_set_rid.is_valid())
	):
		return;

	_dispose_tape_sample_rids();
	_tape_atlas_capacity = _next_power_of_two(target);
	_tape_atlas_buffer_rid = _rd.storage_buffer_create(_tape_atlas_capacity * GPU_TAPE_SLOT_BYTES);
	_tape_meta_buffer_rid = _rd.storage_buffer_create(_tape_atlas_capacity * 2 * BYTES_PER_VEC4);
	var default_out_bytes: int = (
			64
			+ _tape_atlas_capacity
			* (TAPE_SAMPLE_POS_BYTES_PER_BODY + 32 + TAPE_SAMPLE_EVENT_BYTES)
	);
	_ensure_tape_sample_out_bytes(default_out_bytes);


## Ensures [member _tape_sample_out_rid] has at least [param required_bytes] and refreshes bound uniform sets.
func _ensure_tape_sample_out_bytes(required_bytes: int) -> void:
	if (
			required_bytes <= _tape_sample_out_byte_capacity
			and _tape_sample_out_rid.is_valid()
			and (not _tape_sample_gpu_available or _tape_sample_uniform_set_rid.is_valid())
	):
		return;

	if _tape_sample_uniform_set_rid.is_valid() and _rd.uniform_set_is_valid(_tape_sample_uniform_set_rid):
		_rd.free_rid(_tape_sample_uniform_set_rid);
	_tape_sample_uniform_set_rid = RID();
	if _fused_forecast_uniform_set_rid.is_valid():
		_rd.free_rid(_fused_forecast_uniform_set_rid);
	_fused_forecast_uniform_set_rid = RID();
	if _tape_sample_out_rid.is_valid():
		_rd.free_rid(_tape_sample_out_rid);
		_tape_sample_out_rid = RID();

	_tape_sample_out_byte_capacity = _next_power_of_two(maxi(256, required_bytes));
	_tape_sample_out_rid = _rd.storage_buffer_create(_tape_sample_out_byte_capacity);

	if (
			_tape_sample_gpu_available
			and _tape_sample_shader_rid.is_valid()
			and _tape_atlas_buffer_rid.is_valid()
			and _tape_meta_buffer_rid.is_valid()
	):
		var u_atlas := RDUniform.new();
		u_atlas.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
		u_atlas.binding = 0;
		u_atlas.add_id(_tape_atlas_buffer_rid);

		var u_meta := RDUniform.new();
		u_meta.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
		u_meta.binding = 1;
		u_meta.add_id(_tape_meta_buffer_rid);

		var u_out := RDUniform.new();
		u_out.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
		u_out.binding = 2;
		u_out.add_id(_tape_sample_out_rid);

		_tape_sample_uniform_set_rid = _rd.uniform_set_create(
				[u_atlas, u_meta, u_out],
				_tape_sample_shader_rid,
				0
		);

	if (
			_forecast_body_buffer_rid.is_valid()
			and _forecast_timeline_buffer_rid.is_valid()
			and _forecast_output_buffer_rid.is_valid()
			and _forecast_shader_rid.is_valid()
	):
		_rebuild_forecast_uniform_set();


## Samples all [param bodies] in parallel on the GPU at [param eval_time] via [constant TAPE_SAMPLE_SHADER_PATH],
## writing `body._anchor_relative_position` in-place and returning sparse `5-vec4` event records.
func sample_tapes_gpu(bodies: Array[JrBallisticOrbiter], eval_time: float) -> PackedVector4Array:
	var count: int = bodies.size();
	if count <= 0 or not is_tape_sample_gpu_active():
		return PackedVector4Array();

	_ensure_tape_atlas_capacities(count);
	var pos_end_bytes: int = STEP_OUT_HEADER_BYTES + count * TAPE_SAMPLE_POS_BYTES_PER_BODY;
	var tail_base_knot: int = int(ceili(float(pos_end_bytes) / 32.0));
	_ensure_tape_sample_out_bytes(tail_base_knot * 32 + count * TAPE_SAMPLE_EVENT_BYTES);

	if not _tape_atlas_valid or _tape_meta_body_count != count:
		_sync_all_tapes_to_gpu_atlas(bodies, eval_time);
	elif absf(eval_time - _tape_gpu_time_origin) > MAX_GPU_TIME_ORIGIN_SPAN:
		_rebase_tape_meta_time_origin(bodies, eval_time);
	elif _tape_meta_gpu_dirty:
		_flush_dirty_tape_meta_to_gpu();

	_rd.buffer_update(_tape_sample_out_rid, 0, 4, _zero_u32_bytes);

	var rel_pair: Vector2 = _split_float64_to_f32_pair(eval_time - _tape_gpu_time_origin);
	var events_word_offset: int = tail_base_knot * 8;
	var push_bytes := PackedByteArray();
	push_bytes.resize(PUSH_CONSTANT_BYTES);
	push_bytes.encode_float(0, rel_pair.x);
	push_bytes.encode_float(4, rel_pair.y);
	push_bytes.encode_u32(8, count);
	push_bytes.encode_u32(12, events_word_offset);

	var groups_x: int = int(ceili(float(count) / float(WORKGROUP_SIZE_X)));
	var cl: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(cl, _tape_sample_pipeline_rid);
	_rd.compute_list_bind_uniform_set(cl, _tape_sample_uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(cl, push_bytes, PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(cl, groups_x, 1, 1);
	_rd.compute_list_end();

	_rd.submit();
	_pending_gpu_sample_sync = true;
	_pending_gpu_sample_config = null;
	_pending_gpu_sample_offsets = PackedInt32Array();
	_pending_gpu_sample_tail_base = tail_base_knot;
	return PackedVector4Array();


## Performs a single primary `_rd.buffer_get_data` readback on [member _tape_sample_out_rid] to unpack
## sampled positions, optional fused compact-tail knots, and inline `5-vec4` SOI/fallback events.
func _readback_unified_sample_and_bake_gpu(
		bodies: Array[JrBallisticOrbiter],
		config: JrBallisticForecastConfig,
		offsets: PackedInt32Array,
		tail_base_knot: int
) -> PackedVector4Array:
	var count: int = bodies.size();
	var pos_end_bytes: int = STEP_OUT_HEADER_BYTES + count * TAPE_SAMPLE_POS_BYTES_PER_BODY;
	var bake_knots: int = offsets[config.bodies.size()] if config != null else 0;
	var tail_byte_offset: int = tail_base_knot * 32;
	var events_byte_offset: int = (tail_base_knot + bake_knots) * 32;
	var inline_ev_cap: int = mini(count, INLINE_TAPE_EVENT_CAP);
	var read_bytes: int = events_byte_offset + inline_ev_cap * TAPE_SAMPLE_EVENT_BYTES;

	var rec_tel: bool = _telemetry != null;
	var t_rb0: int = Time.get_ticks_usec() if rec_tel else 0;
	var raw_bytes: PackedByteArray = _rd.buffer_get_data(_tape_sample_out_rid, 0, read_bytes);

	if config != null:
		if bake_knots > 0:
			var bake_vec4s: PackedVector4Array = (
					raw_bytes.slice(tail_byte_offset, events_byte_offset).to_vector4_array()
			);
			_apply_unpacked_tape_slices(config, offsets, bake_vec4s);
		else:
			_apply_unpacked_tape_slices(config, offsets, PackedVector4Array());
		_meta_precommitted_in_fused = false;
		if rec_tel:
			_telemetry.add_phase_usec(
					JrOrreryTelemetry.PHASE_SOLVER_FORECAST_READBACK_UNPACK,
					Time.get_ticks_usec() - t_rb0
			);

	var t_post0: int = (Time.get_ticks_usec() if config != null else t_rb0) if rec_tel else 0;
	var event_count: int = mini(raw_bytes.decode_u32(0), count);
	_last_sampled_pos_vecs = raw_bytes.slice(0, pos_end_bytes).to_vector3_array();
	if event_count == 0:
		if rec_tel:
			_telemetry.add_phase_usec(
					JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_POST,
					Time.get_ticks_usec() - t_post0
			);
		return PackedVector4Array();

	var ev_bytes: PackedByteArray;
	if event_count <= inline_ev_cap:
		ev_bytes = raw_bytes.slice(
				events_byte_offset,
				events_byte_offset + event_count * TAPE_SAMPLE_EVENT_BYTES
		);
	else:
		ev_bytes = raw_bytes.slice(
				events_byte_offset,
				events_byte_offset + inline_ev_cap * TAPE_SAMPLE_EVENT_BYTES
		);
		var extra_bytes: PackedByteArray = _rd.buffer_get_data(
				_tape_sample_out_rid,
				events_byte_offset + inline_ev_cap * TAPE_SAMPLE_EVENT_BYTES,
				(event_count - inline_ev_cap) * TAPE_SAMPLE_EVENT_BYTES
		);
		ev_bytes.append_array(extra_bytes);

	var ev_vec4s: PackedVector4Array = ev_bytes.to_vector4_array();
	if rec_tel:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_BALLISTIC_ANCHOR_POST,
				Time.get_ticks_usec() - t_post0
		);
	return ev_vec4s;


## Ensures rail spline baker SSBOs and uniform set have capacity for [param required_curves].
func _ensure_rail_spline_capacities(required_curves: int) -> void:
	var target: int = maxi(1, required_curves);
	if (
			target <= _rail_spline_capacity
			and _rail_spline_in_rid.is_valid()
			and _rail_spline_out_rid.is_valid()
			and _rail_spline_uniform_set_rid.is_valid()
			and _rd.uniform_set_is_valid(_rail_spline_uniform_set_rid)
	):
		return;

	_dispose_rail_spline_rids();
	_rail_spline_capacity = _next_power_of_two(target);
	_rail_spline_in_rid = _rd.storage_buffer_create(_rail_spline_capacity * 4);
	_rail_spline_out_rid = _rd.storage_buffer_create(_rail_spline_capacity * BYTES_PER_RAIL_SPLINE);

	var u_in := RDUniform.new();
	u_in.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_in.binding = 0;
	u_in.add_id(_rail_spline_in_rid);

	var u_out := RDUniform.new();
	u_out.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_out.binding = 1;
	u_out.add_id(_rail_spline_out_rid);

	_rail_spline_uniform_set_rid = _rd.uniform_set_create([u_in, u_out], _rail_spline_shader_rid, 0);


## Dispatches [constant RAIL_SPLINE_SHADER_PATH] on the GPU to bake [param unique_eccs] in parallel.
func _bake_unique_splines_gpu(unique_eccs: PackedFloat32Array) -> PackedVector4Array:
	var curve_count: int = unique_eccs.size();
	_ensure_rail_spline_capacities(curve_count);

	var ecc_bytes: PackedByteArray = unique_eccs.to_byte_array();
	_rd.buffer_update(_rail_spline_in_rid, 0, ecc_bytes.size(), ecc_bytes);

	var push_bytes := PackedByteArray();
	push_bytes.resize(PUSH_CONSTANT_BYTES);
	push_bytes.encode_u32(0, curve_count);

	var cl: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(cl, _rail_spline_pipeline_rid);
	_rd.compute_list_bind_uniform_set(cl, _rail_spline_uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(cl, push_bytes, push_bytes.size());
	_rd.compute_list_dispatch(cl, curve_count, 1, 1);
	_rd.compute_list_end();

	_rd.submit();
	_rd.sync();

	var out_bytes: PackedByteArray = _rd.buffer_get_data(
			_rail_spline_out_rid,
			0,
			curve_count * BYTES_PER_RAIL_SPLINE
	);
	return out_bytes.to_vector4_array();


## CPU fallback matching [constant RAIL_SPLINE_SHADER_PATH] when no GPU [RenderingDevice] is active.
func _bake_unique_splines_cpu(unique_eccs: PackedFloat32Array) -> PackedVector4Array:
	var seg_count: int = JrRailOrbiter.NU_SPLINE_SEGMENTS;
	var curve_count: int = unique_eccs.size();
	var out_segments := PackedVector4Array();
	out_segments.resize(curve_count * seg_count);

	var deltas := PackedFloat32Array();
	deltas.resize(seg_count);
	var ctrls := PackedFloat32Array();
	ctrls.resize(seg_count);

	for c: int in range(curve_count):
		_bake_single_eccentricity_cpu(unique_eccs[c], c * seg_count, deltas, ctrls);
		var base: int = c * seg_count;
		for k: int in range(seg_count):
			var p0: float = ctrls[(k + 255) & 255];
			var p1: float = ctrls[k];
			var p2: float = ctrls[(k + 1) & 255];
			var p3: float = ctrls[(k + 2) & 255];
			out_segments[base + k] = Vector4(
					deltas[k],
					0.5 * (p2 - p0),
					0.5 * (p0 - 2.0 * p1 + p2),
					(p3 - p0 + 3.0 * (p1 - p2)) * (1.0 / 6.0)
			);
	return out_segments;


## Populates [param deltas] and periodic cubic B-spline control points [param ctrls] for [param ecc].
func _bake_single_eccentricity_cpu(
		ecc: float,
		_out_offset: int,
		deltas: PackedFloat32Array,
		ctrls: PackedFloat32Array
) -> void:
	var clamped_e: float = clampf(ecc, 0.0, 0.9999);
	var beta: float = clamped_e / (1.0 + sqrt(maxf(0.0, 1.0 - clamped_e * clamped_e)));
	var step_rad: float = JrRailOrbiter.NU_SPLINE_STEP;

	for k: int in range(256):
		if k == 0 or k == 128 or clamped_e <= 1e-7:
			deltas[k] = 0.0;
			continue;
		var mean_anom: float = -PI + float(k) * step_rad;
		var de: float = clamped_e * sin(mean_anom);
		for _iter: int in range(8):
			var e_anom: float = mean_anom + de;
			de -= (de - clamped_e * sin(e_anom)) / maxf(1e-6, 1.0 - clamped_e * cos(e_anom));
		var e_final: float = mean_anom + de;
		var nu_minus_e: float = 2.0 * atan(
				(beta * sin(e_final)) / maxf(1e-7, 1.0 - beta * cos(e_final))
		);
		deltas[k] = de + nu_minus_e;

	const POLE_Z1: float = -0.2679491924311227;
	const GAIN_C0: float = 1.7320508075688772;
	for k: int in range(256):
		var sum_val: float = deltas[k];
		var z_pow: float = POLE_Z1;
		for tap: int in range(1, 17):
			sum_val += z_pow * (deltas[(k - tap) & 255] + deltas[(k + tap) & 255]);
			z_pow *= POLE_Z1;
		ctrls[k] = GAIN_C0 * sum_val;


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

	if body.is_tape_position_pending():
		body.ensure_tape_position();
	if body.is_tape_velocity_pending():
		body.ensure_tape_velocity();
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
			body._tape_vel_pending = false;
			body._obeying_shared_tape_clock = false;
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
	if not _tape_atlas_buffer_rid.is_valid():
		_ensure_tape_atlas_capacities(8);

	var target_bodies: int = maxi(1, req_bodies);
	var target_timeline: int = maxi(2, req_timeline_entries);
	var target_output: int = maxi(1, req_output_points);
	var recreated: bool = false;

	if target_bodies > _forecast_body_capacity or not _forecast_body_buffer_rid.is_valid():
		_free_forecast_uniform_set();
		if _forecast_body_buffer_rid.is_valid():
			_rd.free_rid(_forecast_body_buffer_rid);
		_forecast_body_capacity = _next_power_of_two(target_bodies);
		_forecast_body_buffer_rid = _rd.storage_buffer_create(
				_forecast_body_capacity * BYTES_PER_BODY
		);
		recreated = true;

	if (
			target_timeline > _forecast_timeline_capacity
			or not _forecast_timeline_buffer_rid.is_valid()
	):
		_free_forecast_uniform_set();
		if _forecast_timeline_buffer_rid.is_valid():
			_rd.free_rid(_forecast_timeline_buffer_rid);
		_forecast_timeline_capacity = _next_power_of_two(target_timeline);
		_forecast_timeline_buffer_rid = _rd.storage_buffer_create(
				_forecast_timeline_capacity * BYTES_PER_VEC4
		);
		_tl_gpu_dirty = true;
		recreated = true;

	if target_output > _forecast_output_capacity or not _forecast_output_buffer_rid.is_valid():
		_free_forecast_uniform_set();
		if _forecast_output_buffer_rid.is_valid():
			_rd.free_rid(_forecast_output_buffer_rid);
		_forecast_output_capacity = _next_power_of_two(target_output);
		_forecast_output_buffer_rid = _rd.storage_buffer_create(
				_forecast_output_capacity * BYTES_PER_VEC4
		);
		recreated = true;

	if (
			recreated
			or not _forecast_uniform_set_rid.is_valid()
			or (_tape_sample_out_rid.is_valid() and not _fused_forecast_uniform_set_rid.is_valid())
	):
		_rebuild_forecast_uniform_set();


## Rebuilds [member _forecast_uniform_set_rid] and [member _fused_forecast_uniform_set_rid].
func _rebuild_forecast_uniform_set() -> void:
	_free_forecast_uniform_set();

	var u_bodies := RDUniform.new();
	u_bodies.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_bodies.binding = 0;
	u_bodies.add_id(_forecast_body_buffer_rid);

	var u_timeline := RDUniform.new();
	u_timeline.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_timeline.binding = 1;
	u_timeline.add_id(_forecast_timeline_buffer_rid);

	var u_output := RDUniform.new();
	u_output.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_output.binding = 2;
	u_output.add_id(_forecast_output_buffer_rid);

	var u_atlas := RDUniform.new();
	u_atlas.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	u_atlas.binding = 3;
	u_atlas.add_id(_tape_atlas_buffer_rid);

	_forecast_uniform_set_rid = _rd.uniform_set_create(
			[u_bodies, u_timeline, u_output, u_atlas],
			_forecast_shader_rid,
			0
	);

	if _tape_sample_out_rid.is_valid():
		var u_fused_out := RDUniform.new();
		u_fused_out.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
		u_fused_out.binding = 2;
		u_fused_out.add_id(_tape_sample_out_rid);

		_fused_forecast_uniform_set_rid = _rd.uniform_set_create(
				[u_bodies, u_timeline, u_fused_out, u_atlas],
				_forecast_shader_rid,
				0
		);


## Safely frees [member _forecast_uniform_set_rid] and [member _fused_forecast_uniform_set_rid].
func _free_forecast_uniform_set() -> void:
	if _rd != null and _forecast_uniform_set_rid.is_valid():
		_rd.free_rid(_forecast_uniform_set_rid);
	_forecast_uniform_set_rid = RID();
	if _rd != null and _fused_forecast_uniform_set_rid.is_valid():
		_rd.free_rid(_fused_forecast_uniform_set_rid);
	_fused_forecast_uniform_set_rid = RID();


## Dispatches [code]ballistic_forecast.glsl[/code] on [member _rd] to integrate future
## anchor-relative trajectories for all [param bodies] in parallel.
func _compute_forecasts_gpu(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		timeline_bytes: PackedByteArray,
		inv_timeline_dt: float
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	if _telemetry != null:
		return _compute_forecasts_gpu_instrumented(
				bodies,
				well_list,
				timeline_bytes,
				inv_timeline_dt
		);

	var body_count: int = bodies.size();
	var well_count: int = well_list.size();
	var offsets: PackedInt32Array = _compute_forecast_output_offsets(bodies);
	var total_knots: int = offsets[body_count];
	var total_vec4s: int = total_knots * 2;
	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_bytes.size() / BYTES_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_vec4s);

	var body_bytes: PackedByteArray = (
			_pack_forecast_bodies_buffer(bodies, well_list, offsets).to_byte_array()
	);
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);

	if not timeline_bytes.is_empty():
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, timeline_bytes.size(), timeline_bytes);
		_tl_gpu_dirty = true;

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

	_rd.submit();
	_rd.sync();

	var out_bytes: PackedByteArray = _rd.buffer_get_data(
			_forecast_output_buffer_rid,
			0,
			total_vec4s * BYTES_PER_VEC4
	);
	return _unpack_forecast_output(bodies, offsets, out_bytes.to_vector4_array());


## Executes [method _compute_forecasts_gpu] while recording microsecond sub-phase timings into
## [member _telemetry].
func _compute_forecasts_gpu_instrumented(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		timeline_bytes: PackedByteArray,
		inv_timeline_dt: float
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var body_count: int = bodies.size();
	var well_count: int = well_list.size();

	var t0: int = Time.get_ticks_usec();
	var offsets: PackedInt32Array = _compute_forecast_output_offsets(bodies);
	var total_knots: int = offsets[body_count];
	var total_vec4s: int = total_knots * 2;
	var body_bytes: PackedByteArray = (
			_pack_forecast_bodies_buffer(bodies, well_list, offsets).to_byte_array()
	);
	var t1: int = Time.get_ticks_usec();

	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_bytes.size() / BYTES_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_vec4s);
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if not timeline_bytes.is_empty():
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, timeline_bytes.size(), timeline_bytes);
		_tl_gpu_dirty = true;

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
			total_vec4s * BYTES_PER_VEC4
	);
	var unpacked: Dictionary[JrBallisticOrbiter, PackedVector4Array] = _unpack_forecast_output(
			bodies,
			offsets,
			out_bytes.to_vector4_array()
	);
	var t4: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_PACK, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH, t2 - t1);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_SUBMIT_SYNC, t3 - t2);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_READBACK_UNPACK, t4 - t3);
	return unpacked;


## Dispatches [code]ballistic_forecast.glsl[/code] on [member _rd] to bake or append Cubic Hermite
## trajectory knots into each job's [JrBallisticTrajectoryTape].
func _bake_tapes_gpu(
		config: JrBallisticForecastConfig,
		well_count: int,
		timeline_bytes: PackedByteArray,
		inv_timeline_dt: float
) -> void:
	if _telemetry != null:
		_bake_tapes_gpu_instrumented(config, well_count, timeline_bytes, inv_timeline_dt);
		return;

	var body_count: int = config.bodies.size();
	_meta_precommitted_in_fused = false;
	_job_out_base_knot_offset = 0;
	_prepare_tape_atlas_for_gpu_bake(config);
	var offsets: PackedInt32Array = _plan_gpu_tape_job_layout(config);
	var total_vec4s: int = offsets[body_count] * 2;
	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_bytes.size() / BYTES_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_vec4s);
	var body_bytes: PackedByteArray = _pack_tape_jobs_buffer(config, offsets, well_count);
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if _tl_gpu_dirty and not timeline_bytes.is_empty():
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, timeline_bytes.size(), timeline_bytes);
		_tl_gpu_dirty = false;

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

	_rd.submit();
	_rd.sync();

	var out_bytes: PackedByteArray = _rd.buffer_get_data(
			_forecast_output_buffer_rid,
			0,
			total_vec4s * BYTES_PER_VEC4
	);
	_apply_unpacked_tape_slices(config, offsets, out_bytes.to_vector4_array());


## Executes [method _bake_tapes_gpu] while recording microsecond sub-phase timings into
## [member _telemetry].
func _bake_tapes_gpu_instrumented(
		config: JrBallisticForecastConfig,
		well_count: int,
		timeline_bytes: PackedByteArray,
		inv_timeline_dt: float
) -> void:
	var body_count: int = config.bodies.size();

	var t0: int = Time.get_ticks_usec();
	_meta_precommitted_in_fused = false;
	_job_out_base_knot_offset = 0;
	_prepare_tape_atlas_for_gpu_bake(config);
	var offsets: PackedInt32Array = _plan_gpu_tape_job_layout(config);
	var total_vec4s: int = offsets[body_count] * 2;
	var body_bytes: PackedByteArray = _pack_tape_jobs_buffer(config, offsets, well_count);
	var t1: int = Time.get_ticks_usec();

	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_bytes.size() / BYTES_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	_ensure_forecast_capacities(body_count, timeline_vec4_entries, total_vec4s);
	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if _tl_gpu_dirty and not timeline_bytes.is_empty():
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, timeline_bytes.size(), timeline_bytes);
		_tl_gpu_dirty = false;

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
			total_vec4s * BYTES_PER_VEC4
	);
	_apply_unpacked_tape_slices(config, offsets, out_bytes.to_vector4_array());
	var t4: int = Time.get_ticks_usec();

	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_PACK, t1 - t0);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH, t2 - t1);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_SUBMIT_SYNC, t3 - t2);
	_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_READBACK_UNPACK, t4 - t3);


## Returns `true` when every job in [param config] either writes directly to `TapeAtlasBuffer`
## or exceeds `GPU_TAPE_SLOT_KNOTS` (delegating to sparse CPU fallback without atlas upload).
func _can_all_jobs_precommit_atlas(config: JrBallisticForecastConfig) -> bool:
	var count: int = config.bodies.size();
	var total_slots: int = config.total_driver_bodies;
	if total_slots < MIN_GPU_TAPE_ATLAS_BODIES:
		return false;

	var has_modes: bool = not config.job_append_mode.is_empty();
	var has_steps: bool = not config.job_step_counts.is_empty();
	for i: int in range(count):
		var body: JrBallisticOrbiter = config.bodies[i];
		var slot_idx: int = body._ballistic_seq_index;
		if slot_idx < 0 or slot_idx >= total_slots:
			return false;
		var steps: int = maxi(1, config.job_step_counts[i] if has_steps else body.preview_steps);
		if (steps + 1) > GPU_TAPE_SLOT_KNOTS:
			continue;
		var mode: int = config.job_append_mode[i] if has_modes else JrBallisticForecastConfig.JOB_RESET;
		var tail_tol: float = maxf(1e-7, 1e-4 * body._trajectory_tape.knot_dt);
		if (
				mode == JrBallisticForecastConfig.JOB_APPEND_TAIL
				and body._trajectory_tape.is_valid
				and config.trim_before_time < body._trajectory_tape.get_end_time() - tail_tol
		):
			return false;
	return true;


## Records `ballistic_forecast.glsl` -> memory barrier -> `ballistic_tape_sample.glsl` in a single
## compute list and performs a single unified SSBO readback (`timing_params = Vector3(inv_timeline_dt, well_count, eval_dt)`).
func _execute_fused_bake_and_sample_gpu(
		config: JrBallisticForecastConfig,
		all_bodies: Array[JrBallisticOrbiter],
		timeline_bytes: PackedByteArray,
		timing_params: Vector3
) -> PackedVector4Array:
	var record_tel: bool = _telemetry != null;
	var t0: int = Time.get_ticks_usec() if record_tel else 0;

	var inv_timeline_dt: float = timing_params.x;
	var well_count: int = int(roundf(timing_params.y));
	var eval_time: float = _fused_eval_time;
	var total_count: int = all_bodies.size();
	var job_count: int = config.bodies.size();

	_ensure_tape_atlas_capacities(total_count);
	if not _tape_atlas_valid or _tape_meta_body_count != total_count:
		if job_count < total_count:
			_sync_all_tapes_to_gpu_atlas(all_bodies, config.start_time);
		else:
			_tape_gpu_time_origin = config.start_time;
			_tape_meta_body_count = total_count;
			_tape_meta_vec4s.resize(total_count * 2);
			_tape_meta_vec4s.fill(Vector4.ZERO);
			_mark_tape_meta_slot_dirty(0);
			_mark_tape_meta_slot_dirty(maxi(0, total_count - 1));
	elif absf(eval_time - _tape_gpu_time_origin) > MAX_GPU_TIME_ORIGIN_SPAN:
		_rebase_tape_meta_time_origin(all_bodies, eval_time);

	var pos_end_bytes: int = STEP_OUT_HEADER_BYTES + total_count * TAPE_SAMPLE_POS_BYTES_PER_BODY;
	var tail_base_knot: int = int(ceili(float(pos_end_bytes) / 32.0));

	_meta_precommitted_in_fused = true;
	var offsets: PackedInt32Array = _plan_gpu_tape_job_layout(config);
	var bake_knots: int = offsets[job_count];
	_job_out_base_knot_offset = tail_base_knot;
	var body_bytes: PackedByteArray = _pack_tape_jobs_buffer(config, offsets, well_count);
	_job_out_base_knot_offset = 0;
	_tape_atlas_valid = true;
	var t1: int = Time.get_ticks_usec() if record_tel else 0;

	@warning_ignore("integer_division")
	var timeline_vec4_entries: int = timeline_bytes.size() / BYTES_PER_VEC4;
	@warning_ignore("integer_division")
	var timeline_segments: int = (
			maxi(1, (timeline_vec4_entries / maxi(1, well_count * 2)) - 1)
			if well_count > 0
			else 1
	);

	var events_word_offset: int = (tail_base_knot + bake_knots) * 8;
	var required_sample_out_bytes: int = (
			(tail_base_knot + bake_knots) * 32
			+ total_count * TAPE_SAMPLE_EVENT_BYTES
	);
	_ensure_tape_sample_out_bytes(required_sample_out_bytes);
	_ensure_forecast_capacities(job_count, timeline_vec4_entries, 2);

	_rd.buffer_update(_forecast_body_buffer_rid, 0, body_bytes.size(), body_bytes);
	if _tl_gpu_dirty and not timeline_bytes.is_empty():
		_rd.buffer_update(_forecast_timeline_buffer_rid, 0, timeline_bytes.size(), timeline_bytes);
		_tl_gpu_dirty = false;

	_flush_dirty_tape_meta_to_gpu();

	var bake_push: PackedByteArray = _build_forecast_push_constants(
			inv_timeline_dt,
			job_count,
			well_count,
			timeline_segments
	);
	var rel_pair: Vector2 = _split_float64_to_f32_pair(eval_time - _tape_gpu_time_origin);
	var sample_push := PackedByteArray();
	sample_push.resize(PUSH_CONSTANT_BYTES);
	sample_push.encode_float(0, rel_pair.x);
	sample_push.encode_float(4, rel_pair.y);
	sample_push.encode_u32(8, total_count);
	sample_push.encode_u32(12, events_word_offset);

	var bake_groups: int = int(ceili(float(job_count) / float(WORKGROUP_SIZE_X)));
	var sample_groups: int = int(ceili(float(total_count) / float(WORKGROUP_SIZE_X)));

	var cl: int = _rd.compute_list_begin();
	_rd.compute_list_bind_compute_pipeline(cl, _forecast_pipeline_rid);
	_rd.compute_list_bind_uniform_set(cl, _fused_forecast_uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(cl, bake_push, PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(cl, bake_groups, 1, 1);
	_rd.compute_list_add_barrier(cl);
	_rd.compute_list_bind_compute_pipeline(cl, _tape_sample_pipeline_rid);
	_rd.compute_list_bind_uniform_set(cl, _tape_sample_uniform_set_rid, 0);
	_rd.compute_list_set_push_constant(cl, sample_push, PUSH_CONSTANT_BYTES);
	_rd.compute_list_dispatch(cl, sample_groups, 1, 1);
	_rd.compute_list_end();
	var t2: int = Time.get_ticks_usec() if record_tel else 0;

	_rd.submit();
	var t3: int = Time.get_ticks_usec() if record_tel else 0;

	if record_tel:
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_PACK, t1 - t0);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_UPLOAD_DISPATCH, t2 - t1);
		_telemetry.add_phase_usec(JrOrreryTelemetry.PHASE_SOLVER_FORECAST_SUBMIT_SYNC, t3 - t2);

	_last_fused_bake_sampled = true;
	_pending_gpu_sample_sync = true;
	_pending_gpu_sample_config = config;
	_pending_gpu_sample_offsets = offsets;
	_pending_gpu_sample_tail_base = tail_base_knot;
	return PackedVector4Array();


## Computes prefix-sum output start offsets (and total point count at index
## [code]bodies.size()[/code]).
func _compute_forecast_output_offsets(bodies: Array[JrBallisticOrbiter]) -> PackedInt32Array:
	var count: int = bodies.size();
	var offsets: PackedInt32Array = [];
	offsets.resize(count + 1);
	var running_total: int = 0;
	for i: int in range(count):
		offsets[i] = running_total;
		running_total += maxi(1, bodies[i].preview_steps) + 1;
	offsets[count] = running_total;
	return offsets;


## Plans per-job VRAM atlas write flags, compact 2-`vec4` tail-summary flags, and output knot offsets.
func _plan_gpu_tape_job_layout(config: JrBallisticForecastConfig) -> PackedInt32Array:
	var count: int = config.bodies.size();
	var offsets := PackedInt32Array();
	offsets.resize(count + 1);
	_job_wrote_atlas_flags.resize(count);
	_job_wrote_atlas_flags.fill(0);
	_job_compact_out_flags.resize(count);
	_job_compact_out_flags.fill(0);

	var use_gpu_atlas: bool = (
			_tape_sample_gpu_available
			and config.total_driver_bodies >= MIN_GPU_TAPE_ATLAS_BODIES
	);
	var has_steps: bool = not config.job_step_counts.is_empty();
	var has_modes: bool = not config.job_append_mode.is_empty();
	var running_total: int = 0;

	for i: int in range(count):
		offsets[i] = running_total;
		var body: JrBallisticOrbiter = config.bodies[i];
		var steps: int = (
				maxi(1, config.job_step_counts[i])
				if has_steps
				else maxi(1, body.preview_steps)
		);
		var pt_count: int = steps + 1;
		var slot_idx: int = body._ballistic_seq_index;
		var mode: int = config.job_append_mode[i] if has_modes else JrBallisticForecastConfig.JOB_RESET;
		var tail_tol: float = maxf(1e-7, 1e-4 * body._trajectory_tape.knot_dt);

		if (
				use_gpu_atlas
				and slot_idx >= 0
				and slot_idx < config.total_driver_bodies
				and pt_count <= GPU_TAPE_SLOT_KNOTS
				and (
						mode != JrBallisticForecastConfig.JOB_APPEND_TAIL
						or not body._trajectory_tape.is_valid
						or config.trim_before_time >= body._trajectory_tape.get_end_time() - tail_tol
				)
		):
			_job_wrote_atlas_flags[i] = 1;
			if (
					not (config.enable_forecasts and body.runtime_forecast_enabled)
					and (
							mode != JrBallisticForecastConfig.JOB_PERIODIC_LOOP
							or config.total_driver_bodies >= MIN_GPU_TAPE_SAMPLE_BODIES
					)
			):
				_job_compact_out_flags[i] = 1;
				running_total += 1;
				continue;

		running_total += pt_count;

	offsets[count] = running_total;
	return offsets;


## Resolves the maximum timeline span (`t1 - start_time`) required across all jobs in [param config].
func _resolve_tape_batch_duration(config: JrBallisticForecastConfig) -> float:
	if config.timeline_end_time > config.start_time:
		return config.timeline_end_time - config.start_time;

	var max_dur: float = 0.0;
	var count: int = config.bodies.size();
	var has_t0: bool = not config.job_start_times.is_empty();
	var has_steps: bool = not config.job_step_counts.is_empty();
	var has_dts: bool = not config.job_knot_dts.is_empty();

	for i: int in range(count):
		var body: JrBallisticOrbiter = config.bodies[i];
		var steps: int = maxi(1, config.job_step_counts[i] if has_steps else body.preview_steps);
		var dt_i: float = (
				config.job_knot_dts[i]
				if has_dts
				else (maxf(0.0, body.preview_duration) / float(steps))
		);
		var offset_t0: float = (
				maxf(0.0, config.job_start_times[i] - config.start_time)
				if has_t0
				else 0.0
		);
		max_dur = maxf(max_dur, offset_t0 + float(steps) * dt_i);
	return max_dur;


## Computes an adaptive gravity-well timeline sample count bounded by orbital angular step and well count.
func _compute_adaptive_timeline_segments(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement],
		max_duration: float
) -> int:
	var well_count: int = well_list.size();
	if well_count <= 1 or max_duration <= 1e-9:
		return 1;

	var max_mean_n: float = 0.0;
	for w: int in range(1, well_count):
		var orb := well_list[w] as JrRailOrbiter;
		if orb != null:
			max_mean_n = maxf(max_mean_n, orb.mean_motion);

	var arc_rad: float = max_mean_n * max_duration;
	var max_step_rad: float = (
			0.03
			if (config.enable_forecasts or config.job_step_counts.is_empty())
			else 0.05
	);
	var needed: int = int(ceili(arc_rad / max_step_rad));
	var min_segs: int = maxi(8, config.timeline_segments) if config.timeline_segments > 0 else 8;
	var max_cap: int = clampi(65536 / maxi(1, well_count), 32, 1024);
	return clampi(maxi(min_segs, needed), 8, max_cap);


## Packs anchor-relative initial states, step intervals, step counts, thrust vectors,
## and anchor metadata for [code]ForecastBodyInputBuffer[/code].
func _pack_forecast_bodies_buffer(
		bodies: Array[JrBallisticOrbiter],
		well_list: Array[JrOrreryElement],
		offsets: PackedInt32Array
) -> PackedFloat32Array:
	var count: int = bodies.size();
	var well_count: int = well_list.size();
	var floats := PackedFloat32Array();
	floats.resize(count * FLOATS_PER_BODY);

	for i: int in range(count):
		var body: JrBallisticOrbiter = bodies[i];
		var pos: Vector3 = body.get_anchor_relative_position();
		var vel: Vector3 = body.get_anchor_relative_velocity();
		var steps: int = maxi(1, body.preview_steps);
		var step_dt: float = maxf(0.0, body.preview_duration) / float(steps);
		var ext: Vector3 = body.get_scaled_external_acceleration();
		var anchor_elem: JrOrreryElement = body.get_anchor_element();
		var anchor_idx: int = (
				anchor_elem._runtime_well_index
				if (
						anchor_elem != null
						and anchor_elem._runtime_well_index >= 0
						and anchor_elem._runtime_well_index < well_count
						and well_list[anchor_elem._runtime_well_index] == anchor_elem
				)
				else 0
		);
		var mode_bits: int = 0;
		if body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL:
			mode_bits |= 1;
		if body.enable_tidal_forces:
			mode_bits |= 2;
		if body.pinned_anchor != null:
			mode_bits |= 4;
		var base: int = i * FLOATS_PER_BODY;

		floats[base] = pos.x;
		floats[base + 1] = pos.y;
		floats[base + 2] = pos.z;
		floats[base + 3] = step_dt;

		floats[base + 4] = vel.x;
		floats[base + 5] = vel.y;
		floats[base + 6] = vel.z;
		floats[base + 7] = float(steps);

		floats[base + 8] = ext.x;
		floats[base + 9] = ext.y;
		floats[base + 10] = ext.z;
		floats[base + 11] = 0.0;

		floats[base + 12] = float(offsets[i]);
		floats[base + 13] = float(anchor_idx);
		floats[base + 14] = float(mode_bits);
		floats[base + 15] = float(FORECAST_INTERNAL_SUBSTEPS);

	return floats;


## Packs mixed full-horizon and tail-extension jobs from [param config] into `ForecastBodyInput[]`.
func _pack_tape_jobs_buffer(
		config: JrBallisticForecastConfig,
		offsets: PackedInt32Array,
		well_count: int
) -> PackedByteArray:
	var count: int = config.bodies.size();
	var vec4s := PackedVector4Array();
	vec4s.resize(count * VEC4S_PER_BODY);

	var has_modes: bool = not config.job_append_mode.is_empty();
	var has_t0s: bool = not config.job_start_times.is_empty();
	var has_dts: bool = not config.job_knot_dts.is_empty();
	var has_steps: bool = not config.job_step_counts.is_empty();
	var has_subs: bool = not config.job_substeps.is_empty();

	for i: int in range(count):
		var body: JrBallisticOrbiter = config.bodies[i];
		var tape: JrBallisticTrajectoryTape = body._trajectory_tape;
		var mode: int = config.job_append_mode[i] if has_modes else JrBallisticForecastConfig.JOB_RESET;
		var job_t0: float = config.job_start_times[i] if has_t0s else config.start_time;
		var pos: Vector3;
		var vel: Vector3;
		var anchor_idx: int;
		if mode == JrBallisticForecastConfig.JOB_APPEND_TAIL and tape.is_valid:
			if body.runtime_forecast_enabled and tape.epoch_unperturbed and not body._has_external_thrust:
				anchor_idx = tape.epoch_well_idx;
				var mu_ep: float = (
						_tl_well_mu[anchor_idx]
						if (anchor_idx >= 0 and anchor_idx < well_count and anchor_idx < _tl_well_mu.size())
						else 0.0
				);
				if mu_ep > 0.0:
					_kepler_drift_into(tape.epoch_pos, tape.epoch_vel, mu_ep, job_t0 - tape.epoch_time);
					pos = _drift_r;
					vel = _drift_v;
				else:
					pos = tape.get_tail_position();
					vel = tape.get_tail_velocity();
			else:
				var tp: Vector4 = tape.tail_pos_anchor;
				var tv: Vector4 = tape.tail_vel_flags;
				pos = Vector3(tp.x, tp.y, tp.z);
				vel = Vector3(tv.x, tv.y, tv.z);
				anchor_idx = int(roundf(tp.w));
		else:
			if body.is_tape_position_pending():
				body.ensure_tape_position();
			if body.is_tape_velocity_pending():
				body.ensure_tape_velocity();
			pos = body._anchor_relative_position;
			vel = body._anchor_relative_velocity;
			anchor_idx = body.get_anchor_well_index();
			if (
					body._tape_anchor_idx < 0
					and body._anchor_element != null
					and body._anchor_element._runtime_well_index == anchor_idx
			):
				body._tape_anchor_idx = anchor_idx;

		var steps: int = maxi(1, config.job_step_counts[i] if has_steps else body.preview_steps);
		var step_dt: float = (
				config.job_knot_dts[i]
				if has_dts
				else (maxf(0.0, body.preview_duration) / float(steps))
		);
		var start_offset: float = maxf(0.0, job_t0 - config.start_time);
		var substeps: int = clampi(
				config.job_substeps[i] if has_subs else FORECAST_INTERNAL_SUBSTEPS,
				1,
				64
		);
		var ext: Vector3 = (
				body.get_scaled_external_acceleration()
				if body._has_external_thrust
				else Vector3.ZERO
		);

		var mode_bits: int = 0;
		if body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL:
			mode_bits |= 1;
		if body.enable_tidal_forces:
			mode_bits |= 2;
		if body.pinned_anchor != null:
			mode_bits |= 4;
		var slot_idx: int = body._ballistic_seq_index;
		if i < _job_compact_out_flags.size() and _job_compact_out_flags[i] != 0:
			mode_bits |= 8;
		if i < _job_wrote_atlas_flags.size() and _job_wrote_atlas_flags[i] != 0:
			mode_bits |= ((slot_idx + 1) << 4);

		if _meta_precommitted_in_fused:
			_precommit_job_tape_meta(config, i, step_dt, steps);

		var base: int = i * VEC4S_PER_BODY;
		vec4s[base] = Vector4(pos.x, pos.y, pos.z, step_dt);
		vec4s[base + 1] = Vector4(vel.x, vel.y, vel.z, float(steps));
		vec4s[base + 2] = Vector4(ext.x, ext.y, ext.z, start_offset);
		vec4s[base + 3] = Vector4(
				float(offsets[i] + _job_out_base_knot_offset),
				float(anchor_idx),
				float(mode_bits),
				float(substeps)
		);

	return vec4s.to_byte_array();


## Pre-updates `TapeMetaBuffer` slot metadata for job [param idx] prior to a fused GPU bake+sample dispatch.
func _precommit_job_tape_meta(
		config: JrBallisticForecastConfig,
		idx: int,
		step_dt: float,
		steps: int
) -> void:
	var body: JrBallisticOrbiter = config.bodies[idx];
	var slot_idx: int = body._ballistic_seq_index;
	var base: int = slot_idx * 2;
	if base < 0 or base + 1 >= _tape_meta_vec4s.size():
		return;

	var knot_count: int = steps + 1;
	if knot_count < 2 or knot_count > GPU_TAPE_SLOT_KNOTS:
		_tape_meta_vec4s[base] = Vector4.ZERO;
		_tape_meta_vec4s[base + 1] = Vector4(float(slot_idx), 0.0, 0.0, float(body._tape_anchor_idx));
		_mark_tape_meta_slot_dirty(slot_idx);
		return;

	var tape: JrBallisticTrajectoryTape = body._trajectory_tape;
	var mode: int = (
			config.job_append_mode[idx]
			if idx < config.job_append_mode.size()
			else JrBallisticForecastConfig.JOB_RESET
	);
	var new_t0: float;
	var eff_dt: float;
	var loop_p: float = 0.0;
	if mode == JrBallisticForecastConfig.JOB_APPEND_TAIL and tape.is_valid:
		new_t0 = tape.get_end_time();
		eff_dt = tape.knot_dt;
	else:
		new_t0 = (
				config.job_start_times[idx]
				if idx < config.job_start_times.size()
				else config.start_time
		);
		eff_dt = maxf(1e-7, step_dt);
		if (
				mode == JrBallisticForecastConfig.JOB_PERIODIC_LOOP
				and idx < config.job_loop_periods.size()
				and config.job_loop_periods[idx] > 0.0
		):
			loop_p = config.job_loop_periods[idx];

	var t0_pair: Vector2 = _split_float64_to_f32_pair(new_t0 - _tape_gpu_time_origin);
	_tape_meta_vec4s[base] = Vector4(t0_pair.x, t0_pair.y, 1.0 / eff_dt, eff_dt);
	_tape_meta_vec4s[base + 1] = Vector4(
			float(slot_idx),
			float(knot_count),
			loop_p,
			float(body._tape_anchor_idx)
	);
	_mark_tape_meta_slot_dirty(slot_idx);


## Slices contiguous `pos_anchor` arrays out of [param all_vec4s] in native C++.
func _unpack_forecast_output(
		bodies: Array[JrBallisticOrbiter],
		offsets: PackedInt32Array,
		all_vec4s: PackedVector4Array
) -> Dictionary[JrBallisticOrbiter, PackedVector4Array]:
	var results: Dictionary[JrBallisticOrbiter, PackedVector4Array] = {};
	for i: int in range(bodies.size()):
		var start_knot: int = offsets[i];
		var pt_count: int = offsets[i + 1] - start_knot;
		var pos_base: int = start_knot * 2;
		results[bodies[i]] = all_vec4s.slice(pos_base, pos_base + pt_count);
	return results;


## Slices contiguous `[pos_anchor | vel_flags]` spans (or 2-`vec4` compact tail summaries) out of
## [param all_vec4s] and commits them into each body's [JrBallisticTrajectoryTape].
func _apply_unpacked_tape_slices(
		config: JrBallisticForecastConfig,
		offsets: PackedInt32Array,
		all_vec4s: PackedVector4Array
) -> void:
	if _self_weak_ref == null:
		_self_weak_ref = weakref(self);

	var count: int = config.bodies.size();
	var has_compact: bool = _job_compact_out_flags.size() == count;
	for i: int in range(count):
		var start_knot: int = offsets[i];
		var pos_base: int = start_knot * 2;
		if has_compact and _job_compact_out_flags[i] != 0:
			_commit_body_compact_atlas_job(
					config,
					i,
					all_vec4s[pos_base],
					all_vec4s[pos_base + 1]
			);
			continue;

		var pt_count: int = offsets[i + 1] - start_knot;
		var vel_base: int = pos_base + pt_count;
		var pos_slice: PackedVector4Array = all_vec4s.slice(pos_base, vel_base);
		var vel_slice: PackedVector4Array = all_vec4s.slice(vel_base, vel_base + pt_count);
		_commit_body_tape_job(config, i, pos_slice, vel_slice);

	_finalize_tape_atlas_after_bake(config, true);


## Commits a compact 2-`vec4` GPU-atlas tail summary into `config.bodies[idx]` without CPU knot slicing.
func _commit_body_compact_atlas_job(
		config: JrBallisticForecastConfig,
		idx: int,
		tail_pos: Vector4,
		tail_vel: Vector4
) -> void:
	var body: JrBallisticOrbiter = config.bodies[idx];
	var tape: JrBallisticTrajectoryTape = body._trajectory_tape;
	var mode: int = (
			config.job_append_mode[idx]
			if idx < config.job_append_mode.size()
			else JrBallisticForecastConfig.JOB_RESET
	);
	var steps: int = maxi(
			1,
			config.job_step_counts[idx] if idx < config.job_step_counts.size() else body.preview_steps
	);
	var slot_idx: int = body._ballistic_seq_index;

	if mode == JrBallisticForecastConfig.JOB_APPEND_TAIL and tape.is_valid:
		var prev_unpert: bool = tape.epoch_unperturbed;
		var new_t0: float = tape.get_end_time();
		tape.reset_gpu_atlas_grid(new_t0, tape.knot_dt, steps + 1);
		tape.bind_gpu_atlas_tail(tail_pos, tail_vel, slot_idx, _self_weak_ref);
		tape.epoch_unperturbed = prev_unpert and (tail_vel.w > 0.5);
	else:
		var knot_dt: float = (
				config.job_knot_dts[idx]
				if idx < config.job_knot_dts.size()
				else (maxf(1e-6, body.preview_duration) / float(steps))
		);
		var job_t0: float = (
				config.job_start_times[idx]
				if idx < config.job_start_times.size()
				else config.start_time
		);
		if body.is_tape_position_pending():
			body.ensure_tape_position();
		if body.is_tape_velocity_pending():
			body.ensure_tape_velocity();
		tape.reset_gpu_atlas_grid(job_t0, knot_dt, steps + 1);
		tape.bind_gpu_atlas_tail(tail_pos, tail_vel, slot_idx, _self_weak_ref);
		tape.epoch_pos = body._anchor_relative_position;
		tape.epoch_vel = body._anchor_relative_velocity;
		tape.epoch_time = job_t0;
		tape.epoch_well_idx = body.get_anchor_well_index();
		if (
				body._tape_anchor_idx < 0
				and body._anchor_element != null
				and body._anchor_element._runtime_well_index == tape.epoch_well_idx
		):
			body._tape_anchor_idx = tape.epoch_well_idx;
		tape.epoch_unperturbed = tail_vel.w > 0.5;
		if mode == JrBallisticForecastConfig.JOB_PERIODIC_LOOP and idx < config.job_loop_periods.size():
			tape.mark_periodic_loop(config.job_loop_periods[idx]);

	body._is_on_rails = tape.epoch_unperturbed;
	if not _meta_precommitted_in_fused and slot_idx >= 0 and slot_idx < config.total_driver_bodies:
		_write_tape_meta_entry(slot_idx, body, tape);


## Commits a newly integrated `(pos_slice, vel_slice)` job output into `config.bodies[idx]`.
func _commit_body_tape_job(
		config: JrBallisticForecastConfig,
		idx: int,
		pos_slice: PackedVector4Array,
		vel_slice: PackedVector4Array
) -> void:
	var body: JrBallisticOrbiter = config.bodies[idx];
	var tape: JrBallisticTrajectoryTape = body._trajectory_tape;
	var mode: int = (
			config.job_append_mode[idx]
			if idx < config.job_append_mode.size()
			else JrBallisticForecastConfig.JOB_RESET
	);

	if mode == JrBallisticForecastConfig.JOB_APPEND_TAIL and tape.is_valid:
		tape.append_tail_knots(pos_slice, vel_slice, config.trim_before_time);
		tape.epoch_unperturbed = tape.epoch_unperturbed and tape.was_completely_unperturbed();
	else:
		var steps: int = maxi(1, pos_slice.size() - 1);
		var knot_dt: float = (
				config.job_knot_dts[idx]
				if idx < config.job_knot_dts.size()
				else (maxf(1e-6, body.preview_duration) / float(steps))
		);
		var job_t0: float = (
				config.job_start_times[idx]
				if idx < config.job_start_times.size()
				else config.start_time
		);
		if body.is_tape_position_pending():
			body.ensure_tape_position();
		if body.is_tape_velocity_pending():
			body.ensure_tape_velocity();
		tape.reset_with_knots(job_t0, knot_dt, pos_slice, vel_slice);
		tape.epoch_pos = body._anchor_relative_position;
		tape.epoch_vel = body._anchor_relative_velocity;
		tape.epoch_time = job_t0;
		tape.epoch_well_idx = body.get_anchor_well_index();
		if (
				body._tape_anchor_idx < 0
				and body._anchor_element != null
				and body._anchor_element._runtime_well_index == tape.epoch_well_idx
		):
			body._tape_anchor_idx = tape.epoch_well_idx;
		tape.epoch_unperturbed = tape.was_completely_unperturbed();
		if mode == JrBallisticForecastConfig.JOB_PERIODIC_LOOP and idx < config.job_loop_periods.size():
			tape.mark_periodic_loop(config.job_loop_periods[idx]);

	body._is_on_rails = tape.epoch_unperturbed;

	if _tape_sample_gpu_available and config.total_driver_bodies >= MIN_GPU_TAPE_ATLAS_BODIES:
		var slot_idx: int = body._ballistic_seq_index;
		var wrote_direct: bool = idx < _job_wrote_atlas_flags.size() and _job_wrote_atlas_flags[idx] != 0;
		if slot_idx >= 0 and slot_idx < config.total_driver_bodies:
			if not _meta_precommitted_in_fused:
				_write_tape_meta_entry(slot_idx, body, tape);
			if (
					(_tape_atlas_valid or config.bodies.size() == config.total_driver_bodies)
					and not wrote_direct
					and tape.is_valid
					and tape.knot_count <= GPU_TAPE_SLOT_KNOTS
			):
				_upload_single_tape_to_atlas(slot_idx, tape);


## Prepares resident GPU trajectory-tape atlas and metadata storage before a GPU bake pass.
func _prepare_tape_atlas_for_gpu_bake(config: JrBallisticForecastConfig) -> void:
	if not _tape_sample_gpu_available or config.total_driver_bodies < MIN_GPU_TAPE_ATLAS_BODIES:
		_job_wrote_atlas_flags.clear();
		_job_compact_out_flags.clear();
		return;

	var total_slots: int = config.total_driver_bodies;
	_ensure_tape_atlas_capacities(total_slots);
	if _tape_meta_body_count != total_slots or _tape_meta_vec4s.size() != total_slots * 2:
		_tape_atlas_valid = false;
		_tape_meta_body_count = total_slots;
		_tape_gpu_time_origin = config.start_time;
		_tape_meta_vec4s.resize(total_slots * 2);
		_tape_meta_vec4s.fill(Vector4.ZERO);
		_mark_tape_meta_slot_dirty(0);
		_mark_tape_meta_slot_dirty(maxi(0, total_slots - 1));
	elif not _tape_atlas_valid and config.bodies.size() == total_slots:
		_tape_gpu_time_origin = config.start_time;


## Coalesces metadata uploads to `TapeMetaBuffer` once at the end of a trajectory-tape bake pass.
func _finalize_tape_atlas_after_bake(
		config: JrBallisticForecastConfig,
		is_gpu_bake: bool
) -> void:
	if not _tape_sample_gpu_available or config.total_driver_bodies < MIN_GPU_TAPE_ATLAS_BODIES:
		_tape_atlas_valid = false;
		return;

	if not is_gpu_bake and not _tape_atlas_valid:
		return;

	if is_gpu_bake and not _tape_atlas_valid and config.bodies.size() == config.total_driver_bodies:
		_tape_atlas_valid = true;

	if _tape_atlas_valid and _tape_meta_gpu_dirty:
		_flush_dirty_tape_meta_to_gpu();


## Marks [param slot_idx] dirty within [member _tape_meta_vec4s] for partial `TapeMetaBuffer` uploads.
func _mark_tape_meta_slot_dirty(slot_idx: int) -> void:
	if not _tape_meta_gpu_dirty:
		_tape_meta_gpu_dirty = true;
		_tape_meta_dirty_min_slot = slot_idx;
		_tape_meta_dirty_max_slot = slot_idx;
	else:
		_tape_meta_dirty_min_slot = mini(_tape_meta_dirty_min_slot, slot_idx);
		_tape_meta_dirty_max_slot = maxi(_tape_meta_dirty_max_slot, slot_idx);


## Uploads only the dirty `[min_slot .. max_slot]` byte span of [member _tape_meta_vec4s] to VRAM.
func _flush_dirty_tape_meta_to_gpu() -> void:
	if not _tape_meta_gpu_dirty or not _tape_meta_buffer_rid.is_valid():
		_tape_meta_gpu_dirty = false;
		_tape_meta_dirty_min_slot = 0;
		_tape_meta_dirty_max_slot = -1;
		return;

	var max_valid_slot: int = (_tape_meta_vec4s.size() >> 1) - 1;
	if max_valid_slot >= 0:
		var lo: int = clampi(_tape_meta_dirty_min_slot, 0, max_valid_slot);
		var hi: int = clampi(_tape_meta_dirty_max_slot, lo, max_valid_slot);
		if hi >= lo:
			var slice_bytes: PackedByteArray = _tape_meta_vec4s.slice(lo * 2, (hi + 1) * 2).to_byte_array();
			_rd.buffer_update(_tape_meta_buffer_rid, lo * 32, slice_bytes.size(), slice_bytes);

	_tape_meta_gpu_dirty = false;
	_tape_meta_dirty_min_slot = 0;
	_tape_meta_dirty_max_slot = -1;


## Writes `(m0, m1)` metadata for [param slot_idx] into [member _tape_meta_vec4s].
func _write_tape_meta_entry(
		slot_idx: int,
		body: JrBallisticOrbiter,
		tape: JrBallisticTrajectoryTape
) -> void:
	var base: int = slot_idx * 2;
	if base < 0 or base + 1 >= _tape_meta_vec4s.size():
		return;

	if not tape.is_valid or tape.knot_count < 2 or tape.knot_count > GPU_TAPE_SLOT_KNOTS:
		_tape_meta_vec4s[base] = Vector4.ZERO;
		_tape_meta_vec4s[base + 1] = Vector4(float(slot_idx), 0.0, 0.0, float(body._tape_anchor_idx));
		_mark_tape_meta_slot_dirty(slot_idx);
		return;

	var t0_pair: Vector2 = _split_float64_to_f32_pair(tape.start_time - _tape_gpu_time_origin);
	var loop_p: float = tape.loop_period if tape.is_periodic_loop else 0.0;
	_tape_meta_vec4s[base] = Vector4(t0_pair.x, t0_pair.y, tape.inv_knot_dt, tape.knot_dt);
	_tape_meta_vec4s[base + 1] = Vector4(
			float(slot_idx),
			float(tape.knot_count),
			loop_p,
			float(body._tape_anchor_idx)
	);
	_mark_tape_meta_slot_dirty(slot_idx);


## Uploads CPU-integrated or trimmed [param tape] knots into Knot-Major SoA `TapeAtlasBuffer` at [param slot_idx].
func _upload_single_tape_to_atlas(slot_idx: int, tape: JrBallisticTrajectoryTape) -> void:
	var stride: int = _tape_atlas_capacity;
	if not _tape_atlas_buffer_rid.is_valid() or slot_idx < 0 or slot_idx >= stride:
		return;
	if not tape.is_valid or tape.knot_count < 2 or tape.knot_count > GPU_TAPE_SLOT_KNOTS:
		return;
	if not tape.cpu_knots_resident:
		if _tape_atlas_valid and tape.atlas_slot_index == slot_idx:
			return;
		tape.ensure_cpu_knots();
		if not tape.cpu_knots_resident:
			return;

	var pos_bytes: PackedByteArray = tape.pos_anchor.to_byte_array();
	var vel_bytes: PackedByteArray = tape.vel_flags.to_byte_array();
	var k_count: int = mini(tape.knot_count, mini(tape.pos_anchor.size(), tape.vel_flags.size()));
	var vel_row_base: int = GPU_TAPE_SLOT_KNOTS * stride;
	for k: int in range(k_count):
		var src_off: int = k * BYTES_PER_VEC4;
		var p_slice: PackedByteArray = pos_bytes.slice(src_off, src_off + BYTES_PER_VEC4);
		var v_slice: PackedByteArray = vel_bytes.slice(src_off, src_off + BYTES_PER_VEC4);
		_rd.buffer_update(
				_tape_atlas_buffer_rid,
				(k * stride + slot_idx) * BYTES_PER_VEC4,
				BYTES_PER_VEC4,
				p_slice
		);
		_rd.buffer_update(
				_tape_atlas_buffer_rid,
				(vel_row_base + k * stride + slot_idx) * BYTES_PER_VEC4,
				BYTES_PER_VEC4,
				v_slice
		);


## Synchronizes all [param bodies] tapes and metadata into Knot-Major SoA `TapeAtlasBuffer` and `TapeMetaBuffer`.
func _sync_all_tapes_to_gpu_atlas(bodies: Array[JrBallisticOrbiter], eval_time: float) -> void:
	var count: int = bodies.size();
	_ensure_tape_atlas_capacities(count);
	var stride: int = _tape_atlas_capacity;
	_tape_gpu_time_origin = eval_time;
	_tape_meta_body_count = count;
	_tape_meta_vec4s.resize(count * 2);

	var any_cpu_knots: bool = false;
	for i: int in range(count):
		var t_chk: JrBallisticTrajectoryTape = bodies[i]._trajectory_tape;
		if (
				t_chk.is_valid
				and t_chk.knot_count >= 2
				and t_chk.knot_count <= GPU_TAPE_SLOT_KNOTS
				and t_chk.cpu_knots_resident
		):
			any_cpu_knots = true;
			break;

	var atlas_v4s := PackedVector4Array();
	var vel_row_base: int = GPU_TAPE_SLOT_KNOTS * stride;
	if any_cpu_knots:
		atlas_v4s.resize(GPU_TAPE_SLOT_VEC4S * stride);

	for i: int in range(count):
		var body: JrBallisticOrbiter = bodies[i];
		var tape: JrBallisticTrajectoryTape = body._trajectory_tape;
		if (
				tape.is_valid
				and body._tape_anchor_idx < 0
				and body._anchor_element != null
				and body._anchor_element._runtime_well_index == tape.epoch_well_idx
		):
			body._tape_anchor_idx = tape.epoch_well_idx;
		if (
				any_cpu_knots
				and tape.is_valid
				and tape.knot_count >= 2
				and tape.knot_count <= GPU_TAPE_SLOT_KNOTS
				and tape.cpu_knots_resident
		):
			var k_lim: int = mini(tape.knot_count, mini(tape.pos_anchor.size(), tape.vel_flags.size()));
			for k: int in range(k_lim):
				var off: int = k * stride + i;
				atlas_v4s[off] = tape.pos_anchor[k];
				atlas_v4s[vel_row_base + off] = tape.vel_flags[k];
			tape.atlas_slot_index = i;
		_write_tape_meta_entry(i, body, tape);

	if any_cpu_knots and _tape_atlas_buffer_rid.is_valid():
		var atlas_bytes: PackedByteArray = atlas_v4s.to_byte_array();
		_rd.buffer_update(_tape_atlas_buffer_rid, 0, atlas_bytes.size(), atlas_bytes);

	if count > 0 and _tape_meta_buffer_rid.is_valid():
		var meta_bytes: PackedByteArray = _tape_meta_vec4s.to_byte_array();
		_rd.buffer_update(_tape_meta_buffer_rid, 0, meta_bytes.size(), meta_bytes);
	_tape_meta_gpu_dirty = false;
	_tape_meta_dirty_min_slot = 0;
	_tape_meta_dirty_max_slot = -1;
	_tape_atlas_valid = true;


## Rebases `t0_rel_hi / t0_rel_lo` in `TapeMetaBuffer` relative to [param new_origin].
func _rebase_tape_meta_time_origin(bodies: Array[JrBallisticOrbiter], new_origin: float) -> void:
	_tape_gpu_time_origin = new_origin;
	var count: int = bodies.size();
	_tape_meta_vec4s.resize(count * 2);
	for i: int in range(count):
		_write_tape_meta_entry(i, bodies[i], bodies[i]._trajectory_tape);
	if count > 0 and _tape_meta_buffer_rid.is_valid():
		var meta_bytes: PackedByteArray = _tape_meta_vec4s.to_byte_array();
		_rd.buffer_update(_tape_meta_buffer_rid, 0, meta_bytes.size(), meta_bytes);
	_tape_meta_gpu_dirty = false;
	_tape_meta_dirty_min_slot = 0;
	_tape_meta_dirty_max_slot = -1;


## Encodes the 16-byte push constant payload for [code]ballistic_forecast.glsl[/code].
func _build_forecast_push_constants(
		inv_timeline_dt: float,
		body_count: int,
		well_count: int,
		timeline_segments: int
) -> PackedByteArray:
	var bytes: PackedByteArray = [];
	bytes.resize(PUSH_CONSTANT_BYTES);
	bytes.encode_float(0, inv_timeline_dt);
	bytes.encode_u32(4, body_count);
	bytes.encode_u32(8, well_count);
	bytes.encode_u32(12, timeline_segments);
	return bytes;


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

	_prepare_cpu_forecast_tables(well_count, timeline_floats);
	for body: JrBallisticOrbiter in bodies:
		var anchor_elem: JrOrreryElement = body.get_anchor_element();
		var anchor_idx: int = (
				anchor_elem._runtime_well_index
				if (
						anchor_elem != null
						and anchor_elem._runtime_well_index >= 0
						and anchor_elem._runtime_well_index < well_count
						and well_list[anchor_elem._runtime_well_index] == anchor_elem
				)
				else 0
		);
		results[body] = _integrate_single_body_forecast_cpu(body, anchor_idx, inv_timeline_dt);

	_fc_timeline = PackedFloat32Array();
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_SOLVER_FORECAST_CPU,
				Time.get_ticks_usec() - t0
		);
	return results;


## Executes identical rolling trajectory-tape integration on the CPU fallback path.
func _bake_tapes_cpu(
		config: JrBallisticForecastConfig,
		well_count: int,
		timeline_floats: PackedFloat32Array,
		inv_timeline_dt: float
) -> void:
	var t0: int = Time.get_ticks_usec() if _telemetry != null else 0;
	_meta_precommitted_in_fused = false;
	_job_wrote_atlas_flags.clear();
	_job_compact_out_flags.clear();
	_prepare_cpu_forecast_tables(well_count, timeline_floats);
	for i: int in range(config.bodies.size()):
		_integrate_single_tape_job_cpu(config, i, inv_timeline_dt);
	_fc_timeline = PackedFloat32Array();
	_finalize_tape_atlas_after_bake(config, false);
	if _telemetry != null:
		_telemetry.add_phase_usec(
				JrOrreryTelemetry.PHASE_SOLVER_FORECAST_CPU,
				Time.get_ticks_usec() - t0
		);


## Integrates a single trajectory-tape job `idx` in [param config] on the CPU matching
## `ballistic_forecast.glsl`.
func _integrate_single_tape_job_cpu(
		config: JrBallisticForecastConfig,
		idx: int,
		inv_timeline_dt: float
) -> void:
	var body: JrBallisticOrbiter = config.bodies[idx];
	var mode: int = (
			config.job_append_mode[idx]
			if idx < config.job_append_mode.size()
			else JrBallisticForecastConfig.JOB_RESET
	);
	var steps: int = maxi(
			1,
			config.job_step_counts[idx] if idx < config.job_step_counts.size() else body.preview_steps
	);
	var step_dt: float = (
			config.job_knot_dts[idx]
			if idx < config.job_knot_dts.size()
			else (maxf(0.0, body.preview_duration) / float(steps))
	);
	var job_t0: float = (
			config.job_start_times[idx]
			if idx < config.job_start_times.size()
			else config.start_time
	);
	var substeps: int = clampi(
			config.job_substeps[idx] if idx < config.job_substeps.size() else FORECAST_INTERNAL_SUBSTEPS,
			1,
			64
	);

	var r_rel: Vector3;
	var v_rel: Vector3;
	var anchor_idx: int;
	if mode == JrBallisticForecastConfig.JOB_APPEND_TAIL and body._trajectory_tape.is_valid:
		var tape_in: JrBallisticTrajectoryTape = body._trajectory_tape;
		if body.runtime_forecast_enabled and tape_in.epoch_unperturbed and not body._has_external_thrust:
			anchor_idx = tape_in.epoch_well_idx;
			var mu_ep: float = (
					_fc_well_mu[anchor_idx]
					if (anchor_idx >= 0 and anchor_idx < _fc_well_count)
					else 0.0
			);
			if mu_ep > 0.0:
				_kepler_drift_into(tape_in.epoch_pos, tape_in.epoch_vel, mu_ep, job_t0 - tape_in.epoch_time);
				r_rel = _drift_r;
				v_rel = _drift_v;
			else:
				r_rel = tape_in.get_tail_position();
				v_rel = tape_in.get_tail_velocity();
		else:
			r_rel = tape_in.get_tail_position();
			v_rel = tape_in.get_tail_velocity();
			anchor_idx = tape_in.get_tail_well_index();
	else:
		if body.is_tape_position_pending():
			body.ensure_tape_position();
		if body.is_tape_velocity_pending():
			body.ensure_tape_velocity();
		r_rel = body._anchor_relative_position;
		v_rel = body._anchor_relative_velocity;
		anchor_idx = body.get_anchor_well_index();

	var pos_out := PackedVector4Array();
	var vel_out := PackedVector4Array();
	pos_out.resize(steps + 1);
	vel_out.resize(steps + 1);

	if step_dt <= 0.0:
		var static_p := Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));
		var static_v := Vector4(v_rel.x, v_rel.y, v_rel.z, 0.0);
		for k: int in range(steps + 1):
			pos_out[k] = static_p;
			vel_out[k] = static_v;
		_commit_body_tape_job(config, idx, pos_out, vel_out);
		return;

	var ext: Vector3 = (
			body.get_scaled_external_acceleration()
			if body._has_external_thrust
			else Vector3.ZERO
	);
	var has_thrust: bool = ext != Vector3.ZERO;
	var is_orbital: bool = body.thrust_frame == JrBallisticOrbiter.ThrustFrame.ORBITAL;
	var enable_tidal: bool = body.enable_tidal_forces;
	var is_pinned: bool = body.pinned_anchor != null;
	var elapsed: float = maxf(0.0, job_t0 - config.start_time);
	var mu_anchor: float = (
			_fc_well_mu[anchor_idx]
			if (anchor_idx >= 0 and anchor_idx < _fc_well_count)
			else 0.0
	);

	var sc0: float = (elapsed * inv_timeline_dt) if enable_tidal else -1.0;
	var pert_accel: Vector3 = (
			_compute_fc_perturbing_fast(r_rel, anchor_idx, sc0)
			+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
	);
	var all_unperturbed: bool = (not has_thrust) and (mu_anchor > 0.0) and (not _pert_active_out);
	var r_job0: Vector3 = r_rel;
	var v_job0: Vector3 = v_rel;
	var elapsed0: float = elapsed;
	pos_out[0] = Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));
	vel_out[0] = Vector4(v_rel.x, v_rel.y, v_rel.z, 1.0 if all_unperturbed else 0.0);

	var sub_dt: float = step_dt / float(substeps);
	for k: int in range(steps):
		if all_unperturbed:
			var dt_from_job0: float = float(k + 1) * step_dt;
			_kepler_drift_into(r_job0, v_job0, mu_anchor, dt_from_job0);
			var r_cand: Vector3 = _drift_r;
			var v_cand: Vector3 = _drift_v;
			var t_cand: float = elapsed0 + dt_from_job0;
			var sc_cand: float = (t_cand * inv_timeline_dt) if enable_tidal else -1.0;
			_compute_fc_perturbing_fast(r_cand, anchor_idx, sc_cand);
			if not _pert_active_out:
				var switched: bool = false;
				if not is_pinned and _fc_well_count > 0:
					_fc_cur_r = r_cand;
					_fc_cur_v = v_cand;
					_fc_cur_anchor = anchor_idx;
					switched = _maybe_switch_fc_anchor_fast(t_cand * inv_timeline_dt, inv_timeline_dt);
				if not switched:
					r_rel = r_cand;
					v_rel = v_cand;
					elapsed = t_cand;
					pos_out[k + 1] = Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));
					vel_out[k + 1] = Vector4(v_rel.x, v_rel.y, v_rel.z, 1.0);
					continue;
			all_unperturbed = false;
			pert_accel = Vector3.ZERO;

		for _s: int in range(substeps):
			v_rel += 0.5 * sub_dt * pert_accel;
			_kepler_drift_into(r_rel, v_rel, mu_anchor, sub_dt);
			r_rel = _drift_r;
			v_rel = _drift_v;
			elapsed += sub_dt;

			var sc: float = (elapsed * inv_timeline_dt) if enable_tidal else -1.0;
			pert_accel = (
					_compute_fc_perturbing_fast(r_rel, anchor_idx, sc)
					+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
			);
			if _pert_active_out:
				all_unperturbed = false;
			v_rel += 0.5 * sub_dt * pert_accel;

		if not is_pinned and _fc_well_count > 0:
			_fc_cur_r = r_rel;
			_fc_cur_v = v_rel;
			_fc_cur_anchor = anchor_idx;
			if _maybe_switch_fc_anchor_fast(elapsed * inv_timeline_dt, inv_timeline_dt):
				all_unperturbed = false;
				r_rel = _fc_cur_r;
				v_rel = _fc_cur_v;
				anchor_idx = _fc_cur_anchor;
				mu_anchor = (
						_fc_well_mu[anchor_idx]
						if (anchor_idx >= 0 and anchor_idx < _fc_well_count)
						else 0.0
				);
				var sc_sw: float = (elapsed * inv_timeline_dt) if enable_tidal else -1.0;
				pert_accel = (
						_compute_fc_perturbing_fast(r_rel, anchor_idx, sc_sw)
						+ _evaluate_thrust_cpu(r_rel, v_rel, ext, is_orbital)
				);

		pos_out[k + 1] = Vector4(r_rel.x, r_rel.y, r_rel.z, float(anchor_idx));
		vel_out[k + 1] = Vector4(v_rel.x, v_rel.y, v_rel.z, 1.0 if all_unperturbed else 0.0);

	_commit_body_tape_job(config, idx, pos_out, vel_out);


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


## Evaluates interpolated timeline perturbing gravity on the CPU using precomputed tables,
## updating [member _pert_active_out].
func _compute_fc_perturbing_fast(
		r_rel: Vector3,
		anchor_idx: int,
		sample_coord_or_neg: float
) -> Vector3:
	_pert_active_out = false;
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
	var any_active: bool = false;

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
			any_active = true;
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

	_pert_active_out = any_active;
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
			_fc_cur_v += _sample_fc_well_vel(_fc_cur_anchor, idx_a, frac, inv_timeline_dt);
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
				_fc_cur_v -= _sample_fc_well_vel(w, idx_a, frac, inv_timeline_dt);
				_fc_cur_anchor = w;
				entered_child = true;
				break;
		if not entered_child:
			break;

	return _fc_cur_anchor != start_anchor;


## Evaluates 2nd-order finite-difference local velocity of [param well_idx] at timeline knot [param seg_idx].
func _sample_fc_well_knot_vel(well_idx: int, seg_idx: int, inv_timeline_dt: float) -> Vector3:
	if well_idx <= 0 or well_idx >= _fc_well_count or _fc_max_seg <= 0:
		return Vector3.ZERO;
	if _fc_max_seg == 1:
		var p0_1: Vector3 = _sample_fc_well_pos(well_idx, 0, 0, 0.0);
		var p1_1: Vector3 = _sample_fc_well_pos(well_idx, 1, 1, 0.0);
		return (p1_1 - p0_1) * inv_timeline_dt;
	if seg_idx <= 0:
		var p0: Vector3 = _sample_fc_well_pos(well_idx, 0, 0, 0.0);
		var p1: Vector3 = _sample_fc_well_pos(well_idx, 1, 1, 0.0);
		var p2: Vector3 = _sample_fc_well_pos(well_idx, 2, 2, 0.0);
		return (-1.5 * p0 + 2.0 * p1 - 0.5 * p2) * inv_timeline_dt;
	if seg_idx >= _fc_max_seg:
		var pn: Vector3 = _sample_fc_well_pos(well_idx, _fc_max_seg, _fc_max_seg, 0.0);
		var pn1: Vector3 = _sample_fc_well_pos(well_idx, _fc_max_seg - 1, _fc_max_seg - 1, 0.0);
		var pn2: Vector3 = _sample_fc_well_pos(well_idx, _fc_max_seg - 2, _fc_max_seg - 2, 0.0);
		return (1.5 * pn - 2.0 * pn1 + 0.5 * pn2) * inv_timeline_dt;
	var p_prev: Vector3 = _sample_fc_well_pos(well_idx, seg_idx - 1, seg_idx - 1, 0.0);
	var p_next: Vector3 = _sample_fc_well_pos(well_idx, seg_idx + 1, seg_idx + 1, 0.0);
	return (p_next - p_prev) * (0.5 * inv_timeline_dt);


## Samples interpolated 2nd-order local velocity of [param well_idx] at `idx_a + frac`.
func _sample_fc_well_vel(
		well_idx: int,
		idx_a: int,
		frac: float,
		inv_timeline_dt: float
) -> Vector3:
	if well_idx <= 0 or well_idx >= _fc_well_count or _fc_max_seg <= 0:
		return Vector3.ZERO;
	var va: Vector3 = _sample_fc_well_knot_vel(well_idx, idx_a, inv_timeline_dt);
	var idx_b: int = mini(idx_a + 1, _fc_max_seg);
	if idx_a == idx_b or frac <= 0.0:
		return va;
	return va.lerp(_sample_fc_well_knot_vel(well_idx, idx_b, inv_timeline_dt), frac);


## Returns [member JrBallisticForecastConfig.prebuilt_wells] when populated by [JrOrbitDriver],
## or builds the ordered gravity-well list from [param config].
func _resolve_forecast_well_list(config: JrBallisticForecastConfig) -> Array[JrOrreryElement]:
	if not config.prebuilt_wells.is_empty():
		return config.prebuilt_wells;
	return _build_forecast_well_list(config);


## Builds the ordered list of all gravity-well elements (root driver at index 0,
## followed by all valid [JrRailOrbiter] nodes in parent-before-child order) and assigns
## `_runtime_well_index` on each element for `O(1)` lookup.
func _build_forecast_well_list(config: JrBallisticForecastConfig) -> Array[JrOrreryElement]:
	var wells: Array[JrOrreryElement] = [];
	if config.driver_element != null:
		config.driver_element._runtime_well_index = 0;
	wells.append(config.driver_element);
	for orbiter: JrRailOrbiter in config.orbit_sequence:
		if orbiter != null and orbiter.is_orbit_valid():
			orbiter._runtime_well_index = wells.size();
			wells.append(orbiter);
		elif orbiter != null:
			orbiter._runtime_well_index = -1;
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


## Computes a compact floating-point signature of [param orb]'s Keplerian orbital elements
## to detect runtime rail mutations and invalidate [member _tl_grid_v4s].
static func _compute_well_orbital_sig(orb: JrRailOrbiter) -> float:
	return (
			orb.periapsis * 1.0001
			+ orb.eccentricity * 13.0007
			+ orb.mean_motion * 7.0003
			+ orb.mean_anomaly_at_epoch * 3.0011
			+ orb.inclination * 5.0013
			+ orb.argument_of_periapsis * 11.0017
			+ orb.longitude_of_the_ascending_node * 17.0019
	);


## Validates or rebuilds the static per-well `mu`, parent index, and Hill-radius multipliers
## cached for [method _sample_well_timeline_bytes].
func _ensure_timeline_well_cache(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement]
) -> void:
	var well_count: int = well_list.size();
	var has_prebuilt: bool = not config.prebuilt_wells.is_empty();
	var cache_valid: bool = (
			_tl_cached_wells.size() == well_count
			and _tl_well_mu_f64.size() == well_count
			and _tl_well_orbital_sig.size() == well_count
			and _tl_cached_driver_mu == config.driver_gravity_strength
			and _tl_cached_ref_plane == config.reference_plane
	);
	if cache_valid:
		if well_count > 0 and _tl_cached_wells[0] != well_list[0]:
			cache_valid = false;
		for w: int in range(1, well_count):
			if not cache_valid:
				break;
			var orb := well_list[w] as JrRailOrbiter;
			if orb != _tl_cached_wells[w] or orb.needs_nu_spline_bake():
				cache_valid = false;
				break;
			var partner_flag: int = 1 if orb.binary_partner != null else 0;
			var fast_flag: int = (
					1
					if (
							orb._has_ref_frame_cache
							and (config.reference_plane == null or config.reference_plane == orb._reference_plane)
					)
					else 0
			);
			var p_elem: JrOrreryElement = (
					orb._tracked_scene_parent
					if (has_prebuilt and orb._tracked_scene_parent != null)
					else config.orbiter_parents.get(orb, config.driver_element)
			);
			var p_idx: int = (
					p_elem._runtime_well_index
					if (p_elem != null and p_elem._runtime_well_index >= 0 and p_elem._runtime_well_index < w)
					else 0
			);
			if (
					partner_flag != _tl_cached_has_partner[w]
					or fast_flag != _tl_use_fast_pos[w]
					or float(p_idx) != _tl_well_parent_float[w]
					or orb.get_gravity_strength() != _tl_well_mu_f64[w]
					or _compute_well_orbital_sig(orb) != _tl_well_orbital_sig[w]
			):
				cache_valid = false;
				break;

	if cache_valid:
		return;

	_tl_grid_valid = false;
	_tl_grid_bytes = PackedByteArray();
	_tl_cached_wells = well_list.duplicate();
	_tl_cached_driver_mu = config.driver_gravity_strength;
	_tl_cached_ref_plane = config.reference_plane;
	_tl_cached_has_partner.resize(well_count);
	_tl_well_mu.resize(well_count);
	_tl_well_mu_f64.resize(well_count);
	_tl_well_parent_float.resize(well_count);
	_tl_well_entry_ratio.resize(well_count);
	_tl_well_exit_ratio.resize(well_count);
	_tl_use_fast_pos.resize(well_count);
	_tl_well_orbital_sig.resize(well_count);
	_tl_max_mean_n = 0.0;

	var eff_mu_by_idx := PackedFloat32Array();
	var unbaked_wells: Array[JrRailOrbiter] = [];
	eff_mu_by_idx.resize(well_count);
	if well_count > 0:
		_tl_well_mu[0] = config.driver_gravity_strength;
		_tl_well_mu_f64[0] = config.driver_gravity_strength;
		_tl_well_parent_float[0] = -1.0;
		_tl_well_entry_ratio[0] = 0.0;
		_tl_well_exit_ratio[0] = 0.0;
		_tl_use_fast_pos[0] = 0;
		_tl_cached_has_partner[0] = 0;
		_tl_well_orbital_sig[0] = 0.0;
		eff_mu_by_idx[0] = (
				config.driver_element.get_effective_parent_mu()
				if config.driver_element != null
				else config.driver_gravity_strength
		);

	for w: int in range(1, well_count):
		var orbiter := well_list[w] as JrRailOrbiter;
		if orbiter._reference_plane == null and config.reference_plane != null:
			orbiter.set_reference_plane(config.reference_plane);
		if orbiter.needs_nu_spline_bake():
			unbaked_wells.append(orbiter);
		_tl_cached_has_partner[w] = 1 if orbiter.binary_partner != null else 0;
		_tl_well_orbital_sig[w] = _compute_well_orbital_sig(orbiter);
		_tl_max_mean_n = maxf(_tl_max_mean_n, orbiter.mean_motion);
		eff_mu_by_idx[w] = orbiter.get_effective_parent_mu();
		var mu: float = orbiter.get_gravity_strength();
		var parent_elem: JrOrreryElement = (
				orbiter._tracked_scene_parent
				if (has_prebuilt and orbiter._tracked_scene_parent != null)
				else config.orbiter_parents.get(orbiter, config.driver_element)
		);
		var parent_idx: int = (
				parent_elem._runtime_well_index
				if (
						parent_elem != null
						and parent_elem._runtime_well_index >= 0
						and parent_elem._runtime_well_index < w
				)
				else 0
		);
		var parent_mu: float = eff_mu_by_idx[parent_idx];
		var cube_root_ratio: float = (
				pow(mu / (3.0 * parent_mu), 1.0 / 3.0)
				if (mu > 0.0 and parent_mu > 0.0)
				else 0.0
		);
		_tl_well_mu[w] = mu;
		_tl_well_mu_f64[w] = mu;
		_tl_well_parent_float[w] = float(parent_idx);
		_tl_well_entry_ratio[w] = cube_root_ratio * HILL_ENTRY_MULTIPLIER;
		_tl_well_exit_ratio[w] = cube_root_ratio * HILL_EXIT_MULTIPLIER;

	if not unbaked_wells.is_empty():
		bake_rail_splines(unbaked_wells);

	for w: int in range(1, well_count):
		var orbiter := well_list[w] as JrRailOrbiter;
		_tl_use_fast_pos[w] = (
				1
				if (
						orbiter._has_ref_frame_cache
						and (config.reference_plane == null or config.reference_plane == orbiter._reference_plane)
				)
				else 0
		);


## Evaluates and writes `[idx_range.x .. idx_range.y]` grid rows into [member _tl_grid_v4s] at [param step_dt].
func _populate_timeline_grid_range(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement],
		idx_range: Vector2i,
		step_dt: float
) -> void:
	var well_count: int = well_list.size();
	var cull_sq: float = _tl_grid_cull_sq;
	var root_v0 := Vector4(0.0, 0.0, 0.0, config.driver_gravity_strength);
	var root_v1 := Vector4(0.0, 0.0, -1.0, cull_sq);
	var grid_v4s: PackedVector4Array = _tl_grid_v4s;
	_tl_grid_v4s = PackedVector4Array();
	var v_idx: int = (idx_range.x - _tl_grid_start_idx) * well_count * 2;

	for g_idx: int in range(idx_range.x, idx_range.y + 1):
		var eval_time: float = float(g_idx) * step_dt;
		grid_v4s[v_idx] = root_v0;
		grid_v4s[v_idx + 1] = root_v1;
		v_idx += 2;
		for w: int in range(1, well_count):
			var orbiter := well_list[w] as JrRailOrbiter;
			var local_pos: Vector3 = (
					orbiter.sample_local_position_fast(eval_time)
					if _tl_use_fast_pos[w] != 0
					else orbiter.get_relative_position(eval_time, config.reference_plane)
			);
			var e_ratio: float = _tl_well_entry_ratio[w];
			var dist: float = local_pos.length() if e_ratio > 0.0 else 0.0;
			grid_v4s[v_idx] = Vector4(local_pos.x, local_pos.y, local_pos.z, _tl_well_mu[w]);
			grid_v4s[v_idx + 1] = Vector4(
					dist * e_ratio,
					dist * _tl_well_exit_ratio[w],
					_tl_well_parent_float[w],
					cull_sq
			);
			v_idx += 2;

	_tl_grid_v4s = grid_v4s;
	_tl_grid_bytes = grid_v4s.to_byte_array();
	_tl_gpu_dirty = true;


## Samples `well_samples` over a binary power-of-two timestamp grid (`step_dt = 2^k`), reusing
## overlapping slices from [member _tl_grid_v4s] across staggered cohort refills.
func _sample_tape_timeline_cached(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement]
) -> PackedByteArray:
	_ensure_timeline_well_cache(config, well_list);
	var well_count: int = well_list.size();
	var raw_end: float = (
			config.timeline_end_time
			if config.timeline_end_time > config.start_time
			else (config.start_time + _resolve_tape_batch_duration(config))
	);
	var raw_dur: float = raw_end - config.start_time;
	if well_count <= 1 or raw_dur <= 1e-9 or _tl_max_mean_n <= 1e-9:
		var fallback_segs: int = _compute_adaptive_timeline_segments(config, well_list, raw_dur);
		_tl_last_inv_dt = float(fallback_segs) / raw_dur if raw_dur > 1e-12 else 0.0;
		return _sample_well_timeline_bytes(config, well_list, raw_dur, fallback_segs);

	var max_step_rad: float = 0.03 if config.enable_forecasts else 0.05;
	var desired_dt: float = max_step_rad / _tl_max_mean_n;
	var exp2_val: float = floorf(log(maxf(1e-9, desired_dt)) / 0.6931471805599453);
	var step_dt: float = pow(2.0, exp2_val);
	var max_cap: int = clampi(65536 / maxi(1, well_count), 32, 1024);
	var ref_span: float = maxf(raw_dur, 2.0);
	while int(ceili(ref_span / step_dt)) + 2 > max_cap:
		step_dt *= 2.0;

	var inv_dt: float = 1.0 / step_dt;
	if absf(config.start_time) * inv_dt > 1.0e8 or absf(raw_end) * inv_dt > 1.0e8:
		var fb_segs: int = _compute_adaptive_timeline_segments(config, well_list, raw_dur);
		_tl_last_inv_dt = float(fb_segs) / raw_dur if raw_dur > 1e-12 else 0.0;
		return _sample_well_timeline_bytes(config, well_list, raw_dur, fb_segs);

	var req_idx0: int = int(floor(config.start_time * inv_dt));
	var req_idx1: int = maxi(req_idx0 + 8, int(ceil(raw_end * inv_dt)));
	_tl_last_inv_dt = inv_dt;

	var cull_sq: float = config.tidal_cull_threshold * config.tidal_cull_threshold;
	var row_stride: int = well_count * 2;
	var lookahead: int = 256;
	var max_window: int = maxi(req_idx1 - req_idx0 + lookahead, mini(max_cap, 768));

	if _tl_grid_valid and _tl_grid_step_dt == step_dt and _tl_grid_cull_sq == cull_sq:
		if req_idx0 >= _tl_grid_start_idx and req_idx1 <= _tl_grid_end_idx:
			config.start_time = float(_tl_grid_start_idx) * step_dt;
			return _tl_grid_bytes;
		if (
				req_idx0 >= _tl_grid_start_idx
				and req_idx0 <= _tl_grid_end_idx
				and req_idx1 > _tl_grid_end_idx
				and (req_idx1 - _tl_grid_end_idx) <= 512
		):
			var prev_end: int = _tl_grid_end_idx;
			var new_end: int = req_idx1 + lookahead;
			var keep_start: int = maxi(_tl_grid_start_idx, mini(req_idx0, new_end - max_window));
			var keep_v4s: PackedVector4Array = _tl_grid_v4s.slice(
					(keep_start - _tl_grid_start_idx) * row_stride,
					(prev_end - _tl_grid_start_idx + 1) * row_stride
			);
			keep_v4s.resize((new_end - keep_start + 1) * row_stride);
			_tl_grid_start_idx = keep_start;
			_tl_grid_end_idx = new_end;
			_tl_grid_v4s = keep_v4s;
			_populate_timeline_grid_range(config, well_list, Vector2i(prev_end + 1, new_end), step_dt);
			config.start_time = float(_tl_grid_start_idx) * step_dt;
			return _tl_grid_bytes;

	var full_end: int = req_idx1 + lookahead;
	_tl_grid_valid = true;
	_tl_grid_step_dt = step_dt;
	_tl_grid_cull_sq = cull_sq;
	_tl_grid_start_idx = req_idx0;
	_tl_grid_end_idx = full_end;
	_tl_grid_v4s.resize((full_end - req_idx0 + 1) * row_stride);
	_populate_timeline_grid_range(config, well_list, Vector2i(req_idx0, full_end), step_dt);
	config.start_time = float(_tl_grid_start_idx) * step_dt;
	return _tl_grid_bytes;


## Pre-samples parent-relative positions, gravitational parameters, parent well indices,
## and Hill-sphere switch radii across `segments + 1` uniform timestamps directly as raw bytes.
func _sample_well_timeline_bytes(
		config: JrBallisticForecastConfig,
		well_list: Array[JrOrreryElement],
		max_duration: float,
		segments: int
) -> PackedByteArray:
	var well_count: int = well_list.size();
	if well_count == 0:
		return PackedByteArray();

	_ensure_timeline_well_cache(config, well_list);

	var cull_sq: float = config.tidal_cull_threshold * config.tidal_cull_threshold;
	var root_v0 := Vector4(0.0, 0.0, 0.0, config.driver_gravity_strength);
	var root_v1 := Vector4(0.0, 0.0, -1.0, cull_sq);
	var vec4s: PackedVector4Array = _tl_vec4s;
	_tl_vec4s = PackedVector4Array();
	var req_v4s: int = (segments + 1) * well_count * 2;
	if vec4s.size() != req_v4s:
		vec4s.resize(req_v4s);

	var step_time: float = max_duration / float(segments);
	var v_idx: int = 0;

	for seg_idx: int in range(segments + 1):
		var eval_time: float = config.start_time + step_time * float(seg_idx);
		vec4s[v_idx] = root_v0;
		vec4s[v_idx + 1] = root_v1;
		v_idx += 2;

		for w: int in range(1, well_count):
			var orbiter := well_list[w] as JrRailOrbiter;
			var local_pos: Vector3 = (
					orbiter.sample_local_position_fast(eval_time)
					if _tl_use_fast_pos[w] != 0
					else orbiter.get_relative_position(eval_time, config.reference_plane)
			);
			var e_ratio: float = _tl_well_entry_ratio[w];
			var dist: float = local_pos.length() if e_ratio > 0.0 else 0.0;
			vec4s[v_idx] = Vector4(local_pos.x, local_pos.y, local_pos.z, _tl_well_mu[w]);
			vec4s[v_idx + 1] = Vector4(
					dist * e_ratio,
					dist * _tl_well_exit_ratio[w],
					_tl_well_parent_float[w],
					cull_sq
			);
			v_idx += 2;

	var out_bytes: PackedByteArray = vec4s.to_byte_array();
	_tl_vec4s = vec4s;
	_tl_grid_valid = false;
	_tl_gpu_dirty = true;
	return out_bytes;


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
