@tool
class_name JrBallisticTrajectoryTape
extends RefCounted
## Rolling Cubic Hermite trajectory-horizon buffer for a single [JrBallisticOrbiter].
##
## Stores uniform-dt anchor-relative position, velocity, gravity-well anchor index, and
## perturbation flags integrated ahead of simulation time on the GPU (or CPU fallback),
## along with lazily cached world-inertial and perspective-projected knot arrays for
## `O(1)` state evaluation and C++ SIMD forecast polyline slicing. Supports VRAM-resident
## GPU-atlas-only tapes whose CPU knot arrays are downloaded on demand per slot.

## Simulation timestamp (64-bit float) corresponding to knot index `0`.
var start_time: float = 0.0;

## Uniform simulation time interval (`> 0.0`) between consecutive knots.
var knot_dt: float = 0.0;

## Precomputed reciprocal `1.0 / knot_dt` for division-free segment index lookup.
var inv_knot_dt: float = 0.0;

## Total number of knots currently stored in [member pos_anchor] / VRAM atlas.
var knot_count: int = 0;

## Whether this tape currently holds at least two integrated knots for valid Hermite sampling.
var is_valid: bool = false;

## Whether [member pos_anchor] and [member vel_flags] are currently populated in CPU memory.
var cpu_knots_resident: bool = true;

## Slot index in the solver's VRAM `TapeAtlasBuffer` when [member cpu_knots_resident] is `false`.
var atlas_slot_index: int = -1;

## Weak reference to the owning `JrBallisticComputeSolver` for lazy single-slot VRAM download.
var atlas_solver_weak: WeakRef = null;

## Cached final knot position and anchor index (`xyz = r_tail, w = float(well_idx)`).
var tail_pos_anchor: Vector4 = Vector4.ZERO;

## Cached final knot velocity and unperturbed flag (`xyz = v_tail, w = unperturbed_flag`).
var tail_vel_flags: Vector4 = Vector4.ZERO;

## Whether this tape encodes one closed period of an unperturbed bound Keplerian orbit.
var is_periodic_loop: bool = false;

## Exact 64-bit orbital period when [member is_periodic_loop] is `true`.
var loop_period: float = 0.0;

## Knot positions and anchor indices (`xyz = anchor_relative_pos, w = float(well_idx)`).
var pos_anchor: PackedVector4Array = PackedVector4Array();

## Knot velocities and flags (`xyz = anchor_relative_vel, w = cumulative_unperturbed_flag`).
var vel_flags: PackedVector4Array = PackedVector4Array();

## Cached driver-inertial positions (`R_well(t_k) + r_rel_k`) at each knot timestamp `t_k`.
var inertial_points: PackedVector3Array = PackedVector3Array();

## Number of prefix knots `[0 .. inertial_built_count - 1]` populated in [member inertial_points].
var inertial_built_count: int = 0;

## Active `JrOrbitDriver` perspective cache epoch matching [member persp_points].
var persp_epoch: int = -1;

## Cached time-invariant perspective-offset vectors at each knot timestamp `t_k`.
var persp_points: PackedVector3Array = PackedVector3Array();

## Number of prefix knots `[0 .. persp_built_count - 1]` populated in [member persp_points].
var persp_built_count: int = 0;

## Whether contiguous Sphere-of-Influence (SOI) span runs have been indexed for current knots.
var soi_spans_ready: bool = false;

## Start knot index of each contiguous `well_idx` run across `[0 .. knot_count - 1]`.
var span_starts: PackedInt32Array = PackedInt32Array();

## Inclusive end knot index of each contiguous `well_idx` run across `[0 .. knot_count - 1]`.
var span_ends: PackedInt32Array = PackedInt32Array();

## Gravity-well index (`int(roundf(pos_anchor[k].w))`) of each contiguous SOI run.
var span_wells: PackedInt32Array = PackedInt32Array();

## Exact 64-bit anchor-relative position at [member epoch_time] when [member epoch_unperturbed] is `true`.
var epoch_pos: Vector3 = Vector3.ZERO;

## Exact 64-bit anchor-relative velocity at [member epoch_time] when [member epoch_unperturbed] is `true`.
var epoch_vel: Vector3 = Vector3.ZERO;

## 64-bit simulation timestamp of the unperturbed conic origin.
var epoch_time: float = 0.0;

## Gravity-well index of the unperturbed conic origin.
var epoch_well_idx: int = 0;

## Whether the trajectory from [member epoch_time] through [method get_end_time] has remained 100% unperturbed.
var epoch_unperturbed: bool = false;


## Clears all stored knots and cached projection buffers, marking this tape invalid.
func clear() -> void:
	start_time = 0.0;
	knot_dt = 0.0;
	inv_knot_dt = 0.0;
	knot_count = 0;
	is_valid = false;
	cpu_knots_resident = true;
	atlas_slot_index = -1;
	atlas_solver_weak = null;
	tail_pos_anchor = Vector4.ZERO;
	tail_vel_flags = Vector4.ZERO;
	is_periodic_loop = false;
	loop_period = 0.0;
	epoch_unperturbed = false;
	pos_anchor.clear();
	vel_flags.clear();
	inertial_points.clear();
	inertial_built_count = 0;
	persp_epoch = -1;
	persp_points.clear();
	persp_built_count = 0;
	soi_spans_ready = false;
	span_starts.clear();
	span_ends.clear();
	span_wells.clear();


## Replaces the entire trajectory tape with newly integrated CPU-resident knots starting at
## [param t0] with uniform spacing [param dt].
func reset_with_knots(
		t0: float,
		dt: float,
		new_pos: PackedVector4Array,
		new_vel: PackedVector4Array
) -> void:
	start_time = t0;
	knot_dt = maxf(1e-9, dt);
	inv_knot_dt = 1.0 / knot_dt;
	pos_anchor = new_pos;
	vel_flags = new_vel;
	knot_count = mini(new_pos.size(), new_vel.size());
	is_valid = knot_count >= 2;
	cpu_knots_resident = true;
	if knot_count > 0:
		tail_pos_anchor = new_pos[knot_count - 1];
		tail_vel_flags = new_vel[knot_count - 1];
	else:
		tail_pos_anchor = Vector4.ZERO;
		tail_vel_flags = Vector4.ZERO;
	is_periodic_loop = false;
	loop_period = 0.0;
	inertial_points.clear();
	inertial_built_count = 0;
	persp_epoch = -1;
	persp_points.clear();
	persp_built_count = 0;
	soi_spans_ready = false;


## Resets timing metadata for a tape whose full knot arrays reside in VRAM `TapeAtlasBuffer`.
func reset_gpu_atlas_grid(t0: float, dt: float, count: int) -> void:
	start_time = t0;
	knot_dt = maxf(1e-9, dt);
	inv_knot_dt = 1.0 / knot_dt;
	knot_count = count;
	is_valid = count >= 2;
	cpu_knots_resident = false;
	is_periodic_loop = false;
	loop_period = 0.0;
	if not pos_anchor.is_empty():
		pos_anchor.clear();
	if not vel_flags.is_empty():
		vel_flags.clear();
	if inertial_built_count > 0:
		inertial_points.clear();
		inertial_built_count = 0;
	persp_epoch = -1;
	if persp_built_count > 0:
		persp_points.clear();
		persp_built_count = 0;
	soi_spans_ready = false;


## Binds the 2-`vec4` tail summary and VRAM slot reference for a GPU-atlas-resident tape.
func bind_gpu_atlas_tail(
		tail_pos: Vector4,
		tail_vel: Vector4,
		slot_idx: int,
		solver_weak: WeakRef
) -> void:
	tail_pos_anchor = tail_pos;
	tail_vel_flags = tail_vel;
	atlas_slot_index = slot_idx;
	atlas_solver_weak = solver_weak;


## Downloads [member pos_anchor] and [member vel_flags] from VRAM `TapeAtlasBuffer` on demand
## if [member cpu_knots_resident] is currently `false`.
func ensure_cpu_knots() -> void:
	if cpu_knots_resident or not is_valid:
		return;
	if atlas_solver_weak != null:
		var solver_obj: Object = atlas_solver_weak.get_ref();
		if solver_obj != null:
			solver_obj.call(&"materialize_tape_from_gpu_atlas", self);
	cpu_knots_resident = true;


## Marks this tape as a closed periodic Keplerian orbit of exact duration [param period].
func mark_periodic_loop(period: float) -> void:
	if is_valid and period > 0.0:
		is_periodic_loop = true;
		loop_period = period;


## Trims consumed prefix knots strictly older than the segment containing [param trim_before_time]
## and appends `new_pos[1..]` and `new_vel[1..]` onto the tail of this tape.
func append_tail_knots(
		new_pos: PackedVector4Array,
		new_vel: PackedVector4Array,
		trim_before_time: float
) -> void:
	var incoming_count: int = mini(new_pos.size(), new_vel.size());
	if not is_valid or incoming_count < 2:
		return;

	var end_t: float = start_time + float(knot_count - 1) * knot_dt;
	if trim_before_time >= end_t - maxf(1e-7, 1e-4 * knot_dt):
		start_time = end_t;
		pos_anchor = new_pos;
		vel_flags = new_vel;
		knot_count = incoming_count;
		is_valid = true;
		cpu_knots_resident = true;
		tail_pos_anchor = new_pos[incoming_count - 1];
		tail_vel_flags = new_vel[incoming_count - 1];
		inertial_points.clear();
		inertial_built_count = 0;
		persp_epoch = -1;
		persp_points.clear();
		persp_built_count = 0;
		soi_spans_ready = false;
		return;

	ensure_cpu_knots();
	var raw_idx: int = int(floorf((trim_before_time - start_time) * inv_knot_dt)) - 1;
	var keep_from: int = clampi(raw_idx, 0, maxi(0, knot_count - 2));
	if keep_from > 0:
		_trim_prefix_knots(keep_from);

	pos_anchor.append_array(new_pos.slice(1, incoming_count));
	vel_flags.append_array(new_vel.slice(1, incoming_count));
	knot_count = mini(pos_anchor.size(), vel_flags.size());
	is_valid = knot_count >= 2;
	cpu_knots_resident = true;
	if knot_count > 0:
		tail_pos_anchor = pos_anchor[knot_count - 1];
		tail_vel_flags = vel_flags[knot_count - 1];
	soi_spans_ready = false;


## Returns the simulation timestamp of the final knot (`INF` when [member is_periodic_loop] is `true`).
func get_end_time() -> float:
	if not is_valid:
		return start_time;
	if is_periodic_loop:
		return INF;
	return start_time + float(knot_count - 1) * knot_dt;


## Returns `true` when [param eval_time] lies within the valid sampling domain of this tape.
func contains_time(eval_time: float) -> bool:
	if not is_valid:
		return false;
	if is_periodic_loop:
		return true;
	var end_t: float = start_time + float(knot_count - 1) * knot_dt;
	return eval_time >= start_time - 1e-7 and eval_time <= end_t + 1e-7;


## Returns the remaining future lookahead duration (`get_end_time() - eval_time`).
func remaining_horizon(eval_time: float) -> float:
	if not is_valid:
		return -INF;
	if is_periodic_loop:
		return INF;
	return (start_time + float(knot_count - 1) * knot_dt) - eval_time;


## Returns the anchor-relative position stored at the final tail knot in `O(1)`.
func get_tail_position() -> Vector3:
	if knot_count <= 0:
		return Vector3.ZERO;
	return Vector3(tail_pos_anchor.x, tail_pos_anchor.y, tail_pos_anchor.z);


## Returns the anchor-relative velocity stored at the final tail knot in `O(1)`.
func get_tail_velocity() -> Vector3:
	if knot_count <= 0:
		return Vector3.ZERO;
	return Vector3(tail_vel_flags.x, tail_vel_flags.y, tail_vel_flags.z);


## Returns the gravity-well index stored at the final tail knot in `O(1)`.
func get_tail_well_index() -> int:
	if knot_count <= 0:
		return 0;
	return int(roundf(tail_pos_anchor.w));


## Returns `true` if the final knot reports zero perturbations across the baked interval in `O(1)`.
func was_completely_unperturbed() -> bool:
	if knot_count <= 0:
		return false;
	return tail_vel_flags.w > 0.5;


## Lazily builds contiguous Sphere-of-Influence well runs (`span_starts`, `span_ends`, `span_wells`)
## across `[0 .. knot_count - 1]` for fast SIMD slice projection.
func ensure_soi_spans() -> void:
	if soi_spans_ready:
		return;
	span_starts.clear();
	span_ends.clear();
	span_wells.clear();
	if knot_count <= 0:
		soi_spans_ready = true;
		return;

	ensure_cpu_knots();
	var cur_well: int = int(roundf(pos_anchor[0].w));
	var cur_start: int = 0;
	for k: int in range(1, knot_count):
		var w: int = int(roundf(pos_anchor[k].w));
		if w != cur_well:
			span_starts.append(cur_start);
			span_ends.append(k - 1);
			span_wells.append(cur_well);
			cur_well = w;
			cur_start = k;

	span_starts.append(cur_start);
	span_ends.append(knot_count - 1);
	span_wells.append(cur_well);
	soi_spans_ready = true;


## Drops the first [param drop_count] knots from knot and cached projection arrays in native C++.
func _trim_prefix_knots(drop_count: int) -> void:
	pos_anchor = pos_anchor.slice(drop_count);
	vel_flags = vel_flags.slice(drop_count);
	start_time += float(drop_count) * knot_dt;

	if inertial_built_count > drop_count:
		inertial_points = inertial_points.slice(drop_count);
		inertial_built_count -= drop_count;
	else:
		inertial_points.clear();
		inertial_built_count = 0;

	if persp_built_count > drop_count:
		persp_points = persp_points.slice(drop_count);
		persp_built_count -= drop_count;
	else:
		persp_points.clear();
		persp_built_count = 0;
