@tool
class_name JrBallisticForecastConfig
extends RefCounted
## Configuration data bundle for computing batched forward-integrated N-body
## trajectory horizons and forecasts across multiple [JrBallisticOrbiter] bodies
## in [JrBallisticComputeSolver].
##
## Groups the tracked ballistic bodies, timeline window, per-job step parameters
## (supporting mixed full-horizon bakes, incremental tail-chunk appends, and closed
## periodic loops in a single GPU dispatch), tidal culling threshold, and the owning
## [JrOrbitDriver]'s hierarchical gravity-well topology so solver methods remain
## concise, strongly typed, and under four parameters.

## Job mode constant performing a full tape reset from the body's current state.
const JOB_RESET: int = 0;

## Job mode constant appending a new tail chunk from the body's existing tape tail state.
const JOB_APPEND_TAIL: int = 1;

## Job mode constant baking one closed periodic orbit loop from the body's current state.
const JOB_PERIODIC_LOOP: int = 2;

## Tracked [JrBallisticOrbiter] bodies to integrate forward in this batch.
var bodies: Array[JrBallisticOrbiter] = [];

## Earliest simulation timestamp (`t0`) at which the shared gravity-well timeline begins.
var start_time: float = 0.0;

## Latest simulation timestamp (`t1`) covered by the shared gravity-well timeline.
var timeline_end_time: float = 0.0;

## Timestamp before which consumed prefix knots may be trimmed during `JOB_APPEND_TAIL` jobs.
var trim_before_time: float = 0.0;

## Minimum number of uniform gravity-well timeline segments across `[start_time, timeline_end_time]`.
var timeline_segments: int = 128;

## Root [JrOrreryElement] (the owning [JrOrbitDriver]) at well index 0.
var driver_element: JrOrreryElement = null;

## Gravitational strength (`mu`) of the root [JrOrbitDriver] at the system origin.
var driver_gravity_strength: float = 0.0;

## Dimensionless tidal culling ratio threshold (e.g. `1e-4`).
var tidal_cull_threshold: float = 0.0001;

## Active 3D [JrReferenceDirection] of the system.
var reference_plane: JrReferenceDirection = null;

## Ordered array of tracked [JrRailOrbiter] nodes (parent-before-child) in the driver hierarchy.
var orbit_sequence: Array[JrRailOrbiter] = [];

## Lookup mapping each tracked [JrRailOrbiter] to its immediate parent [JrOrreryElement].
var orbiter_parents: Dictionary[JrRailOrbiter, JrOrreryElement] = {};

## Optional per-body 64-bit start timestamps (`>= start_time`). Defaults to [member start_time] when empty.
var job_start_times: PackedFloat64Array = PackedFloat64Array();

## Optional per-body uniform knot intervals (`> 0.0`). Defaults to `preview_duration / preview_steps` when empty.
var job_knot_dts: PackedFloat32Array = PackedFloat32Array();

## Optional per-body step counts (`>= 1`). Defaults to `preview_steps` when empty.
var job_step_counts: PackedInt32Array = PackedInt32Array();

## Optional per-body internal KDK substep counts (`>= 1`). Defaults to `2` when empty.
var job_substeps: PackedInt32Array = PackedInt32Array();

## Optional per-body job mode (`JOB_RESET`, `JOB_APPEND_TAIL`, or `JOB_PERIODIC_LOOP`).
var job_append_mode: PackedByteArray = PackedByteArray();

## Optional per-body exact 64-bit orbital periods when `job_append_mode[i] == JOB_PERIODIC_LOOP`.
var job_loop_periods: PackedFloat64Array = PackedFloat64Array();

## Total tracked [JrBallisticOrbiter] count on the owning [JrOrbitDriver] (`> 0` enables resident GPU tape-atlas writes).
var total_driver_bodies: int = 0;

