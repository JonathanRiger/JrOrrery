@tool

## Configuration data bundle for computing batched forward-integrated N-body
## trajectory forecasts across multiple [JrBallisticOrbiter] bodies in [JrBallisticComputeSolver].
##
## Groups the tracked ballistic bodies, current simulation timestamp, timeline
## resolution, tidal culling threshold, and the owning [JrOrbitDriver]'s hierarchical
## gravity-well topology so solver methods remain concise, strongly typed, and under four parameters.
extends RefCounted
class_name JrBallisticForecastConfig

## Tracked [JrBallisticOrbiter] bodies to forecast forward from [member start_time].
var bodies: Array[JrBallisticOrbiter] = [];

## Simulation timestamp at which the forecast begins.
var start_time: float = 0.0;

## Number of uniform gravity-well timeline segments sampled across the maximum body duration.
var timeline_segments: int = 128;

## Root [JrOrreryElement] (the owning [JrOrbitDriver]) at well index 0.
var driver_element: JrOrreryElement = null;

## Gravitational strength (mu) of the root [JrOrbitDriver] at the system origin.
var driver_gravity_strength: float = 0.0;

## Dimensionless tidal culling ratio threshold (e.g. 1e-4).
var tidal_cull_threshold: float = 0.0001;

## Active 3D [JrReferenceDirection] of the system.
var reference_plane: JrReferenceDirection = null;

## Ordered array of tracked [JrRailOrbiter] nodes (parent-before-child) in the driver hierarchy.
var orbit_sequence: Array[JrRailOrbiter] = [];

## Lookup mapping each tracked [JrRailOrbiter] to its immediate parent [JrOrreryElement].
var orbiter_parents: Dictionary[JrRailOrbiter, JrOrreryElement] = {};
