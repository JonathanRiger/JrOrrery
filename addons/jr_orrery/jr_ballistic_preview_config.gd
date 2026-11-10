@tool
class_name JrBallisticPreviewConfig
extends RefCounted
## Configuration data bundle for computing a forward-integrated editor trajectory
## preview in [JrBallisticComputeSolver].
##
## Groups initial ballistic state, preview horizon parameters, and the owning
## [JrOrbitDriver]'s hierarchical gravity-well topology so solver methods remain
## concise and strongly typed.

## Starting inertial position of the [JrBallisticOrbiter] body at [code]time = 0.0[/code].
var initial_inertial_position := Vector3.ZERO

## Starting inertial velocity of the [JrBallisticOrbiter] body at [code]time = 0.0[/code].
var initial_inertial_velocity := Vector3.ZERO

## Continuous external acceleration vector applied throughout the preview horizon.
var external_acceleration := Vector3.ZERO

## Simulation timestamp from which forward gravity-well evaluation begins.
var start_time: float = 0.0

## Total forward simulation duration (in simulation time units) to integrate.
var duration: float = 30.0

## Number of integration steps across [member duration].
var steps: int = 192

## Gravitational strength (mu) of the root [JrOrbitDriver] at the system origin.
var driver_gravity_strength: float = 0.0

## Active 3D [JrReferenceDirection] of the system.
var reference_plane: JrReferenceDirection = null

## Ordered array of tracked [JrRailOrbiter] nodes (parent-before-child) in the driver hierarchy.
var orbit_sequence: Array[JrRailOrbiter] = []

## Lookup mapping each tracked [JrRailOrbiter] to its immediate parent [JrOrreryElement].
var orbiter_parents: Dictionary[JrRailOrbiter, JrOrreryElement] = {}
