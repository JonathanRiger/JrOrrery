@tool
class_name JrReferenceDirection
extends RefCounted
## Establishes a unified 3D reference coordinate system (the ecliptic plane and
## its pole) for all bodies in a [JrOrbitDriver] orbital system.
##
## Generated and maintained by [JrOrbitDriver] as a lightweight [RefCounted]
## object passed down to [JrOrbiter] position and orientation solvers. At zero
## rotation ([constant Vector3.ZERO]), the primary axis ([member vernal_point])
## points along [constant Vector3.FORWARD] ([code]-Z[/code]), the [member ecliptic_north_pole]
## points along [constant Vector3.UP] ([code]+Y[/code]), and the +90-degree in-plane
## progression direction points along [constant Vector3.RIGHT] ([code]+X[/code]) when
## [member positive_is_clockwise] is [code]true[/code] (or [constant Vector3.LEFT] when
## [code]false[/code]).

## Emitted when any orientation property of this reference direction changes.
signal changed

## Convenience property exposing [member reference_rotation] in degrees within
## the range [code](-180.0, 180.0][/code] per axis. Modifying this property updates
## [member reference_rotation] in radians.
var reference_rotation_deg: Vector3:
	get:
		return Vector3(
				rad_to_deg(reference_rotation.x),
				rad_to_deg(reference_rotation.y),
				rad_to_deg(reference_rotation.z)
		)
	set(value):
		reference_rotation = Vector3(
				deg_to_rad(value.x),
				deg_to_rad(value.y),
				deg_to_rad(value.z)
		)

## Euler rotation of the reference frame from the default orientation
## ([constant Vector3.FORWARD] vernal point, [constant Vector3.UP] ecliptic north pole),
## stored in radians and normalized to the range [code](-PI, PI][/code] per component.
var reference_rotation := Vector3.ZERO:
	set(value):
		var wrapped := Vector3(
				wrap_angle(value.x),
				wrap_angle(value.y),
				wrap_angle(value.z)
		)
		if not reference_rotation.is_equal_approx(wrapped):
			reference_rotation = wrapped
			_update_transform()

## Euler axis rotation order used when constructing the 3D orientation from
## [member reference_rotation].
var euler_order: EulerOrder = EULER_ORDER_YXZ:
	set(value):
		if euler_order != value:
			euler_order = value
			_update_transform()

## Defines the direction of positive angular progression in the ecliptic plane
## when viewed from the [member ecliptic_north_pole] ([code]+Y[/code]).
## If [code]true[/code], positive angular progression is clockwise
## ([code]FORWARD (-Z) -> RIGHT (+X) -> BACK (+Z) -> LEFT (-X)[/code]).
## If [code]false[/code], positive angular progression is counter-clockwise
## ([code]FORWARD (-Z) -> LEFT (-X) -> BACK (+Z) -> RIGHT (+X)[/code]).
var positive_is_clockwise: bool = true:
	set(value):
		if positive_is_clockwise != value:
			positive_is_clockwise = value
			_update_transform()

## If [code]true[/code], forces all orbits, ballistic trajectories, external thrust
## vectors, and body axial tilts in the system to lie flat on this ecliptic plane
## (allowing both flat prograde [code]0 deg[/code] and flat retrograde [code]180 deg[/code]
## motion).
var lock_to_ecliptic: bool = false:
	set(value):
		if lock_to_ecliptic != value:
			lock_to_ecliptic = value
			changed.emit()

## The unit direction vector of the vernal point (0 radians in the ecliptic plane),
## corresponding to [constant Vector3.FORWARD] rotated by [member reference_rotation].
var vernal_point: Vector3:
	get:
		return -_rotation_basis.z

## The unit normal vector of the ecliptic north pole, corresponding to
## [constant Vector3.UP] rotated by [member reference_rotation].
var ecliptic_north_pole: Vector3:
	get:
		return _rotation_basis.y

## The unit direction vector pointing toward +90 degrees of angular progression
## within the ecliptic plane.
var progression_axis: Vector3:
	get:
		return _transform.basis.x

## Read-only right-handed [Basis] computed directly from [member reference_rotation]
## and [member euler_order] without handedness reflection.
var rotation_basis: Basis:
	get:
		return _rotation_basis

## Read-only [Transform3D] that maps coordinates from canonical 3D ecliptic space
## into this reference frame. Its [code]-Z[/code] axis points along [member vernal_point],
## its [code]+Y[/code] axis points along [member ecliptic_north_pole], and its
## [code]+X[/code] axis points along [member progression_axis].
var orbit_transform: Transform3D:
	get:
		return _transform

## Pure right-handed rotation [Basis] computed from [member reference_rotation]
## and [member euler_order].
var _rotation_basis := Basis.IDENTITY

## Cached [Transform3D] representing the basis vectors of this reference frame,
## incorporating [member positive_is_clockwise] on the X (+90-degree progression) axis.
var _transform := Transform3D.IDENTITY


## Wraps an [param angle] in radians into the half-open interval [code](-PI, PI][/code].
static func wrap_angle(angle: float) -> float:
	return PI - fposmod(PI - angle, TAU)


## Initializes the 3D reference direction with an optional Euler rotation (in radians),
## Euler order, angular progression orientation, and ecliptic plane lock.
func _init(
		new_reference_rotation: Vector3 = Vector3.ZERO,
		new_euler_order: EulerOrder = EULER_ORDER_YXZ,
		new_positive_is_clockwise: bool = true,
		new_lock_to_ecliptic: bool = false
) -> void:
	configure(
			new_reference_rotation,
			new_euler_order,
			new_positive_is_clockwise,
			new_lock_to_ecliptic
	)


## Updates all orientation parameters of the reference direction in a single pass
## and emits [signal changed] if any value changed.
func configure(
		new_reference_rotation: Vector3,
		new_euler_order: EulerOrder,
		new_positive_is_clockwise: bool,
		new_lock_to_ecliptic: bool = false
) -> void:
	var wrapped_rot := Vector3(
			wrap_angle(new_reference_rotation.x),
			wrap_angle(new_reference_rotation.y),
			wrap_angle(new_reference_rotation.z)
	)
	if (
			reference_rotation.is_equal_approx(wrapped_rot)
			and euler_order == new_euler_order
			and positive_is_clockwise == new_positive_is_clockwise
			and lock_to_ecliptic == new_lock_to_ecliptic
	):
		return
	reference_rotation = wrapped_rot
	euler_order = new_euler_order
	positive_is_clockwise = new_positive_is_clockwise
	lock_to_ecliptic = new_lock_to_ecliptic
	_update_transform()


## Projects a 3D vector [param vec_in_driver_space] onto this ecliptic plane by removing
## any component parallel to [member ecliptic_north_pole].
func project_vector_to_ecliptic(vec_in_driver_space: Vector3) -> Vector3:
	if not lock_to_ecliptic:
		return vec_in_driver_space
	var pole: Vector3 = ecliptic_north_pole
	return vec_in_driver_space - pole * vec_in_driver_space.dot(pole)


## Snaps an orbital inclination [param inc] (in radians) to flat prograde ([code]0.0[/code])
## or flat retrograde ([code]PI[/code]) when [member lock_to_ecliptic] is [code]true[/code].
func snap_inclination(inc: float) -> float:
	if not lock_to_ecliptic:
		return inc
	return PI if absf(wrap_angle(inc)) > PI * 0.5 else 0.0


## Rebuilds the cached [member _rotation_basis] and [member _transform] from
## [member reference_rotation], [member euler_order], and [member positive_is_clockwise],
## then emits [signal changed].
func _update_transform() -> void:
	_rotation_basis = Basis.from_euler(reference_rotation, euler_order).orthonormalized()
	var x_axis: Vector3 = _rotation_basis.x if positive_is_clockwise else -_rotation_basis.x
	var y_axis: Vector3 = _rotation_basis.y
	var z_axis: Vector3 = _rotation_basis.z
	_transform = Transform3D(Basis(x_axis, y_axis, z_axis), Vector3.ZERO)
	changed.emit()
