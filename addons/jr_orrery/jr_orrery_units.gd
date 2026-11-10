@tool
class_name JrOrreryUnits
extends RefCounted
## Static utility class defining physical unit enumerations and 64-bit SI conversion
## functions for [JrOrbitDriver], [JrRailOrbiter], and [JrBallisticOrbiter].
##
## Converts physical masses, distances, times, and accelerations into the owning
## [JrOrbitDriver]'s chosen length and time units while preserving 64-bit floating-point
## precision on the CPU prior to GPU compute dispatch.

## Physical distance units supported by [JrOrbitDriver].
enum DistanceUnit {
	## SI meters (1.0 m).
	METERS = 0,
	## Kilometers (1,000.0 m).
	KILOMETERS = 1,
	## Megameters (1,000,000.0 m).
	MEGAMETERS = 2,
	## Nominal Earth equatorial/mean volumetric radius (6,371,000.0 m).
	EARTH_RADII = 3,
	## Mean Lunar semi-major axis distance (384,399,000.0 m).
	LUNAR_DISTANCES = 4,
	## IAU 2012 exact Astronomical Unit (149,597,870,700.0 m).
	ASTRONOMICAL_UNITS = 5,
}

## Physical time units supported by [JrOrbitDriver].
enum TimeUnit {
	## SI seconds (1.0 s).
	SECONDS = 0,
	## Minutes (60.0 s).
	MINUTES = 1,
	## Hours (3,600.0 s).
	HOURS = 2,
	## Julian ephemeris days (86,400.0 s).
	DAYS = 3,
	## Julian years of 365.25 days (31,557,600.0 s).
	YEARS = 4,
}

## Physical mass units supported by [JrOrbitDriver] and [JrRailOrbiter].
enum MassUnit {
	## Direct gravitational parameter mu = G * M in driver units (dist_unit^3 / time_unit^2).
	DRIVER_GM = 0,
	## SI kilograms (1.0 kg).
	KILOGRAMS = 1,
	## Metric tons / megagrams (1,000.0 kg).
	METRIC_TONS = 2,
	## Teragrams (1e9 kg); 1 Tg produces mu ~ 6.6743e-2 m^3/s^2.
	TERAGRAMS = 3,
	## Lunar masses (~7.342e22 kg, IAU GM = 4.902800066e12 m^3/s^2).
	LUNAR_MASSES = 4,
	## Earth masses (~5.972168e24 kg, IAU nominal GM = 3.986004418e14 m^3/s^2).
	EARTH_MASSES = 5,
	## Jupiter masses (~1.898125e27 kg, IAU nominal GM = 1.266865319e17 m^3/s^2).
	JUPITER_MASSES = 6,
	## Solar masses (~1.988416e30 kg, IAU nominal GM = 1.32712440041279419e20 m^3/s^2).
	SOLAR_MASSES = 7,
}

## Physical acceleration units supported by [JrBallisticOrbiter] for continuous
## external thrust vectors.
enum AccelerationUnit {
	## Raw driver units (distance_unit / time_unit^2) without additional scaling.
	DRIVER_UNITS = 0,
	## SI meters per second squared (1.0 m/s^2).
	METERS_PER_SECOND_SQUARED = 1,
	## Standard Earth surface gravity g_0 (9.80665 m/s^2).
	STANDARD_GRAVITIES_G = 2,
}

## CODATA 2018 / IAU gravitational constant G in m^3 kg^-1 s^-2.
const G_SI: float = 6.67430e-11

## Standard Earth gravitational acceleration g_0 in m/s^2.
const STANDARD_GRAVITY_SI: float = 9.80665

## IAU 2015 nominal solar mass parameter GM_Sun in m^3/s^2.
const GM_SOLAR_SI: float = 1.32712440041279419e20

## IAU 2015 nominal Jovian mass parameter GM_Jupiter in m^3/s^2.
const GM_JUPITER_SI: float = 1.26686531900e17

## IAU 2015 nominal terrestrial mass parameter GM_Earth in m^3/s^2.
const GM_EARTH_SI: float = 3.986004418e14

## DE440 / IAU lunar mass parameter GM_Moon in m^3/s^2.
const GM_LUNAR_SI: float = 4.902800066e12


## Returns the length in SI meters corresponding to one unit of [param dist_unit].
static func get_distance_scale_meters(dist_unit: DistanceUnit) -> float:
	match dist_unit:
		DistanceUnit.METERS:
			return 1.0
		DistanceUnit.KILOMETERS:
			return 1.0e3
		DistanceUnit.MEGAMETERS:
			return 1.0e6
		DistanceUnit.EARTH_RADII:
			return 6.371e6
		DistanceUnit.LUNAR_DISTANCES:
			return 3.84399e8
		DistanceUnit.ASTRONOMICAL_UNITS:
			return 149_597_870_700.0
	return 1.0


## Returns the duration in SI seconds corresponding to one unit of [param time_unit].
static func get_time_scale_seconds(time_unit: TimeUnit) -> float:
	match time_unit:
		TimeUnit.SECONDS:
			return 1.0
		TimeUnit.MINUTES:
			return 60.0
		TimeUnit.HOURS:
			return 3600.0
		TimeUnit.DAYS:
			return 86400.0
		TimeUnit.YEARS:
			return 31_557_600.0
	return 1.0


## Returns the mass in SI kilograms corresponding to one unit of [param mass_unit].
static func get_mass_scale_kilograms(mass_unit: MassUnit) -> float:
	match mass_unit:
		MassUnit.DRIVER_GM:
			return 1.0 / G_SI
		MassUnit.KILOGRAMS:
			return 1.0
		MassUnit.METRIC_TONS:
			return 1.0e3
		MassUnit.TERAGRAMS:
			return 1.0e9
		MassUnit.LUNAR_MASSES:
			return GM_LUNAR_SI / G_SI
		MassUnit.EARTH_MASSES:
			return GM_EARTH_SI / G_SI
		MassUnit.JUPITER_MASSES:
			return GM_JUPITER_SI / G_SI
		MassUnit.SOLAR_MASSES:
			return GM_SOLAR_SI / G_SI
	return 1.0


## Returns the SI standard gravitational parameter GM (in m^3/s^2) for one unit
## of [param mass_unit], using exact IAU nominal GM constants for planetary/stellar units.
static func get_unit_gm_si(mass_unit: MassUnit) -> float:
	match mass_unit:
		MassUnit.DRIVER_GM:
			return 1.0
		MassUnit.KILOGRAMS:
			return G_SI
		MassUnit.METRIC_TONS:
			return G_SI * 1.0e3
		MassUnit.TERAGRAMS:
			return G_SI * 1.0e9
		MassUnit.LUNAR_MASSES:
			return GM_LUNAR_SI
		MassUnit.EARTH_MASSES:
			return GM_EARTH_SI
		MassUnit.JUPITER_MASSES:
			return GM_JUPITER_SI
		MassUnit.SOLAR_MASSES:
			return GM_SOLAR_SI
	return G_SI


## Computes the standard gravitational parameter mu = GM in SI units (m^3/s^2)
## for [param mass_val] expressed in [param mass_unit].
static func compute_mu_si(mass_val: float, mass_unit: MassUnit) -> float:
	if mass_val <= 0.0:
		return 0.0
	return mass_val * get_unit_gm_si(mass_unit)


## Returns the multiplicative factor (time_scale_s^2 / dist_scale_m^3) that converts
## an SI gravitational parameter (m^3/s^2) into driver units (dist_unit^3 / time_unit^2).
static func get_si_to_driver_mu_factor(dist_unit: DistanceUnit, time_unit: TimeUnit) -> float:
	var dist_m: float = get_distance_scale_meters(dist_unit)
	var time_s: float = get_time_scale_seconds(time_unit)
	var ratio: float = time_s / dist_m
	return (ratio * ratio) / dist_m


## Computes the gravitational parameter mu in driver units (dist_unit^3 / time_unit^2)
## for [param mass_val] expressed in [param mass_unit].
static func compute_mu_in_driver_units(
		mass_val: float,
		mass_unit: MassUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	if mass_val <= 0.0:
		return 0.0
	if mass_unit == MassUnit.DRIVER_GM:
		return mass_val
	var mu_si: float = compute_mu_si(mass_val, mass_unit)
	return mu_si * get_si_to_driver_mu_factor(dist_unit, time_unit)


## Alias for [method compute_mu_in_driver_units].
static func compute_mu(
		mass_val: float,
		mass_unit: MassUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	return compute_mu_in_driver_units(mass_val, mass_unit, dist_unit, time_unit)


## Alias for [method compute_mu_in_driver_units].
static func convert_mass_to_mu(
		mass_val: float,
		mass_unit: MassUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	return compute_mu_in_driver_units(mass_val, mass_unit, dist_unit, time_unit)


## Converts a gravitational parameter [param mu_driver] (in dist_unit^3 / time_unit^2)
## back into a mass value in [param mass_unit].
static func mass_from_mu_in_driver_units(
		mu_driver: float,
		mass_unit: MassUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	if mu_driver <= 0.0:
		return 0.0
	if mass_unit == MassUnit.DRIVER_GM:
		return mu_driver
	var factor: float = get_si_to_driver_mu_factor(dist_unit, time_unit)
	var unit_gm: float = get_unit_gm_si(mass_unit)
	var denom: float = factor * unit_gm
	if denom <= 0.0:
		return 0.0
	return mu_driver / denom


## Alias for [method mass_from_mu_in_driver_units].
static func mu_to_mass(
		mu_driver: float,
		mass_unit: MassUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	return mass_from_mu_in_driver_units(mu_driver, mass_unit, dist_unit, time_unit)


## Returns the multiplicative scale factor that converts an acceleration in
## [param accel_unit] into driver units (dist_unit / time_unit^2).
static func get_acceleration_scale_to_driver(
		accel_unit: AccelerationUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> float:
	if accel_unit == AccelerationUnit.DRIVER_UNITS:
		return 1.0
	var dist_m: float = get_distance_scale_meters(dist_unit)
	var time_s: float = get_time_scale_seconds(time_unit)
	var meters_to_driver_accel: float = (time_s * time_s) / dist_m
	if accel_unit == AccelerationUnit.STANDARD_GRAVITIES_G:
		return STANDARD_GRAVITY_SI * meters_to_driver_accel
	return meters_to_driver_accel


## Converts an acceleration vector [param accel] from [param accel_unit] into
## driver units (dist_unit / time_unit^2).
static func convert_acceleration_to_driver_units(
		accel: Vector3,
		accel_unit: AccelerationUnit,
		dist_unit: DistanceUnit,
		time_unit: TimeUnit
) -> Vector3:
	if accel == Vector3.ZERO or accel_unit == AccelerationUnit.DRIVER_UNITS:
		return accel
	return accel * get_acceleration_scale_to_driver(accel_unit, dist_unit, time_unit)
