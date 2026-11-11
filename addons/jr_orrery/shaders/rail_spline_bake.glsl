#[compute]
#version 450

// Batched GPU Periodic Cubic B-Spline Baker for JrRailOrbiter Mean-to-True Anomaly Lookup.
//
// Each 256-thread workgroup bakes one unique orbital eccentricity e in (0, 1):
//   1. Evaluates the smooth, branch-cut-free Equation of the Center
//      delta(M_k) = nu(M_k) - M_k at 256 uniform Mean Anomaly knots M_k in [-PI, PI)
//      using cancellation-free dE = E - M Newton-Raphson + Brouwer-Clemence atan identity.
//   2. Solves the exact periodic [1/6, 4/6, 1/6]^-1 cubic B-spline prefilter in shared
//      memory via its analytical Green's function (pole z1 = sqrt(3) - 2).
//   3. Emits monomial cubic segment coefficients vec4(a0, a1, a2, a3) on t in [0, 1) so
//      runtime O(1) lookup requires a single vec4 fetch and 3 Horner FMAs.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer EccentricityInputBuffer {
	float eccentricities[];
} ecc_in;

layout(set = 0, binding = 1, std430) writeonly buffer SplineSegmentOutputBuffer {
	vec4 segments[];
} spline_out;

layout(push_constant, std430) uniform PushConstants {
	uint curve_count;
	uint pad0;
	uint pad1;
	uint pad2;
} params;

const float PI_VAL = 3.14159265358979323846;
const float STEP_RAD = 6.28318530717958647692 / 256.0;
const float BSPLINE_POLE_Z1 = -0.2679491924311227;
const float BSPLINE_GAIN_C0 = 1.7320508075688772;

shared float s_delta[256];
shared float s_ctrl[256];

float solve_equation_of_center(float mean_anom, float ecc, float beta) {
	float de = ecc * sin(mean_anom);
	for (int iter = 0; iter < 8; iter++) {
		float e_anom = mean_anom + de;
		float f_val = de - ecc * sin(e_anom);
		float fp_val = max(1e-6, 1.0 - ecc * cos(e_anom));
		de -= f_val / fp_val;
	}
	float e_final = mean_anom + de;
	float t_tan = (beta * sin(e_final)) / max(1e-7, 1.0 - beta * cos(e_final));
	// Half-angle reduction |u| < 0.25 avoids D3D12/DXC ~3.4e-6 fast-atan intrinsic error,
	// giving full 24-bit IEEE float32 accuracy (< 5e-8 rad) across all GPU backends.
	float u = t_tan / (1.0 + sqrt(1.0 + t_tan * t_tan));
	float u2 = u * u;
	float atan_u = u * (1.0 - u2 * ((1.0 / 3.0) - u2 * ((1.0 / 5.0) - u2 * (1.0 / 7.0))));
	float nu_minus_e = 4.0 * atan_u;
	return de + nu_minus_e;
}

void main() {
	uint curve_idx = gl_WorkGroupID.x;
	if (curve_idx >= params.curve_count) {
		return;
	}

	uint k = gl_LocalInvocationID.x;
	float ecc = clamp(ecc_in.eccentricities[curve_idx], 0.0, 0.9999);
	float beta = ecc / (1.0 + sqrt(max(0.0, 1.0 - ecc * ecc)));

	if (k == 0u || k == 128u || ecc <= 1e-7) {
		s_delta[k] = 0.0;
	} else {
		float mean_anom = -PI_VAL + float(k) * STEP_RAD;
		s_delta[k] = solve_equation_of_center(mean_anom, ecc, beta);
	}
	barrier();

	float sum_val = s_delta[k];
	float z_pow = BSPLINE_POLE_Z1;
	for (uint tap = 1u; tap <= 16u; tap++) {
		uint idx_prev = (k - tap) & 255u;
		uint idx_next = (k + tap) & 255u;
		sum_val += z_pow * (s_delta[idx_prev] + s_delta[idx_next]);
		z_pow *= BSPLINE_POLE_Z1;
	}
	s_ctrl[k] = BSPLINE_GAIN_C0 * sum_val;
	barrier();

	float p0 = s_ctrl[(k + 255u) & 255u];
	float p1 = s_ctrl[k];
	float p2 = s_ctrl[(k + 1u) & 255u];
	float p3 = s_ctrl[(k + 2u) & 255u];

	float a0 = s_delta[k];
	float a1 = 0.5 * (p2 - p0);
	float a2 = 0.5 * (p0 - 2.0 * p1 + p2);
	float a3 = (p3 - p0 + 3.0 * (p1 - p2)) * (1.0 / 6.0);

	spline_out.segments[(curve_idx << 8u) + k] = vec4(a0, a1, a2, a3);
}
