#[compute]
#version 450

// Batched forward-looking Cubic Hermite trajectory-horizon baker for JrBallisticOrbiter nodes.
// Uses Universal-Variable Kepler Kick-Drift-Kick in anchor-relative coordinates,
// LCA tree-relative well difference walks, Battin's f(q) differential tides,
// ThrustFrame.ORBITAL prograde/normal/radial basis updates, and mid-horizon
// Hill-sphere anchor handoffs. Supports mixed full-horizon bakes and incremental
// tail-chunk extensions in a single dispatch via per-body start_t_offset and knot_dt.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct ForecastBodyInput {
	vec4 rel_pos_step_dt;     // xyz = start anchor-relative pos, w = knot_dt
	vec4 rel_vel_steps;       // xyz = start anchor-relative vel, w = step count (as float)
	vec4 ext_accel_t0;        // xyz = external thrust accel,     w = start_t_offset (from timeline t0)
	vec4 anchor_frame_flags;  // x = out_base_knot, y = anchor_well_idx, z = packed_mode_bits, w = substeps
};

// Binding 0: Per-body starting state and horizon job configuration (64 bytes per body).
layout(set = 0, binding = 0, std430) restrict readonly buffer ForecastBodyInputBuffer {
	ForecastBodyInput bodies[];
};

// Binding 1: Sampled gravity-well timeline ((timeline_segments + 1) * well_count * 2 vec4 entries).
// Entry (s * well_count + w) * 2 + 0: xyz = parent-relative well pos, w = mu.
// Entry (s * well_count + w) * 2 + 1: x = switch_entry_r, y = switch_exit_r, z = parent_well_idx, w = cull_threshold_sq.
layout(set = 0, binding = 1, std430) restrict readonly buffer WellTimelineBuffer {
	vec4 well_samples[];
};

// Binding 2: Contiguous per-body [pos_anchor[0..steps] | vel_flags[0..steps]] slices (2 vec4s per knot).
// For body with out_base_knot and pt_count = steps + 1:
//   forecast_knots[out_base_knot * 2 + k]            = vec4(r_rel, float(anchor_idx))
//   forecast_knots[out_base_knot * 2 + pt_count + k] = vec4(v_rel, unperturbed_flag)
layout(set = 0, binding = 2, std430) restrict writeonly buffer ForecastPointOutputBuffer {
	vec4 forecast_knots[];
};

// Binding 3: Optional resident GPU trajectory-tape atlas (272 vec4s per body slot:
// [0..135] pos_anchor, [136..271] vel_flags) populated directly when atlas_slot_plus_one > 0.
layout(set = 0, binding = 3, std430) restrict writeonly buffer TapeAtlasWriteBuffer {
	vec4 atlas_knots[];
};

layout(push_constant, std430) uniform ForecastParams {
	float inv_timeline_dt;
	uint body_count;
	uint well_count;
	uint timeline_segments;
} params;

const float TWO_PI = 6.28318530718;
const uint ATLAS_SLOT_VEC4S = 272u;
const uint ATLAS_VEL_OFFSET = 136u;

struct KnotWriteTarget {
	uint pos_base;
	uint vel_base;
	uint atlas_pos_base;
	bool write_atlas;
};

void write_knot(KnotWriteTarget dst, uint k_idx, vec4 pos_v4, vec4 vel_v4) {
	forecast_knots[dst.pos_base + k_idx] = pos_v4;
	forecast_knots[dst.vel_base + k_idx] = vel_v4;
	if (dst.write_atlas) {
		atlas_knots[dst.atlas_pos_base + k_idx] = pos_v4;
		atlas_knots[dst.atlas_pos_base + ATLAS_VEL_OFFSET + k_idx] = vel_v4;
	}
}

float stumpff_c(float z) {
	if (abs(z) < 1e-3) {
		return 0.5 - z * (1.0 / 24.0 - z * (1.0 / 720.0));
	}
	if (z > 0.0) {
		float sz = sqrt(z);
		return (1.0 - cos(sz)) / z;
	}
	float sz = sqrt(-z);
	return (cosh(sz) - 1.0) / (-z);
}

float stumpff_s(float z) {
	if (abs(z) < 1e-3) {
		return (1.0 / 6.0) - z * ((1.0 / 120.0) - z * (1.0 / 5040.0));
	}
	if (z > 0.0) {
		float sz = sqrt(z);
		return (sz - sin(sz)) / (z * sz);
	}
	float sz = sqrt(-z);
	return (sinh(sz) - sz) / ((-z) * sz);
}

void kepler_drift(inout vec3 r0, inout vec3 v0, float mu, float dt) {
	float r_scale = length(r0);
	if (r_scale <= 1e-9 || mu <= 0.0 || abs(dt) <= 1e-12) {
		r0 += v0 * dt;
		return;
	}

	float v_scale = sqrt(mu / r_scale);
	float inv_v_scale = 1.0 / v_scale;
	vec3 r_unit = r0 / r_scale;
	vec3 v_dim = v0 * inv_v_scale;
	float dt_dim = dt * (v_scale / r_scale);

	float sigma0 = dot(r_unit, v_dim);
	float alpha = 2.0 - dot(v_dim, v_dim);

	if (alpha > 1e-4) {
		float period_dim = TWO_PI / (alpha * sqrt(alpha));
		if (abs(dt_dim) > period_dim) {
			dt_dim = mod(dt_dim, period_dim);
		}
		if (dt_dim > 0.5 * period_dim) {
			dt_dim -= period_dim;
		} else if (dt_dim < -0.5 * period_dim) {
			dt_dim += period_dim;
		}
	}

	float chi = (alpha > 1e-4) ? (dt_dim * alpha) : dt_dim;
	chi = clamp(chi, -1e4, 1e4);

	for (int iter = 0; iter < 16; iter++) {
		float chi2 = chi * chi;
		float z = alpha * chi2;
		float cz = stumpff_c(z);
		float sz = stumpff_s(z);
		float chi3 = chi2 * chi;

		float f_val = sigma0 * chi2 * cz + (1.0 - alpha) * chi3 * sz + chi - dt_dim;
		float df = sigma0 * chi * (1.0 - z * sz) + (1.0 - alpha) * chi2 * cz + 1.0;
		float safe_df = (df >= 0.0) ? max(1e-6, df) : min(-1e-6, df);
		float delta = f_val / safe_df;
		chi -= delta;
		if (abs(delta) < 1e-6) {
			break;
		}
	}

	float chi2 = chi * chi;
	float z = alpha * chi2;
	float cz = stumpff_c(z);
	float sz = stumpff_s(z);
	float chi3 = chi2 * chi;

	float r_dim = max(1e-7, 1.0 + sigma0 * chi * (1.0 - z * sz) + (1.0 - alpha) * chi2 * cz);
	float f_lag = 1.0 - chi2 * cz;
	float g_lag = dt_dim - chi3 * sz;
	float df_lag = (chi * (z * sz - 1.0)) / r_dim;
	float dg_lag = 1.0 - (chi2 * cz) / r_dim;

	vec3 r1_dim = f_lag * r_unit + g_lag * v_dim;
	vec3 v1_dim = df_lag * r_unit + dg_lag * v_dim;
	r0 = r1_dim * r_scale;
	v0 = v1_dim * v_scale;
}

int get_well_parent(int well_idx) {
	if (well_idx < 0 || uint(well_idx) >= params.well_count) {
		return -1;
	}
	return int(round(well_samples[well_idx * 2 + 1].z));
}

bool is_descendant_of(int well_candidate, int ancestor_well) {
	if (ancestor_well < 0 || well_candidate <= ancestor_well) {
		return false;
	}
	int curr = well_candidate;
	int guard = 0;
	while (curr > ancestor_well && guard < 32) {
		guard++;
		curr = get_well_parent(curr);
		if (curr == ancestor_well) {
			return true;
		}
	}
	return false;
}

vec3 sample_well_local_pos(int well_idx, uint idx_a, uint idx_b, float frac) {
	if (well_idx < 0 || uint(well_idx) >= params.well_count) {
		return vec3(0.0);
	}
	uint base_a = (idx_a * params.well_count + uint(well_idx)) * 2u;
	uint base_b = (idx_b * params.well_count + uint(well_idx)) * 2u;
	return mix(well_samples[base_a].xyz, well_samples[base_b].xyz, frac);
}

vec3 sample_well_local_vel(int well_idx, uint idx_a, uint idx_b) {
	if (well_idx < 0 || uint(well_idx) >= params.well_count || idx_a == idx_b) {
		return vec3(0.0);
	}
	uint base_a = (idx_a * params.well_count + uint(well_idx)) * 2u;
	uint base_b = (idx_b * params.well_count + uint(well_idx)) * 2u;
	return (well_samples[base_b].xyz - well_samples[base_a].xyz) * params.inv_timeline_dt;
}

vec3 compute_timeline_well_difference(int well_a, int well_b, uint idx_a, uint idx_b, float frac) {
	int curr_a = well_a;
	int curr_b = well_b;
	vec3 diff = vec3(0.0);
	int guard = 0;
	while (curr_a != curr_b && guard < 32) {
		guard++;
		if (curr_a > curr_b) {
			if (curr_a < 0 || uint(curr_a) >= params.well_count) {
				break;
			}
			diff += sample_well_local_pos(curr_a, idx_a, idx_b, frac);
			curr_a = get_well_parent(curr_a);
		} else {
			if (curr_b < 0 || uint(curr_b) >= params.well_count) {
				break;
			}
			diff -= sample_well_local_pos(curr_b, idx_a, idx_b, frac);
			curr_b = get_well_parent(curr_b);
		}
	}
	return diff;
}

vec3 evaluate_thrust_acceleration(vec3 r_rel, vec3 v_rel, vec3 ext_accel, bool is_orbital_thrust) {
	if (dot(ext_accel, ext_accel) <= 0.0) {
		return vec3(0.0);
	}
	if (!is_orbital_thrust) {
		return ext_accel;
	}
	float v_len = length(v_rel);
	float r_len = length(r_rel);
	vec3 prograde = (v_len > 1e-8) ? (v_rel / v_len) : ((r_len > 1e-8) ? (r_rel / r_len) : vec3(0.0, 0.0, -1.0));
	vec3 h_vec = cross(r_rel, v_rel);
	float h_len = length(h_vec);
	vec3 orbit_normal = (h_len > 1e-8) ? (h_vec / h_len) : vec3(0.0, 1.0, 0.0);
	if (abs(dot(prograde, orbit_normal)) > 0.999) {
		vec3 fallback_up = (abs(prograde.y) > 0.9) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
		orbit_normal = normalize(cross(fallback_up, prograde));
	}
	vec3 z_back = -prograde;
	vec3 x_radial = normalize(cross(orbit_normal, z_back));
	vec3 y_normal = normalize(cross(z_back, x_radial));
	return x_radial * ext_accel.x + y_normal * ext_accel.y + z_back * ext_accel.z;
}

vec3 compute_timeline_perturbing_gravity(
	vec3 r_rel,
	int anchor_idx,
	float mu_anchor,
	float t_offset,
	bool enable_tidal,
	out bool any_pert_active
) {
	any_pert_active = false;
	if (params.well_count == 0u) {
		return vec3(0.0);
	}

	float max_seg = float(params.timeline_segments);
	float sample_coord = clamp(t_offset * params.inv_timeline_dt, 0.0, max_seg);
	uint idx_a = min(uint(floor(sample_coord)), params.timeline_segments);
	uint idx_b = min(idx_a + 1u, params.timeline_segments);
	float frac = sample_coord - float(idx_a);

	float cull_threshold_sq = well_samples[1].w;
	float r_rel_sq = dot(r_rel, r_rel);
	vec3 total_accel = vec3(0.0);

	for (uint k = 0u; k < params.well_count; k++) {
		int k_idx = int(k);
		if (k_idx == anchor_idx) {
			continue;
		}
		float mu_k = well_samples[k * 2u].w;
		if (mu_k <= 0.0) {
			continue;
		}

		vec3 d_vec = (anchor_idx >= 0)
			? compute_timeline_well_difference(anchor_idx, k_idx, idx_a, idx_b, frac)
			: -compute_timeline_well_difference(k_idx, 0, idx_a, idx_b, frac);
		float d_sq = dot(d_vec, d_vec);
		if (d_sq <= 1e-12) {
			continue;
		}

		if (mu_anchor <= 0.0) {
			if (!enable_tidal && !is_descendant_of(k_idx, anchor_idx)) {
				continue;
			}
			any_pert_active = true;
			vec3 rho = r_rel + d_vec;
			float rho_sq = dot(rho, rho);
			if (rho_sq > 1e-12) {
				float inv_rho = inversesqrt(rho_sq);
				total_accel -= rho * ((mu_k / rho_sq) * inv_rho);
			}
			continue;
		}

		if (!enable_tidal) {
			continue;
		}

		float r_over_d_sq = r_rel_sq / d_sq;
		float r_over_d_6 = r_over_d_sq * r_over_d_sq * r_over_d_sq;
		float mu_ratio = mu_k / mu_anchor;
		float tidal_ratio_sq = 4.0 * (mu_ratio * mu_ratio) * r_over_d_6;
		if (tidal_ratio_sq < cull_threshold_sq) {
			continue;
		}

		any_pert_active = true;
		vec3 rho = r_rel + d_vec;
		float rho_sq = dot(rho, rho);
		if (rho_sq <= 1e-12) {
			continue;
		}

		float inv_rho = inversesqrt(rho_sq);
		float rho_factor = (mu_k / rho_sq) * inv_rho;
		float q = dot(r_rel, r_rel + 2.0 * d_vec) / d_sq;

		if (abs(q) <= 0.5 && (1.0 + q) > 0.0) {
			float one_plus_q = 1.0 + q;
			float fq = q * (3.0 + 3.0 * q + q * q) / (1.0 + one_plus_q * sqrt(one_plus_q));
			total_accel -= rho_factor * (r_rel - fq * d_vec);
		} else {
			float inv_d = inversesqrt(d_sq);
			float d_factor = (mu_k / d_sq) * inv_d;
			total_accel -= (rho * rho_factor - d_vec * d_factor);
		}
	}

	return total_accel;
}

void maybe_switch_forecast_anchor(
	inout vec3 r_rel,
	inout vec3 v_rel,
	inout int anchor_idx,
	float t_offset
) {
	if (anchor_idx < 0 || uint(anchor_idx) >= params.well_count) {
		return;
	}
	float max_seg = float(params.timeline_segments);
	float sample_coord = clamp(t_offset * params.inv_timeline_dt, 0.0, max_seg);
	uint idx_a = min(uint(floor(sample_coord)), params.timeline_segments);
	uint idx_b = min(idx_a + 1u, params.timeline_segments);
	float frac = sample_coord - float(idx_a);

	// 1. Check exit from current anchor into parent anchor(s).
	for (int depth = 0; depth < 4; depth++) {
		if (anchor_idx <= 0 || uint(anchor_idx) >= params.well_count) {
			break;
		}
		vec4 anchor_meta = well_samples[(idx_a * params.well_count + uint(anchor_idx)) * 2u + 1u];
		float exit_r = anchor_meta.y;
		int parent_idx = int(round(anchor_meta.z));
		if (parent_idx >= 0 && exit_r > 0.0 && dot(r_rel, r_rel) > exit_r * exit_r) {
			r_rel += sample_well_local_pos(anchor_idx, idx_a, idx_b, frac);
			v_rel += sample_well_local_vel(anchor_idx, idx_a, idx_b);
			anchor_idx = parent_idx;
		} else {
			break;
		}
	}

	// 2. Check entry into a direct child well (and nested child wells) of the current anchor.
	for (int depth = 0; depth < 4; depth++) {
		bool entered_child = false;
		for (uint w = 0u; w < params.well_count; w++) {
			int w_idx = int(w);
			if (get_well_parent(w_idx) != anchor_idx) {
				continue;
			}
			float entry_r = well_samples[(idx_a * params.well_count + w) * 2u + 1u].x;
			if (entry_r <= 0.0) {
				continue;
			}
			vec3 child_local_pos = sample_well_local_pos(w_idx, idx_a, idx_b, frac);
			vec3 diff_to_child = r_rel - child_local_pos;
			if (dot(diff_to_child, diff_to_child) < entry_r * entry_r) {
				r_rel = diff_to_child;
				v_rel -= sample_well_local_vel(w_idx, idx_a, idx_b);
				anchor_idx = w_idx;
				entered_child = true;
				break;
			}
		}
		if (!entered_child) {
			break;
		}
	}
}

void main() {
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= params.body_count) {
		return;
	}

	ForecastBodyInput input_body = bodies[gid];
	vec3 r_rel = input_body.rel_pos_step_dt.xyz;
	float step_dt = max(0.0, input_body.rel_pos_step_dt.w);
	vec3 v_rel = input_body.rel_vel_steps.xyz;
	uint steps = max(1u, uint(round(input_body.rel_vel_steps.w)));
	vec3 ext_accel = input_body.ext_accel_t0.xyz;
	float elapsed = max(0.0, input_body.ext_accel_t0.w);

	uint out_knot_base = uint(round(input_body.anchor_frame_flags.x));
	int anchor_idx = int(round(input_body.anchor_frame_flags.y));
	uint mode_bits = uint(round(input_body.anchor_frame_flags.z));
	uint substeps = clamp(uint(round(input_body.anchor_frame_flags.w)), 1u, 64u);

	uint low_mode = mode_bits & 7u;
	uint atlas_slot_plus_one = mode_bits >> 3u;
	bool is_orbital_thrust = (low_mode & 1u) != 0u;
	bool enable_tidal = (low_mode & 2u) != 0u;
	bool is_pinned = (low_mode & 4u) != 0u;

	uint pt_count = steps + 1u;
	uint pos_base = out_knot_base * 2u;
	uint vel_base = pos_base + pt_count;
	bool write_atlas = (atlas_slot_plus_one > 0u) && (pt_count <= ATLAS_VEL_OFFSET);
	uint atlas_pos_base = write_atlas ? ((atlas_slot_plus_one - 1u) * ATLAS_SLOT_VEC4S) : 0u;
	KnotWriteTarget dst = KnotWriteTarget(pos_base, vel_base, atlas_pos_base, write_atlas);

	if (step_dt <= 0.0) {
		for (uint k = 0u; k <= steps; k++) {
			write_knot(dst, k, vec4(r_rel, float(anchor_idx)), vec4(v_rel, 0.0));
		}
		return;
	}

	float mu_anchor = (anchor_idx >= 0 && uint(anchor_idx) < params.well_count)
		? well_samples[uint(anchor_idx) * 2u].w
		: 0.0;
	bool has_thrust = dot(ext_accel, ext_accel) > 0.0;
	bool pert_active = false;
	vec3 pert_accel = compute_timeline_perturbing_gravity(
		r_rel,
		anchor_idx,
		mu_anchor,
		elapsed,
		enable_tidal,
		pert_active
	) + evaluate_thrust_acceleration(r_rel, v_rel, ext_accel, is_orbital_thrust);

	bool all_unperturbed = (!has_thrust) && (mu_anchor > 0.0) && (!pert_active);
	vec3 r_job0 = r_rel;
	vec3 v_job0 = v_rel;
	float elapsed0 = elapsed;
	write_knot(dst, 0u, vec4(r_rel, float(anchor_idx)), vec4(v_rel, all_unperturbed ? 1.0 : 0.0));

	float sub_dt = step_dt / float(substeps);

	for (uint k = 0u; k < steps; k++) {
		if (all_unperturbed) {
			vec3 r_cand = r_job0;
			vec3 v_cand = v_job0;
			float dt_from_job0 = float(k + 1u) * step_dt;
			kepler_drift(r_cand, v_cand, mu_anchor, dt_from_job0);
			float t_cand = elapsed0 + dt_from_job0;
			bool cand_pert = false;
			compute_timeline_perturbing_gravity(
				r_cand,
				anchor_idx,
				mu_anchor,
				t_cand,
				enable_tidal,
				cand_pert
			);
			if (!cand_pert) {
				int cand_anchor = anchor_idx;
				vec3 r_soi = r_cand;
				vec3 v_soi = v_cand;
				if (!is_pinned) {
					maybe_switch_forecast_anchor(r_soi, v_soi, cand_anchor, t_cand);
				}
				if (cand_anchor == anchor_idx) {
					r_rel = r_cand;
					v_rel = v_cand;
					elapsed = t_cand;
					write_knot(dst, k + 1u, vec4(r_rel, float(anchor_idx)), vec4(v_rel, 1.0));
					continue;
				}
			}
			all_unperturbed = false;
			pert_accel = vec3(0.0);
		}

		for (uint s = 0u; s < substeps; s++) {
			v_rel += 0.5 * sub_dt * pert_accel;
			kepler_drift(r_rel, v_rel, mu_anchor, sub_dt);
			elapsed += sub_dt;

			pert_accel = compute_timeline_perturbing_gravity(
				r_rel,
				anchor_idx,
				mu_anchor,
				elapsed,
				enable_tidal,
				pert_active
			) + evaluate_thrust_acceleration(r_rel, v_rel, ext_accel, is_orbital_thrust);
			if (pert_active) {
				all_unperturbed = false;
			}
			v_rel += 0.5 * sub_dt * pert_accel;
		}
		if (!is_pinned) {
			int prev_anchor = anchor_idx;
			maybe_switch_forecast_anchor(r_rel, v_rel, anchor_idx, elapsed);
			if (anchor_idx != prev_anchor) {
				all_unperturbed = false;
				mu_anchor = (anchor_idx >= 0 && uint(anchor_idx) < params.well_count)
					? well_samples[uint(anchor_idx) * 2u].w
					: 0.0;
				pert_accel = compute_timeline_perturbing_gravity(
					r_rel,
					anchor_idx,
					mu_anchor,
					elapsed,
					enable_tidal,
					pert_active
				) + evaluate_thrust_acceleration(r_rel, v_rel, ext_accel, is_orbital_thrust);
			}
		}
		write_knot(
			dst,
			k + 1u,
			vec4(r_rel, float(anchor_idx)),
			vec4(v_rel, all_unperturbed ? 1.0 : 0.0)
		);
	}
}
