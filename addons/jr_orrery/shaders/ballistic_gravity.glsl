#[compute]
#version 450

// Hybrid Universal-Variable Kepler Kick-Drift-Kick & Battin f(q) Tidal Integrator
// for JrBallisticOrbiter nodes.
//
// Operates in anchor-relative coordinates (r_rel = r - r_D, v_rel = v - v_D),
// resolves inter-well vectors via Lowest Common Ancestor (LCA) tree walks to
// prevent float32 catastrophic cancellation at 1+ AU, culls insignificant
// third-body tides without square roots, and supports Tier-1 epoch-frozen
// analytical coasting when unperturbed.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct BallisticBody {
	vec4 rel_pos_anchor;   // xyz = live anchor-relative pos, w = anchor_well_idx
	vec4 rel_vel_dt;       // xyz = live anchor-relative vel, w = unused
	vec4 ext_accel_frame;  // xyz = external thrust accel in driver units, w = signed frame (+unpinned/-pinned, 1=inertial, 2=orbital)
	vec4 rails_tidal_pad;  // x = is_on_rails (in/out), y = enable_tidal_forces, z = safe_max_radius, w = handoff_out
};

struct GravityWell {
	vec4 local_pos_start_mu;     // xyz = parent-relative pos at step start, w = mu
	vec4 local_pos_end_parent;   // xyz = parent-relative pos at step end,   w = parent_well_idx (-1 for root)
	vec4 hill_params;            // x = entry_radius_sq, y = exit_radius_sq, z = can_exit (1/0), w = safe_max_radius
};

struct BallisticEpoch {
	vec4 pos_hi;                 // xyz = frozen epoch anchor-relative pos, w = epoch_time_hi
	vec4 vel_lo;                 // xyz = frozen epoch anchor-relative vel, w = epoch_time_lo
};

layout(set = 0, binding = 0, std430) restrict buffer BallisticBodyBuffer {
	BallisticBody data[];
} bodies;

layout(set = 0, binding = 1, std430) restrict readonly buffer GravityWellBuffer {
	GravityWell data[];
} wells;

layout(set = 0, binding = 2, std430) restrict buffer BallisticEpochBuffer {
	BallisticEpoch data[];
} epochs;

layout(set = 0, binding = 3, std430) restrict buffer StepOutputBuffer {
	uint words[];
} step_out;

layout(push_constant, std430) uniform PushConstants {
	float cull_threshold_sq;
	uint body_count;
	uint well_count;
	uint max_substeps;
	float step_dt;
	float time_hi;
	float time_lo;
	uint events_word_offset;
} params;

const float TWO_PI = 6.28318530718;

// Evaluates Stumpff function C(z) = (1 - cos(sqrt(z))) / z.
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

// Evaluates Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / z^(3/2).
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

// Propagates (r0, v0) analytically over dt under central gravitational parameter mu
// using dimensionless Universal-Variable Newton-Raphson iteration.
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

// Returns true if well_candidate is a descendant of ancestor_well in the well hierarchy.
bool is_descendant_of(int well_candidate, int ancestor_well) {
	if (ancestor_well < 0 || well_candidate <= ancestor_well) {
		return false;
	}
	int curr = well_candidate;
	int guard = 0;
	while (curr > ancestor_well && guard < 32) {
		guard++;
		curr = int(round(wells.data[curr].local_pos_end_parent.w));
		if (curr == ancestor_well) {
			return true;
		}
	}
	return false;
}

// Computes (r_well_a - r_well_b) at normalized substep fraction alpha in [0, 1]
// via a two-pointer Lowest Common Ancestor (LCA) walk so shared ancestor offsets cancel.
vec3 compute_well_difference(int well_a, int well_b, float alpha) {
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
			vec3 local_a = mix(
				wells.data[curr_a].local_pos_start_mu.xyz,
				wells.data[curr_a].local_pos_end_parent.xyz,
				alpha
			);
			diff += local_a;
			curr_a = int(round(wells.data[curr_a].local_pos_end_parent.w));
		} else {
			if (curr_b < 0 || uint(curr_b) >= params.well_count) {
				break;
			}
			vec3 local_b = mix(
				wells.data[curr_b].local_pos_start_mu.xyz,
				wells.data[curr_b].local_pos_end_parent.xyz,
				alpha
			);
			diff -= local_b;
			curr_b = int(round(wells.data[curr_b].local_pos_end_parent.w));
		}
	}
	return diff;
}

// Evaluates external thrust acceleration in the active ThrustFrame.
// When thrust_frame == 2.0 (ORBITAL), constructs the instantaneous orthonormal
// RTN basis (-Z = Prograde, +Y = Normal, +X = Radial Outward).
vec3 evaluate_thrust_acceleration(vec3 r_rel, vec3 v_rel, vec3 ext_accel, float thrust_frame) {
	if (dot(ext_accel, ext_accel) <= 0.0) {
		return vec3(0.0);
	}
	if (thrust_frame < 1.5) {
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

// Evaluates Battin's f(q) third-body tidal accelerations (and direct gravity for children
// of a massless barycenter anchor), setting out_any_tide_active = true if any well is unculled.
vec3 compute_perturbing_gravity(
	vec3 r_rel,
	int anchor_idx,
	float mu_anchor,
	float alpha,
	bool enable_tidal,
	out bool out_any_tide_active
) {
	out_any_tide_active = false;
	if (params.well_count == 0u) {
		return vec3(0.0);
	}

	float r_rel_sq = dot(r_rel, r_rel);
	vec3 total_accel = vec3(0.0);

	for (uint k = 0u; k < params.well_count; k++) {
		int k_idx = int(k);
		if (k_idx == anchor_idx) {
			continue;
		}
		float mu_k = wells.data[k].local_pos_start_mu.w;
		if (mu_k <= 0.0) {
			continue;
		}

		// Vector d = r_anchor - r_k resolved via LCA walk.
		vec3 d_vec = (anchor_idx >= 0) ? compute_well_difference(anchor_idx, k_idx, alpha) : -compute_well_difference(k_idx, 0, alpha);
		float d_sq = dot(d_vec, d_vec);
		if (d_sq <= 1e-12) {
			continue;
		}

		// If the anchor is a massless barycenter (mu_anchor <= 0.0) and well k is a descendant
		// of that barycenter (e.g. binary stars), well k exerts direct Newtonian attraction.
		if (mu_anchor <= 0.0) {
			if (!enable_tidal && !is_descendant_of(k_idx, anchor_idx)) {
				continue;
			}
			out_any_tide_active = true;
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

		// Square-root-free tidal culling: 4 * (mu_k / mu_D)^2 * (r_rel^2 / d^2)^3 < cull_threshold_sq.
		float r_over_d_sq = r_rel_sq / d_sq;
		float r_over_d_6 = r_over_d_sq * r_over_d_sq * r_over_d_sq;
		float mu_ratio = mu_k / mu_anchor;
		float tidal_ratio_sq = 4.0 * (mu_ratio * mu_ratio) * r_over_d_6;
		if (tidal_ratio_sq < params.cull_threshold_sq) {
			continue;
		}

		out_any_tide_active = true;
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

// Evaluates Hill-sphere entry into direct massive child wells or exit to parent well
// when unpinned (pin_flag > -1.5). Returns target well index (>= 0.0) or -1.0 if unchanged.
float evaluate_hill_handoff(vec3 r_rel, int anchor_idx, float pin_flag) {
	if (pin_flag < -1.5 || anchor_idx < 0 || uint(anchor_idx) >= params.well_count) {
		return -1.0;
	}
	int best_child = -1;
	float best_dist_sq = 1e38;
	for (uint k = 1u; k < params.well_count; k++) {
		int parent_idx = int(round(wells.data[k].local_pos_end_parent.w));
		if (parent_idx != anchor_idx) {
			continue;
		}
		float entry_sq = wells.data[k].hill_params.x;
		if (entry_sq <= 0.0) {
			continue;
		}
		vec3 diff = r_rel - wells.data[k].local_pos_end_parent.xyz;
		float dist_sq = dot(diff, diff);
		if (dist_sq < entry_sq && dist_sq < best_dist_sq) {
			best_dist_sq = dist_sq;
			best_child = int(k);
		}
	}
	if (best_child >= 0) {
		return float(best_child);
	}
	if (anchor_idx > 0 && wells.data[anchor_idx].hill_params.z > 0.5) {
		float exit_sq = wells.data[anchor_idx].hill_params.y;
		if (dot(r_rel, r_rel) > exit_sq) {
			return wells.data[anchor_idx].local_pos_end_parent.w;
		}
	}
	return -1.0;
}

// Writes packed (pos, vel) to StepOutputBuffer and emits a sparse event only on rail/SOI transitions.
void write_step_output(uint idx, vec3 r_rel, vec3 v_rel, bool prev_on_rails, bool next_on_rails, float handoff_val) {
	uint pv_base = 6u + idx * 6u;
	step_out.words[pv_base + 0u] = floatBitsToUint(r_rel.x);
	step_out.words[pv_base + 1u] = floatBitsToUint(r_rel.y);
	step_out.words[pv_base + 2u] = floatBitsToUint(r_rel.z);
	step_out.words[pv_base + 3u] = floatBitsToUint(v_rel.x);
	step_out.words[pv_base + 4u] = floatBitsToUint(v_rel.y);
	step_out.words[pv_base + 5u] = floatBitsToUint(v_rel.z);

	int handoff_well = int(round(handoff_val));
	if ((next_on_rails != prev_on_rails) || (handoff_well >= 0)) {
		uint slot = atomicAdd(step_out.words[0], 1u);
		if (slot < params.body_count) {
			uint ev_base = params.events_word_offset + slot * 2u;
			uint rails_bit = next_on_rails ? 1u : 0u;
			uint well_plus_one = (handoff_well >= 0) ? uint(handoff_well + 1) : 0u;
			step_out.words[ev_base + 0u] = idx;
			step_out.words[ev_base + 1u] = rails_bit | (well_plus_one << 1u);
		}
	}
}

void main() {
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= params.body_count) {
		return;
	}

	vec3 r_rel = bodies.data[idx].rel_pos_anchor.xyz;
	int anchor_idx = int(round(bodies.data[idx].rel_pos_anchor.w));
	vec3 v_rel = bodies.data[idx].rel_vel_dt.xyz;
	bool is_on_rails = bodies.data[idx].rails_tidal_pad.x > 0.5;
	float dt = params.step_dt;
	vec3 ext_accel = bodies.data[idx].ext_accel_frame.xyz;
	float raw_frame_pin = bodies.data[idx].ext_accel_frame.w;
	float thrust_frame = abs(raw_frame_pin);
	float pin_flag = (raw_frame_pin < 0.0) ? -2.0 : -1.0;
	bool enable_tidal = bodies.data[idx].rails_tidal_pad.y > 0.5;
	float safe_max_radius = bodies.data[idx].rails_tidal_pad.z;

	float mu_anchor = 0.0;
	if (anchor_idx >= 0 && uint(anchor_idx) < params.well_count) {
		mu_anchor = wells.data[anchor_idx].local_pos_start_mu.w;
	}

	bool has_thrust = dot(ext_accel, ext_accel) > 0.0;

	// Tier-1 Epoch-Frozen On-Rails fast path: 1 analytical Kepler step from resident frozen epoch state.
	if (is_on_rails && !has_thrust && mu_anchor > 0.0) {
		vec4 ep_pos_hi = epochs.data[idx].pos_hi;
		vec4 ep_vel_lo = epochs.data[idx].vel_lo;
		r_rel = ep_pos_hi.xyz;
		v_rel = ep_vel_lo.xyz;
		float r_dt = (params.time_hi - ep_pos_hi.w) + (params.time_lo - ep_vel_lo.w);
		kepler_drift(r_rel, v_rel, mu_anchor, r_dt);
		bool wake_up = false;
		if (enable_tidal) {
			if (safe_max_radius > 0.0 && dot(r_rel, r_rel) > safe_max_radius * safe_max_radius) {
				wake_up = true;
			} else {
				bool any_tide = false;
				compute_perturbing_gravity(r_rel, anchor_idx, mu_anchor, 1.0, true, any_tide);
				wake_up = any_tide;
			}
		}
		bool stay_on_rails = !wake_up;
		float handoff_r = evaluate_hill_handoff(r_rel, anchor_idx, pin_flag);
		bodies.data[idx].rel_pos_anchor.xyz = r_rel;
		bodies.data[idx].rel_vel_dt.xyz = v_rel;
		bodies.data[idx].rails_tidal_pad.x = stay_on_rails ? 1.0 : 0.0;
		bodies.data[idx].rails_tidal_pad.w = handoff_r;
		write_step_output(idx, r_rel, v_rel, true, stay_on_rails, handoff_r);
		return;
	}

	// Tier-2 / Tier-3 Adaptive Universal-Variable Kick-Drift-Kick.
	uint max_steps = max(params.max_substeps, 1u);
	float r_len = max(1e-6, length(r_rel));
	float v_len = length(v_rel);
	float approx_dtheta = (v_len * abs(dt)) / r_len;
	uint step_count = clamp(uint(ceil(approx_dtheta / 0.12)), 1u, max_steps);
	if (has_thrust || mu_anchor <= 0.0) {
		step_count = max(step_count, min(max_steps, 4u));
	}

	float sub_dt = dt / float(step_count);
	float inv_steps = 1.0 / float(step_count);
	bool any_perturbation_ever = has_thrust || (mu_anchor <= 0.0);

	bool tide_start = false;
	vec3 tidal_accel = compute_perturbing_gravity(r_rel, anchor_idx, mu_anchor, 0.0, enable_tidal, tide_start);
	if (tide_start) {
		any_perturbation_ever = true;
	}
	vec3 thrust_accel = evaluate_thrust_acceleration(r_rel, v_rel, ext_accel, thrust_frame);
	vec3 total_pert = tidal_accel + thrust_accel;

	for (uint s = 0u; s < step_count; s++) {
		v_rel += 0.5 * sub_dt * total_pert;
		kepler_drift(r_rel, v_rel, mu_anchor, sub_dt);

		float alpha_end = float(s + 1u) * inv_steps;
		bool tide_end = false;
		tidal_accel = compute_perturbing_gravity(r_rel, anchor_idx, mu_anchor, alpha_end, enable_tidal, tide_end);
		if (tide_end) {
			any_perturbation_ever = true;
		}
		thrust_accel = evaluate_thrust_acceleration(r_rel, v_rel, ext_accel, thrust_frame);
		total_pert = tidal_accel + thrust_accel;
		v_rel += 0.5 * sub_dt * total_pert;
	}

	bool enter_rails = (!any_perturbation_ever && mu_anchor > 0.0);
	if (enter_rails) {
		epochs.data[idx].pos_hi = vec4(r_rel, params.time_hi);
		epochs.data[idx].vel_lo = vec4(v_rel, params.time_lo);
		bodies.data[idx].rails_tidal_pad.z = (anchor_idx >= 0 && uint(anchor_idx) < params.well_count)
			? wells.data[anchor_idx].hill_params.w
			: 0.0;
	}
	float handoff_kdk = evaluate_hill_handoff(r_rel, anchor_idx, pin_flag);
	bodies.data[idx].rel_pos_anchor.xyz = r_rel;
	bodies.data[idx].rel_vel_dt.xyz = v_rel;
	bodies.data[idx].rails_tidal_pad.x = enter_rails ? 1.0 : 0.0;
	bodies.data[idx].rails_tidal_pad.w = handoff_kdk;
	write_step_output(idx, r_rel, v_rel, is_on_rails, enter_rails, handoff_kdk);
}
