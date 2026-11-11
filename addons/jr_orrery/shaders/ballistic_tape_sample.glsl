#[compute]
#version 450

// Parallel O(1) Cubic Hermite trajectory-tape sampler for JrBallisticOrbiter swarms.
// Evaluates anchor-relative positions from resident VRAM knot slots (written directly by
// ballistic_forecast.glsl) and emits sparse CPU fallback or gravity-well handoff events
// only when a body's active segment crosses a Sphere-of-Influence boundary or switches wells.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint ATLAS_SLOT_VEC4S = 272u;
const uint ATLAS_VEL_OFFSET = 136u;
const uint EVENT_CPU_FALLBACK = 0xFFFFFFFFu;

// Binding 0: Resident per-body knot atlas (272 vec4s per slot: [0..135] pos_anchor, [136..271] vel_flags).
layout(set = 0, binding = 0, std430) restrict readonly buffer TapeAtlasBuffer {
	vec4 atlas_knots[];
};

// Binding 1: Per-body tape metadata (2 vec4s = 32 bytes per body):
//   tape_meta[gid * 2 + 0] = (t0_rel_hi, t0_rel_lo, inv_knot_dt, knot_dt)
//   tape_meta[gid * 2 + 1] = (slot_idx, knot_count, loop_period, expected_well_idx)
layout(set = 0, binding = 1, std430) restrict readonly buffer TapeMetaBuffer {
	vec4 tape_meta[];
};

// Binding 2: Packed sampled-position & sparse event output buffer:
//   out_words[0]                            = event_count (atomic counter)
//   out_words[1..5]                         = 24-byte header padding (aligned to 2 x Vector3)
//   out_words[6 + gid * 3 + 0..2]           = floatBitsToUint(r_rel.xyz)
//   out_words[events_word_offset + e*2 + 0] = gid
//   out_words[events_word_offset + e*2 + 1] = ev_code (EVENT_CPU_FALLBACK or uint(new_well_idx))
layout(set = 0, binding = 2, std430) coherent buffer TapeSampleOutputBuffer {
	uint out_words[];
};

layout(push_constant, std430) uniform TapeSampleParams {
	float eval_rel_hi;
	float eval_rel_lo;
	uint body_count;
	uint events_word_offset;
} params;

void emit_sample_event(uint gid, uint code) {
	uint slot = atomicAdd(out_words[0], 1u);
	if (slot < params.body_count) {
		uint base = params.events_word_offset + slot * 2u;
		out_words[base] = gid;
		out_words[base + 1u] = code;
	}
}

void main() {
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= params.body_count) {
		return;
	}

	vec4 m0 = tape_meta[gid * 2u];
	vec4 m1 = tape_meta[gid * 2u + 1u];
	uint knot_count = uint(max(0.0, round(m1.y)));
	if (knot_count < 2u) {
		emit_sample_event(gid, EVENT_CPU_FALLBACK);
		return;
	}

	float rel_t = (params.eval_rel_hi - m0.x) + (params.eval_rel_lo - m0.y);
	float loop_period = m1.z;
	if (loop_period > 0.0) {
		rel_t = mod(rel_t, loop_period);
		if (rel_t < 0.0) {
			rel_t += loop_period;
		}
	}

	float max_idx = float(knot_count - 1u);
	float scaled = clamp(rel_t * m0.z, 0.0, max_idx);
	uint idx_a = min(uint(floor(scaled)), knot_count - 2u);
	float u = scaled - float(idx_a);

	uint slot_idx = uint(max(0.0, round(m1.x)));
	uint slot_base = slot_idx * ATLAS_SLOT_VEC4S;
	vec4 pa = atlas_knots[slot_base + idx_a];

	vec3 r_rel;
	int w_idx;
	if (u <= 1e-5) {
		r_rel = pa.xyz;
		w_idx = int(round(pa.w));
	} else {
		vec4 pb = atlas_knots[slot_base + idx_a + 1u];
		if (abs(pa.w - pb.w) > 0.1) {
			emit_sample_event(gid, EVENT_CPU_FALLBACK);
			return;
		}
		if (u >= 0.99999) {
			r_rel = pb.xyz;
			w_idx = int(round(pb.w));
		} else {
			float dt_k = m0.w;
			float u2 = u * u;
			float u3 = u2 * u;
			float h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
			float h10 = (u3 - 2.0 * u2 + u) * dt_k;
			float h01 = -2.0 * u3 + 3.0 * u2;
			float h11 = (u3 - u2) * dt_k;
			vec3 va = atlas_knots[slot_base + ATLAS_VEL_OFFSET + idx_a].xyz;
			vec3 vb = atlas_knots[slot_base + ATLAS_VEL_OFFSET + idx_a + 1u].xyz;
			r_rel = pa.xyz * h00 + va * h10 + pb.xyz * h01 + vb * h11;
			w_idx = int(round(pa.w));
		}
	}

	uint p_base = 6u + gid * 3u;
	out_words[p_base] = floatBitsToUint(r_rel.x);
	out_words[p_base + 1u] = floatBitsToUint(r_rel.y);
	out_words[p_base + 2u] = floatBitsToUint(r_rel.z);

	int expected_well = int(round(m1.w));
	if (w_idx != expected_well && w_idx >= 0) {
		emit_sample_event(gid, uint(w_idx));
	}
}
