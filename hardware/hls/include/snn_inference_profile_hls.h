//-----------------------------------------------------------------------------
// Lightweight inference/profile HLS top-level.
//
// This variant intentionally keeps the control-register argument order used by
// snn_top_hls so inference software can reuse the existing register map. It
// does not contain synaptic weights, learning state, encoder state, or weight
// checkpoint streams. Synaptic storage belongs to the RTL core groups and
// connectivity table in the hierarchical architecture.
//-----------------------------------------------------------------------------

#ifndef SNN_INFERENCE_PROFILE_HLS_H
#define SNN_INFERENCE_PROFILE_HLS_H

#include "snn_top_hls.h"

void snn_inference_profile_hls(
    ap_uint<32> ctrl_reg,
    ap_uint<32> config_reg,
    ap_uint<32> mode_reg,
    ap_uint<32> time_steps_reg,
    learning_params_t learning_params,
    encoder_config_t encoder_config,
    ap_uint<32> &status_reg,
    ap_uint<32> &spike_count_reg,
    ap_uint<32> &weight_sum_reg,
    ap_uint<32> &version_reg,

    hls::stream<axis_spike_t> &s_axis_spikes,
    hls::stream<axis_spike_t> &m_axis_spikes,

    ap_int<8> reward_signal,

    ap_uint<1> &spike_in_valid,
    rtl_nid_t &spike_in_neuron_id,
    ap_int<8> &spike_in_weight,
    ap_uint<1> spike_in_ready,

    ap_uint<1> spike_out_valid,
    rtl_nid_t spike_out_neuron_id,
    ap_int<8> spike_out_weight,
    ap_uint<1> &spike_out_ready,

    ap_uint<1> &snn_enable,
    ap_uint<1> &snn_reset,
    ap_uint<16> &threshold_out,
    ap_uint<16> &leak_rate_out,

    ap_uint<1> snn_ready,
    ap_uint<1> snn_busy
);

#endif
