//-----------------------------------------------------------------------------
// Lightweight inference/profile HLS bridge for the hierarchical RTL SNN.
//-----------------------------------------------------------------------------

#include "../include/snn_inference_profile_hls.h"

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
) {
    #pragma HLS INTERFACE s_axilite port=ctrl_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=mode_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=time_steps_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=learning_params bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=encoder_config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=spike_count_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=weight_sum_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=version_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reward_signal bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    #pragma HLS INTERFACE axis port=s_axis_spikes
    #pragma HLS INTERFACE axis port=m_axis_spikes

    #pragma HLS INTERFACE ap_none port=spike_in_valid
    #pragma HLS INTERFACE ap_none port=spike_in_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_in_weight
    #pragma HLS INTERFACE ap_none port=spike_in_ready
    #pragma HLS INTERFACE ap_none port=spike_out_valid
    #pragma HLS INTERFACE ap_none port=spike_out_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_out_weight
    #pragma HLS INTERFACE ap_none port=spike_out_ready
    #pragma HLS INTERFACE ap_none port=snn_enable
    #pragma HLS INTERFACE ap_none port=snn_reset
    #pragma HLS INTERFACE ap_none port=threshold_out
    #pragma HLS INTERFACE ap_none port=leak_rate_out
    #pragma HLS INTERFACE ap_none port=snn_ready
    #pragma HLS INTERFACE ap_none port=snn_busy

    static ap_uint<32> timestamp = 0;
    static ap_uint<32> spike_counter = 0;
    static bool first_spike_sent = false;
    static bool first_spike_pending = false;
    static neuron_id_t first_spike_pending_id = 0;
    static weight_t first_spike_pending_weight = 0;
    static ap_uint<1> spike_in_valid_toggle = 0;
    static ap_uint<1> spike_out_ack_toggle = 0;

    bool enable = ctrl_reg[0];
    bool reset = ctrl_reg[1];
    bool clear_counters = ctrl_reg[2];
    bool first_spike_only = ctrl_reg[7];
    ap_uint<16> time_steps =
        (time_steps_reg == 0) ? (ap_uint<16>)1 : (ap_uint<16>)time_steps_reg;

    // Preserve the legacy AXI-Lite register layout. These controls are
    // intentionally unsupported in the inference-only implementation.
    (void)learning_params;
    (void)encoder_config;
    (void)reward_signal;

    if (reset) {
        timestamp = 0;
        spike_counter = 0;
        first_spike_sent = false;
        first_spike_pending = false;
        first_spike_pending_id = 0;
        first_spike_pending_weight = 0;
        spike_in_valid_toggle = 0;
        spike_out_ack_toggle = 0;
    }

    if (clear_counters) {
        spike_counter = 0;
    }

    if (!enable) {
        first_spike_sent = false;
        first_spike_pending = false;
        first_spike_pending_id = 0;
        first_spike_pending_weight = 0;
    }

    snn_enable = enable;
    snn_reset = reset;
    threshold_out = config_reg(15, 0);
    leak_rate_out = config_reg(31, 16);

    TIME_LOOP: for (ap_uint<16> t = 0; t < time_steps; t++) {
        #pragma HLS LOOP_FLATTEN off

        spike_in_valid = spike_in_valid_toggle;
        spike_in_neuron_id = 0;
        spike_in_weight = 0;

        if (enable && spike_in_ready && !s_axis_spikes.empty()) {
            axis_spike_t in_pkt = s_axis_spikes.read();
            neuron_id_t pre_id = in_pkt.data(SPIKE_PKT_ID_HI, SPIKE_PKT_ID_LO);
            weight_t weight = (weight_t)in_pkt.data(SPIKE_PKT_WGT_HI, SPIKE_PKT_WGT_LO);

            spike_in_valid_toggle = (ap_uint<1>)(!spike_in_valid_toggle);
            spike_in_valid = spike_in_valid_toggle;
            spike_in_neuron_id = (rtl_nid_t)pre_id;
            spike_in_weight = weight;
            spike_counter++;
        }

        spike_out_ready = spike_out_ack_toggle;

        if (enable && first_spike_only && first_spike_pending && !m_axis_spikes.full()) {
            axis_spike_t out_pkt;
            out_pkt.data = 0;
            out_pkt.data(SPIKE_PKT_ID_HI, SPIKE_PKT_ID_LO) = first_spike_pending_id;
            out_pkt.data(SPIKE_PKT_WGT_HI, SPIKE_PKT_WGT_LO) =
                (ap_uint<8>)first_spike_pending_weight;
            out_pkt.data(SPIKE_PKT_TS_HI, SPIKE_PKT_TS_LO) =
                timestamp(SPIKE_PKT_TS_HI - SPIKE_PKT_TS_LO, 0);
            out_pkt.keep = 0xF;
            out_pkt.strb = 0xF;
            out_pkt.last = 1;
            out_pkt.id = 0;
            out_pkt.dest = 0;
            out_pkt.user = 0;
            m_axis_spikes.write(out_pkt);
            first_spike_pending = false;
        }

        if (enable && spike_out_valid) {
            neuron_id_t post_id = (neuron_id_t)spike_out_neuron_id;
            weight_t weight = spike_out_weight;
            bool consume_post_spike = false;

            if (first_spike_only) {
                if (!first_spike_sent && !first_spike_pending) {
                    first_spike_pending_id = post_id;
                    first_spike_pending_weight = weight;
                    first_spike_pending = true;
                    first_spike_sent = true;
                }
                consume_post_spike = true;
            } else if (!m_axis_spikes.full()) {
                axis_spike_t out_pkt;
                out_pkt.data = 0;
                out_pkt.data(SPIKE_PKT_ID_HI, SPIKE_PKT_ID_LO) = post_id;
                out_pkt.data(SPIKE_PKT_WGT_HI, SPIKE_PKT_WGT_LO) = (ap_uint<8>)weight;
                out_pkt.data(SPIKE_PKT_TS_HI, SPIKE_PKT_TS_LO) =
                    timestamp(SPIKE_PKT_TS_HI - SPIKE_PKT_TS_LO, 0);
                out_pkt.keep = 0xF;
                out_pkt.strb = 0xF;
                out_pkt.last = 1;
                out_pkt.id = 0;
                out_pkt.dest = 0;
                out_pkt.user = 0;
                m_axis_spikes.write(out_pkt);
                consume_post_spike = true;
            }

            if (consume_post_spike) {
                spike_out_ack_toggle = (ap_uint<1>)(!spike_out_ack_toggle);
                spike_out_ready = spike_out_ack_toggle;
            }
        }

        if (enable) {
            timestamp++;
        }
    }

    ap_uint<32> status = 0;
    status[0] = snn_ready;
    status[1] = snn_busy;
    status[3] = first_spike_only;
    status(7, 6) = mode_reg(1, 0);
    status[16] = first_spike_pending;

    status_reg = status;
    spike_count_reg = spike_counter;
    weight_sum_reg = 0;
    version_reg = VERSION_ID;
}
