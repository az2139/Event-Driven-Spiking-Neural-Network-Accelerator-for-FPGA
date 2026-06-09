// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// ctrl
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of ctrl_reg
//        bit 31~0 - ctrl_reg[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of config_reg
//        bit 31~0 - config_reg[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of mode_reg
//        bit 31~0 - mode_reg[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of time_steps_reg
//        bit 31~0 - time_steps_reg[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of learning_params
//        bit 31~0 - learning_params[31:0] (Read/Write)
// 0x34 : Data signal of learning_params
//        bit 31~0 - learning_params[63:32] (Read/Write)
// 0x38 : Data signal of learning_params
//        bit 31~0 - learning_params[95:64] (Read/Write)
// 0x3c : Data signal of learning_params
//        bit 31~0 - learning_params[127:96] (Read/Write)
// 0x40 : Data signal of learning_params
//        bit 15~0 - learning_params[143:128] (Read/Write)
//        others   - reserved
// 0x44 : reserved
// 0x48 : Data signal of encoder_config
//        bit 31~0 - encoder_config[31:0] (Read/Write)
// 0x4c : Data signal of encoder_config
//        bit 31~0 - encoder_config[63:32] (Read/Write)
// 0x50 : Data signal of encoder_config
//        bit 15~0 - encoder_config[79:64] (Read/Write)
//        others   - reserved
// 0x54 : reserved
// 0x58 : Data signal of status_reg
//        bit 31~0 - status_reg[31:0] (Read)
// 0x5c : Control signal of status_reg
//        bit 0  - status_reg_ap_vld (Read/COR)
//        others - reserved
// 0x68 : Data signal of spike_count_reg
//        bit 31~0 - spike_count_reg[31:0] (Read)
// 0x6c : Control signal of spike_count_reg
//        bit 0  - spike_count_reg_ap_vld (Read/COR)
//        others - reserved
// 0x78 : Data signal of weight_sum_reg
//        bit 31~0 - weight_sum_reg[31:0] (Read)
// 0x7c : Control signal of weight_sum_reg
//        bit 0  - weight_sum_reg_ap_vld (Read/COR)
//        others - reserved
// 0x88 : Data signal of version_reg
//        bit 31~0 - version_reg[31:0] (Read)
// 0x8c : Control signal of version_reg
//        bit 0  - version_reg_ap_vld (Read/COR)
//        others - reserved
// 0x98 : Data signal of reward_signal
//        bit 7~0 - reward_signal[7:0] (Read/Write)
//        others  - reserved
// 0x9c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL              0x00
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_GIE                  0x04
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER                  0x08
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ISR                  0x0c
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CTRL_REG_DATA        0x10
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_CTRL_REG_DATA        32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CONFIG_REG_DATA      0x18
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_CONFIG_REG_DATA      32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_MODE_REG_DATA        0x20
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_MODE_REG_DATA        32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_TIME_STEPS_REG_DATA  0x28
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_TIME_STEPS_REG_DATA  32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA 0x30
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_LEARNING_PARAMS_DATA 144
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA  0x48
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_ENCODER_CONFIG_DATA  80
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_STATUS_REG_DATA      0x58
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_STATUS_REG_DATA      32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_STATUS_REG_CTRL      0x5c
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_SPIKE_COUNT_REG_DATA 0x68
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_SPIKE_COUNT_REG_DATA 32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_SPIKE_COUNT_REG_CTRL 0x6c
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_WEIGHT_SUM_REG_DATA  0x78
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_WEIGHT_SUM_REG_DATA  32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_WEIGHT_SUM_REG_CTRL  0x7c
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_VERSION_REG_DATA     0x88
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_VERSION_REG_DATA     32
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_VERSION_REG_CTRL     0x8c
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_REWARD_SIGNAL_DATA   0x98
#define XSNN_INFERENCE_PROFILE_HLS_CTRL_BITS_REWARD_SIGNAL_DATA   8

