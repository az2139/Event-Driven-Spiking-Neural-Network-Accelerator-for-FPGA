// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xsnn_inference_profile_hls.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XSnn_inference_profile_hls_CfgInitialize(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_BaseAddress = ConfigPtr->Ctrl_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XSnn_inference_profile_hls_Start(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL) & 0x80;
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XSnn_inference_profile_hls_IsDone(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XSnn_inference_profile_hls_IsIdle(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XSnn_inference_profile_hls_IsReady(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XSnn_inference_profile_hls_EnableAutoRestart(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL, 0x80);
}

void XSnn_inference_profile_hls_DisableAutoRestart(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_AP_CTRL, 0);
}

void XSnn_inference_profile_hls_Set_ctrl_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CTRL_REG_DATA, Data);
}

u32 XSnn_inference_profile_hls_Get_ctrl_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CTRL_REG_DATA);
    return Data;
}

void XSnn_inference_profile_hls_Set_config_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CONFIG_REG_DATA, Data);
}

u32 XSnn_inference_profile_hls_Get_config_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_CONFIG_REG_DATA);
    return Data;
}

void XSnn_inference_profile_hls_Set_mode_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_MODE_REG_DATA, Data);
}

u32 XSnn_inference_profile_hls_Get_mode_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_MODE_REG_DATA);
    return Data;
}

void XSnn_inference_profile_hls_Set_time_steps_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_TIME_STEPS_REG_DATA, Data);
}

u32 XSnn_inference_profile_hls_Get_time_steps_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_TIME_STEPS_REG_DATA);
    return Data;
}

void XSnn_inference_profile_hls_Set_learning_params(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Learning_params Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 0, Data.word_0);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 4, Data.word_1);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 8, Data.word_2);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 12, Data.word_3);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 16, Data.word_4);
}

XSnn_inference_profile_hls_Learning_params XSnn_inference_profile_hls_Get_learning_params(XSnn_inference_profile_hls *InstancePtr) {
    XSnn_inference_profile_hls_Learning_params Data;

    Data.word_0 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 0);
    Data.word_1 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 4);
    Data.word_2 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 8);
    Data.word_3 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 12);
    Data.word_4 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 16);
    return Data;
}

void XSnn_inference_profile_hls_Set_encoder_config(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Encoder_config Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 0, Data.word_0);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 4, Data.word_1);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 8, Data.word_2);
}

XSnn_inference_profile_hls_Encoder_config XSnn_inference_profile_hls_Get_encoder_config(XSnn_inference_profile_hls *InstancePtr) {
    XSnn_inference_profile_hls_Encoder_config Data;

    Data.word_0 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 0);
    Data.word_1 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 4);
    Data.word_2 = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ENCODER_CONFIG_DATA + 8);
    return Data;
}

u32 XSnn_inference_profile_hls_Get_status_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_STATUS_REG_DATA);
    return Data;
}

u32 XSnn_inference_profile_hls_Get_status_reg_vld(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_STATUS_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_inference_profile_hls_Get_spike_count_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_SPIKE_COUNT_REG_DATA);
    return Data;
}

u32 XSnn_inference_profile_hls_Get_spike_count_reg_vld(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_SPIKE_COUNT_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_inference_profile_hls_Get_weight_sum_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_WEIGHT_SUM_REG_DATA);
    return Data;
}

u32 XSnn_inference_profile_hls_Get_weight_sum_reg_vld(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_WEIGHT_SUM_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_inference_profile_hls_Get_version_reg(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_VERSION_REG_DATA);
    return Data;
}

u32 XSnn_inference_profile_hls_Get_version_reg_vld(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_VERSION_REG_CTRL);
    return Data & 0x1;
}

void XSnn_inference_profile_hls_Set_reward_signal(XSnn_inference_profile_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_REWARD_SIGNAL_DATA, Data);
}

u32 XSnn_inference_profile_hls_Get_reward_signal(XSnn_inference_profile_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_REWARD_SIGNAL_DATA);
    return Data;
}

void XSnn_inference_profile_hls_InterruptGlobalEnable(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_GIE, 1);
}

void XSnn_inference_profile_hls_InterruptGlobalDisable(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_GIE, 0);
}

void XSnn_inference_profile_hls_InterruptEnable(XSnn_inference_profile_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER, Register | Mask);
}

void XSnn_inference_profile_hls_InterruptDisable(XSnn_inference_profile_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER);
    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER, Register & (~Mask));
}

void XSnn_inference_profile_hls_InterruptClear(XSnn_inference_profile_hls *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_inference_profile_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ISR, Mask);
}

u32 XSnn_inference_profile_hls_InterruptGetEnabled(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_IER);
}

u32 XSnn_inference_profile_hls_InterruptGetStatus(XSnn_inference_profile_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XSnn_inference_profile_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_INFERENCE_PROFILE_HLS_CTRL_ADDR_ISR);
}

