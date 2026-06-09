// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XSNN_INFERENCE_PROFILE_HLS_H
#define XSNN_INFERENCE_PROFILE_HLS_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xsnn_inference_profile_hls_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Ctrl_BaseAddress;
} XSnn_inference_profile_hls_Config;
#endif

typedef struct {
    u64 Ctrl_BaseAddress;
    u32 IsReady;
} XSnn_inference_profile_hls;

typedef u32 word_type;

typedef struct {
    u32 word_0;
    u32 word_1;
    u32 word_2;
    u32 word_3;
    u32 word_4;
} XSnn_inference_profile_hls_Learning_params;

typedef struct {
    u32 word_0;
    u32 word_1;
    u32 word_2;
} XSnn_inference_profile_hls_Encoder_config;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XSnn_inference_profile_hls_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XSnn_inference_profile_hls_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XSnn_inference_profile_hls_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XSnn_inference_profile_hls_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XSnn_inference_profile_hls_Initialize(XSnn_inference_profile_hls *InstancePtr, UINTPTR BaseAddress);
XSnn_inference_profile_hls_Config* XSnn_inference_profile_hls_LookupConfig(UINTPTR BaseAddress);
#else
int XSnn_inference_profile_hls_Initialize(XSnn_inference_profile_hls *InstancePtr, u16 DeviceId);
XSnn_inference_profile_hls_Config* XSnn_inference_profile_hls_LookupConfig(u16 DeviceId);
#endif
int XSnn_inference_profile_hls_CfgInitialize(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Config *ConfigPtr);
#else
int XSnn_inference_profile_hls_Initialize(XSnn_inference_profile_hls *InstancePtr, const char* InstanceName);
int XSnn_inference_profile_hls_Release(XSnn_inference_profile_hls *InstancePtr);
#endif

void XSnn_inference_profile_hls_Start(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_IsDone(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_IsIdle(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_IsReady(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_EnableAutoRestart(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_DisableAutoRestart(XSnn_inference_profile_hls *InstancePtr);

void XSnn_inference_profile_hls_Set_ctrl_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data);
u32 XSnn_inference_profile_hls_Get_ctrl_reg(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_config_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data);
u32 XSnn_inference_profile_hls_Get_config_reg(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_mode_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data);
u32 XSnn_inference_profile_hls_Get_mode_reg(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_time_steps_reg(XSnn_inference_profile_hls *InstancePtr, u32 Data);
u32 XSnn_inference_profile_hls_Get_time_steps_reg(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_learning_params(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Learning_params Data);
XSnn_inference_profile_hls_Learning_params XSnn_inference_profile_hls_Get_learning_params(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_encoder_config(XSnn_inference_profile_hls *InstancePtr, XSnn_inference_profile_hls_Encoder_config Data);
XSnn_inference_profile_hls_Encoder_config XSnn_inference_profile_hls_Get_encoder_config(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_status_reg(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_status_reg_vld(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_spike_count_reg(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_spike_count_reg_vld(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_weight_sum_reg(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_weight_sum_reg_vld(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_version_reg(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_Get_version_reg_vld(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_Set_reward_signal(XSnn_inference_profile_hls *InstancePtr, u32 Data);
u32 XSnn_inference_profile_hls_Get_reward_signal(XSnn_inference_profile_hls *InstancePtr);

void XSnn_inference_profile_hls_InterruptGlobalEnable(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_InterruptGlobalDisable(XSnn_inference_profile_hls *InstancePtr);
void XSnn_inference_profile_hls_InterruptEnable(XSnn_inference_profile_hls *InstancePtr, u32 Mask);
void XSnn_inference_profile_hls_InterruptDisable(XSnn_inference_profile_hls *InstancePtr, u32 Mask);
void XSnn_inference_profile_hls_InterruptClear(XSnn_inference_profile_hls *InstancePtr, u32 Mask);
u32 XSnn_inference_profile_hls_InterruptGetEnabled(XSnn_inference_profile_hls *InstancePtr);
u32 XSnn_inference_profile_hls_InterruptGetStatus(XSnn_inference_profile_hls *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
