# Deep-Convolutional-Network-on-FPGA
a simplified version of LeNet5

## Double Precision
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|resource utilization|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|:------------------:|
|CNN_FW_Conv_V0_DP_L0_0|4,opt,Failed|y|y|y| | | |
|CNN_FW_Conv_V0_DP_L0_0|4,no opt,Failed|y|y|y| | | |
|CNN_FW_Conv_V0_DP_L1_0|3,opt,ing|y|y|y| | | |
|CNN_BP_Conv_V0_DP_L0_0|2,opt,Failed|y|y|**N**| | | |
|CNN_BP_Conv_V0_DP_L0_0|2,no opt,todo|y|y|**N**| | | |
|CNN_BP_Conv_V0_DP_L1_0|1,todo| | | | | | |
|CNN_FW_MaxPool_V0_DP_L0_0|8,Failed|y|y|y| | | |
|CNN_FW_MaxPool_V0_DP_L1_0|**8,SUC**|y|y|y| | | |
|CNN_BP_MaxPool_V0_DP_L0_0|**12,opt,SUC**|y|y|y| | | |
|CNN_BP_MaxPool_V0_DP_L0_0|12,no opt,Failed|y|y|y| | | |
|CNN_BP_MaxPool_V0_DP_L1_0|12,opt,ing|y|y|y| | | |
|CNN_BP_MaxPool_V0_DP_L1_0|12,no opt,Failed|y|y|y| | | |
|CNN_FW_Softmax_V0_DP_L3_0|**12,SUC**|y|y|y| | | |
|CNN_BP_Softmax_V0_DP_L3_0|24,Failed|y|y|y| | | |
|CNN_BP_Softmax_V0_DP_L3_0|12,todo|y|y|y| | | |



##discarded
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|
|CNN_FW_Conv_V0   |N/A|y|y|y|y|y|
|CNN_FW_Conv_V1 - Small|4,Failed;**3,SUC**|y|y|y|y|y|
|CNN_FW_Conv_V1 - Large|6,Failed;4,Failed;**3,SUC**|y|y| |y|y|
|CNN_FW_Conv_V2 - Small|3,SUC;6,Failed;**4,SUC**|y|y|y|y|y|
|CNN_FW_Conv_V2 - Large|6,Failed;**4,SUC**| | | | | |
|CNN_BP_Conv_V0|N/A|y|y|y|y|y|
|CNN_BP_Conv_V1 - Small|2,Failed;**1,SUC**|y|y|y|y|y|
|CNN_BP_Conv_V1 - Large|2,Failed;**1,SUC**|y|y|y|y|y|
|CNN_BP_Conv_V2 - Small|1,SUC;**2,SUC**|y|y|y|y|y|
|CNN_BP_Conv_V2 - Large|**2,SUC**|y|y| |y|y|
|CNN_FW_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_FW_MaxPool_V1 - Small|**8,SUC**;12,ing|y|y|y|y|y|
|CNN_FW_MaxPool_V1 - Large|12,Failed;**8,SUC**|y|y| |y|y|
|CNN_BP_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_BP_MaxPool_V1 - Small|**12,SUC**|y|y|y|y|y|
|CNN_BP_MaxPool_V1 - Large|**12,SUC**|y|y| |y|y|
|CNN_FW_Softmax_V0|Discarded.1| | | | | |
|CNN_FW_Softmax_V1|**12,SUC**|y|y|y|y|y|
|CNN_BP_Softmax_V0|Discarded.1| | | | | |
|CNN_BP_Softmax_V1|12,SUC;**24,SUC**|y|y|y|y|y|

