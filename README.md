# Deep-Convolutional-Network-on-FPGA
a simplified version of LeNet5

## Double Precision
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|
|CNN_FW_Conv_V0   |N/A|y|y|y|y|y|
|CNN_FW_Conv_V1 - Small|4,Failed;4,50MHz,SUC;**3,SUC**|y|y|y|y|y|
|CNN_FW_Conv_V1 - Large|**3,SUC**;6,Failed;4,Failed|y|y| |y|y|
|CNN_FW_Conv_V2 - Small|3,SUC;**4,SUC**;6,Failed|y|y|y|y|y|
|CNN_FW_Conv_V2 - Large|**4,SUC**;6,Failed| | | | | |
|CNN_BP_Conv_V0|N/A|y|y|y|y|y|
|CNN_BP_Conv_V1 - Small|**1,SUC**;2,Failed|y|y|y|y|y|
|CNN_BP_Conv_V1 - Large|**1,SUC**;2,Failed|y|y|y|y|y|
|CNN_BP_Conv_V2 - Small|1,SUC;**2,SUC**|y|y|y|y|y|
|CNN_BP_Conv_V2 - Large|**2,SUC**|y|y| |y|y|
|CNN_FW_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_FW_MaxPool_V1 - Small|12,Failed;**8,SUC**|y|y|y|y|y|
|CNN_FW_MaxPool_V1 - Large|8,SUC;12,ing|y|y| |y|y|
|CNN_BP_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_BP_MaxPool_V1 - Small|**12,SUC**|y|y|y|y|y|
|CNN_BP_MaxPool_V1 - Large|**12,SUC**|y|y| |y|y|
|CNN_FW_Softmax_V0|Discarded.1| | | | | |
|CNN_FW_Softmax_V1|**12,SUC**|y|y|y|y|y|
|CNN_BP_Softmax_V0|Discarded.1| | | | | |
|CNN_BP_Softmax_V1|12,SUC;**24,SUC**|y|y|y|y|y|

