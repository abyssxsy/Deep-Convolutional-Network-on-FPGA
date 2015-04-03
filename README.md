# Deep-Convolutional-Network-on-FPGA
a simplified version of LeNet5

## Double Precision
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|Preliminary Resource Usage|Final Resource Usage|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|:-------------------------|:-------------------|
|CNN_FW_Conv_V0_DP_L0_0|4,opt,Failed|y|y| | | | |
|CNN_FW_Conv_V0_DP_L0_0|**3,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 179955 / 297600 (60.47%)</li><li>LUT: 123671 / 297600 (41.56%)</li><li>Primary FFs: 163510 / 297600 (54.94%)</li><li>Multipliers (25x18): 1062 / 2016 (52.68%)</li><li>DSP blocks: 1062 / 2016 (52.68%)</li><li>Block memory (BRAM18): 611 / 2128 (28.71%)</li></ul>|<ul><li>Logic utilization: 151119 / 297600 (50.78%)</li><li>LUT: 114959 / 297600 (38.63%)</li><li>Primary FFs: 130923 / 297600 (43.99%)</li><li>Secondary FFs: 30416 / 297600 (10.22%)</li><li>Multipliers (25x18): 1059 / 2016 (52.53%)</li><li>DSP blocks: 1059 / 2016 (52.53%)</li><li>Block memory (BRAM18): 612 / 2128 (28.76%)</li></ul>|
|CNN_FW_Conv_V0_DP_L1_0|2,opt,Failed|y|y| | | | | |
|CNN_FW_Conv_V0_DP_L1_0|1,opt,ing1|y|y| | | | | |
|-|--|--|--|-|--|--|-|-|
|CNN_BP_Conv_V0_DP_L0_0|2,no opt,todo|y|y|**N**| | | | |
|CNN_BP_Conv_V0_DP_L0_0|**1,no opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 170196 / 297600 (57.19%)</li><li>LUT: 117895 / 297600 (39.62%)</li><li>Primary FFs: 152393 / 297600 (51.21%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 1839 / 2128 (86.42%)</li></ul>|<ul><li>Logic utilization: 138172 / 297600 (46.43%)</li><li>LUT: 106894 / 297600 (35.92%)</li><li>Primary FFs: 120586 / 297600 (40.52%)</li><li>Secondary FFs: 29720 / 297600 (9.99%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 1840 / 2128 (86.47%)</li></ul>|
|CNN_BP_Conv_V0_DP_L1_0|**1,no opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 165763 / 297600 (55.70%)</li><li>LUT: 114306 / 297600 (38.41%)</li><li>Primary FFs: 148937 / 297600 (50.05%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 798 / 2128 (37.50%)</li></ul>|<ul><li>Logic utilization: 138747 / 297600 (46.62%)</li><li>LUT: 101849 / 297600 (34.22%)</li><li>Primary FFs: 121336 / 297600 (40.77%)</li><li>Secondary FFs: 25512 / 297600 (8.57%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 799 / 2128 (37.55%)</li></ul>|
|-|--|--|--|-|--|--|-|-|
|CNN_FW_MaxPool_V0_DP_L0_0|8,opt,Failed|y|y| | | | |
|CNN_FW_MaxPool_V0_DP_L0_0|**6,opt,SUC**|y|y| | | | |
|CNN_FW_MaxPool_V0_DP_L1_0|**8,SUC**|y|y| |y|y| |
|-|--|--|--|-|--|--|-|-|
|CNN_BP_MaxPool_V0_DP_L0_0|**12,opt,SUC**|y|y| |y|y| |
|CNN_BP_MaxPool_V0_DP_L0_0|12,no opt,Failed|y|y| | | | |
|CNN_BP_MaxPool_V0_DP_L1_0|**12,opt,SUC**|y|y| |y|y| |
|CNN_BP_MaxPool_V0_DP_L1_0|12,no opt,Failed|y|y| | | | |
|-|--|--|--|-|--|--|-|-|
|CNN_FW_Softmax_V0_DP_L3_0|**12,opt,SUC**|y|y| |y|y| |
|-|--|--|--|-|--|--|-|-|
|CNN_BP_Softmax_V0_DP_L3_0|24,opt,Failed|y|y| | | | |
|CNN_BP_Softmax_V0_DP_L3_0|**12,opt,SUC**|y|y| |y|y| |



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

