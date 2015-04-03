# Deep-Convolutional-Network-on-FPGA
a simplified version of LeNet5

## Double Precision
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|Preliminary Resource Usage|Final Resource Usage|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|:-------------------------|:-------------------|
|CNN_FW_Conv_V0_DP_L0_0|4,opt,ing1|y|y| | | | |
|CNN_FW_Conv_V0_DP_L0_0|**3,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 179955 / 297600 (60.47%)</li><li>LUT: 123671 / 297600 (41.56%)</li><li>Primary FFs: 163510 / 297600 (54.94%)</li><li>Multipliers (25x18): 1062 / 2016 (52.68%)</li><li>DSP blocks: 1062 / 2016 (52.68%)</li><li>Block memory (BRAM18): 611 / 2128 (28.71%)</li></ul>|<ul><li>Logic utilization: 151119 / 297600 (50.78%)</li><li>LUT: 114959 / 297600 (38.63%)</li><li>Primary FFs: 130923 / 297600 (43.99%)</li><li>Secondary FFs: 30416 / 297600 (10.22%)</li><li>Multipliers (25x18): 1059 / 2016 (52.53%)</li><li>DSP blocks: 1059 / 2016 (52.53%)</li><li>Block memory (BRAM18): 612 / 2128 (28.76%)</li></ul>|
|CNN_FW_Conv_V0_DP_L1_0|2,opt,Failed|y|y| | | | | |
|CNN_FW_Conv_V0_DP_L1_0|**1,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 108726 / 297600 (36.530%)</li><li>LUT: 65900 / 297600 (22.14%)</li><li>Primary FFs: 92328 / 297600 (31.02%)</li><li>Multipliers (25x18): 362 / 2016 (17.96%)</li><li>DSP blocks: 362 / 2016 (17.96%)</li><li>Block memory (BRAM18): 443 / 2128 (20.82%)</li></ul>|<ul><li>Logic utilization: 88253 / 297600 (29.65%)</li><li>LUT: 61945 / 297600 (20.81%)</li><li>Primary FFs: 74239 / 297600 (24.95%)</li><li>Secondary FFs: 15916 / 297600 (5.35%)</li><li>Multipliers (25x18): 359 / 2016 (17.81%)</li><li>DSP blocks: 359 / 2016 (17.87%)</li><li>Block memory (BRAM18): 444 / 2128 (20.86%)</li></ul>|
|-|--|--|--|-|--|--|-|-|
|CNN_BP_Conv_V0_DP_L0_0|2,opt,ing3|y|y|**N**| | | | |
|CNN_BP_Conv_V0_DP_L0_0|**1,no opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 170196 / 297600 (57.19%)</li><li>LUT: 117895 / 297600 (39.62%)</li><li>Primary FFs: 152393 / 297600 (51.21%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 1839 / 2128 (86.42%)</li></ul>|<ul><li>Logic utilization: 138172 / 297600 (46.43%)</li><li>LUT: 106894 / 297600 (35.92%)</li><li>Primary FFs: 120586 / 297600 (40.52%)</li><li>Secondary FFs: 29720 / 297600 (9.99%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 1840 / 2128 (86.47%)</li></ul>|
|CNN_BP_Conv_V0_DP_L1_0|**1,no opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 165763 / 297600 (55.70%)</li><li>LUT: 114306 / 297600 (38.41%)</li><li>Primary FFs: 148937 / 297600 (50.05%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 798 / 2128 (37.50%)</li></ul>|<ul><li>Logic utilization: 138747 / 297600 (46.62%)</li><li>LUT: 101849 / 297600 (34.22%)</li><li>Primary FFs: 121336 / 297600 (40.77%)</li><li>Secondary FFs: 25512 / 297600 (8.57%)</li><li>Multipliers (25x18): 500 / 2016 (24.80%)</li><li>DSP blocks: 500 / 2016 (24.80%)</li><li>Block memory (BRAM18): 799 / 2128 (37.55%)</li></ul>|
|-|--|--|--|-|--|--|-|-|
|CNN_FW_MaxPool_V0_DP_L0_0|8,opt,ing2|y|y| | | | | |
|CNN_FW_MaxPool_V0_DP_L0_0|**6,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 169030 / 297600 (56.80%)</li><li>LUT: 109708 / 297600 (36.86%)</li><li>Primary FFs: 148978 / 297600 (50.06%)</li><li>Multipliers (25x18): 552 / 2016 (27.38%)</li><li>DSP blocks: 552 / 2016 (27.38%)</li><li>Block memory (BRAM18): 507 / 2128 (23.83%)</li></ul>|<ul><li>Logic utilization: 132959 / 297600 (44.68%)</li><li>LUT: 101993 / 297600 (34.27%)</li><li>Primary FFs: 116100 / 297600 (39.01%)</li><li>Secondary FFs: 29911 / 297600 (10.05%)</li><li>Multipliers (25x18): 552 / 2016 (27.38%)</li><li>DSP blocks: 552 / 2016 (27.38%)</li><li>Block memory (BRAM18): 508 / 2128 (23.87%)</li></ul>|
|CNN_FW_MaxPool_V0_DP_L1_0|**8,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 200661 / 297600 (67.43%)</li><li>LUT: 134813 / 297600 (45.30%)</li><li>Primary FFs: 179563 / 297600 (60.34%)</li><li>Multipliers (25x18): 736 / 2016 (36.51%)</li><li>DSP blocks: 736 / 2016 (36.51%)</li><li>Block memory (BRAM18): 533 / 2128 (25.05%)</li></ul>|<ul><li>Logic utilization: 158446 / 297600 (53.24%)</li><li>LUT: 124093 / 297600 (41.70%)</li><li>Primary FFs: 139704 / 297600 (46.94%)</li><li>Secondary FFs: 36574 / 297600 (12.29%)</li><li>Multipliers (25x18): 736 / 2016 (36.51%)</li><li>DSP blocks: 736 / 2016 (36.51%)</li><li>Block memory (BRAM18): 534 / 2128 (25.09%)</li></ul>|
|-|--|--|--|-|--|--|-|-|
|CNN_BP_MaxPool_V0_DP_L0_0|**12,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 254351 / 297600 (85.47%)</li><li>LUT: 169009 / 297600 (56.79%)</li><li>Primary FFs: 230991 / 297600 (77.62%)</li><li>Multipliers (25x18): 1335 / 2016 (66.22%)</li><li>DSP blocks: 1335 / 2016 (66.22%)</li><li>Block memory (BRAM18): 625 / 2128 (29.37%)</li></ul>|<ul><li>Logic utilization: 193699 / 297600 (65.09%)</li><li>LUT: 155852 / 297600 (52.37%)</li><li>Primary FFs: 176356 / 297600 (59.26%)</li><li>Secondary FFs: 50631 / 297600 (17.01%)</li><li>Multipliers (25x18): 1335 / 2016 (66.22%)</li><li>DSP blocks: 1335 / 2016 (66.22%)</li><li>Block memory (BRAM18): 626 / 2128 (29.42%)</li></ul>|
|CNN_BP_MaxPool_V0_DP_L0_0|12,no opt,Failed|y|y| | | | | |
|CNN_BP_MaxPool_V0_DP_L1_0|**12,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 254379 / 297600 (85.44%)</li><li>LUT: 168961 / 297600 (56.77%)</li><li>Primary FFs: 230918 / 297600 (77.59%)</li><li>Multipliers (25x18): 1335 / 2016 (66.22%)</li><li>DSP blocks: 1335 / 2016 (66.22%)</li><li>Block memory (BRAM18): 601 / 2128 (28.24%)</li></ul>|<ul><li>Logic utilization: 196810 / 297600 (66.13%)</li><li>LUT: 155137 / 297600 (52.13%)</li><li>Primary FFs: 179436 / 297600 (60.29%)</li><li>Secondary FFs: 47478 / 297600 (15.95%)</li><li>Multipliers (25x18): 1335 / 2016 (66.22%)</li><li>DSP blocks: 1335 / 2016 (66.22%)</li><li>Block memory (BRAM18): 602 / 2128 (28.29%)</li></ul>|
|CNN_BP_MaxPool_V0_DP_L1_0|12,no opt,Failed|y|y| | | | | |
|-|--|--|--|-|--|--|-|-|
|CNN_FW_Softmax_V0_DP_L3_0|**12,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 241608 / 297600 (81.19%)</li><li>LUT: 159726 / 297600 (53.67%)</li><li>Primary FFs: 221208 / 297600 (74.33%)</li><li>Multipliers (25x18): 756 / 2016 (37.50%)</li><li>DSP blocks: 756 / 2016 (37.50%)</li><li>Block memory (BRAM18): 474 / 2128 (22.27%)</li></ul>|<ul><li>Logic utilization: 186172 / 297600 (62.56%)</li><li>LUT: 149216 / 297600 (50.14%)</li><li>Primary FFs: 167921 / 297600 (56.43%)</li><li>Secondary FFs: 50393 / 297600 (16.93%)</li><li>Multipliers (25x18): 756 / 2016 (37.50%)</li><li>DSP blocks: 756 / 2016 (37.50%)</li><li>Block memory (BRAM18): 475 / 2128 (22.32%)</li></ul>|
|-|--|--|--|-|--|--|-|-|
|CNN_BP_Softmax_V0_DP_L3_0|24,opt,ing4|y|y| | | | | |
|CNN_BP_Softmax_V0_DP_L3_0|**12,opt,SUC**|y|y| |y|y|<ul><li>Logic utilization: 189433 / 297600 (63.65%)</li><li>LUT: 109679 / 297600 (36.85%)</li><li>Primary FFs: 171678 / 297600 (57.69%)</li><li>Multipliers (25x18): 336 / 2016 (16.67%)</li><li>DSP blocks: 336 / 2016 (16.67%)</li><li>Block memory (BRAM18): 468 / 2128 (21.99%)</li></ul>|<ul><li>Logic utilization: 130485 / 297600 (43.85%)</li><li>LUT: 105249 / 297600 (35.37%)</li><li>Primary FFs: 114851 / 297600 (38.59%)</li><li>Secondary FFs: 19924 / 297600 (6.69%)</li><li>Multipliers (25x18): 336 / 2016 (16.67%)</li><li>DSP blocks: 336 / 2016 (16.67%)</li><li>Block memory (BRAM18): 469 / 2128 (22.04%)</li></ul>|


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

