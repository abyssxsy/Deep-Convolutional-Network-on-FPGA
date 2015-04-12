#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstring>
#include<sys/time.h>
#include<algorithm>
#include<limits>
#include<vector>
#include<cassert>

#include"max_header.h"

typedef double real;
typedef int cateType;
const int timer_count = 3;
timeval old_timer[timer_count];

namespace{
    const char train_set_x_filename[] = "./data/datasets/train_x.bin";
    const char valid_set_x_filename[] = "./data/datasets/valid_x.bin";
    const char test_set_x_filename[]  = "./data/datasets/test_x.bin";
    const char train_set_y_filename[] = "./data/datasets/train_y.bin";
    const char valid_set_y_filename[] = "./data/datasets/valid_y.bin";
    const char test_set_y_filename[]  = "./data/datasets/test_y.bin";
    const char layer0_w_filename[]    = "./data/step_0/layer0_w.bin";
    const char layer0_b_filename[]    = "./data/step_0/layer0_b.bin";
    const char layer1_w_filename[]    = "./data/step_0/layer1_w.bin";
    const char layer1_b_filename[]    = "./data/step_0/layer1_b.bin";
    const char layer2_w_filename[]    = "./data/step_0/layer2_w.bin";
    const char layer2_b_filename[]    = "./data/step_0/layer2_b.bin";
    const char layer3_w_filename[]    = "./data/step_0/layer3_w.bin";
    const char layer3_b_filename[]    = "./data/step_0/layer3_b.bin";

    const int NKERS[4] = {20,50,500,10};
    const int BATCH_SIZE = 384;
    const int L0_FL[4] = {1,NKERS[0],5,5};
    const int L0_IM[7] = {BATCH_SIZE,28,28,24,24,12,12};
    const int L1_FL[4] = {NKERS[0],NKERS[1],5,5};
    const int L1_IM[7] = {BATCH_SIZE,12,12,8,8,4,4};
    const int L2_SP[3] = {BATCH_SIZE,NKERS[1]*4*4,NKERS[2]};
    const int L3_SP[3] = {BATCH_SIZE,NKERS[2],NKERS[3]};
    //TODO: 130,26,26
    const int n_train_batches = 13;
    const int n_valid_batches = 2;
    const int n_test_batches  = 2;

    const int layer0_w_count = L0_FL[0]*L0_FL[1]*L0_FL[2]*L0_FL[3];
    const int layer0_b_count = L0_FL[1];
    const int layer1_w_count = L1_FL[0]*L1_FL[1]*L1_FL[2]*L1_FL[3];
    const int layer1_b_count = L1_FL[1];
    const int layer2_w_count = L2_SP[1]*L2_SP[2];
    const int layer2_b_count = L2_SP[2];
    const int layer3_w_count = L3_SP[1]*L3_SP[2];
    const int layer3_b_count = L3_SP[2];
    const int layer3_pred_count = BATCH_SIZE;

    const int layer0_w_size = layer0_w_count*sizeof(real);
    const int layer0_b_size = layer0_b_count*sizeof(real);
    const int layer1_w_size = layer1_w_count*sizeof(real);
    const int layer1_b_size = layer1_b_count*sizeof(real);
    const int layer2_w_size = layer2_w_count*sizeof(real);
    const int layer2_b_size = layer2_b_count*sizeof(real);
    const int layer3_w_size = layer3_w_count*sizeof(real);
    const int layer3_b_size = layer3_b_count*sizeof(real);
    const int layer3_pred_size = BATCH_SIZE*sizeof(cateType);
    const int layer0_w_grad_size = layer0_w_size*BATCH_SIZE;
    const int layer0_b_grad_size = layer0_b_size*BATCH_SIZE;
    const int layer1_w_grad_size = layer1_w_size*BATCH_SIZE;
    const int layer1_b_grad_size = layer1_b_size*BATCH_SIZE;
    const int layer2_w_grad_size = layer2_w_size*BATCH_SIZE;
    const int layer2_b_grad_size = layer2_b_size*BATCH_SIZE;
    const int layer3_w_grad_size = layer3_w_size*BATCH_SIZE;
    const int layer3_b_grad_size = layer3_b_size*BATCH_SIZE;

    const int layer0_x_count = L0_FL[0]*L0_IM[1]*L0_IM[2]*BATCH_SIZE;
    const int layer0_z2_count = L0_FL[1]*L0_IM[3]*L0_IM[4]*BATCH_SIZE;
    const int layer0_sel_count = layer0_z2_count;
    const int layer0_z_count = L0_FL[1]*L0_IM[5]*L0_IM[6]*BATCH_SIZE;
    const int layer0_a_count = layer0_z_count;
    const int layer1_x_count = layer0_a_count;
    const int layer1_z2_count = L1_FL[1]*L1_IM[3]*L1_IM[4]*BATCH_SIZE;
    const int layer1_sel_count = layer1_z2_count;
    const int layer1_z_count = L1_FL[1]*L1_IM[5]*L1_IM[6]*BATCH_SIZE;
    const int layer1_a_count = layer1_z_count;
    const int layer2_x_count = layer1_a_count;
    const int layer2_z_count = L2_SP[2]*BATCH_SIZE;
    const int layer2_a_count = L2_SP[2]*BATCH_SIZE;
    const int layer3_x_count = layer2_a_count;
    const int layer3_z_count = L3_SP[2]*BATCH_SIZE;
    const int layer3_sm_count = layer3_z_count;

    const int layer0_x_size = layer0_x_count*sizeof(real);
    const int layer0_z2_size = layer0_z2_count*sizeof(real);
    const int layer0_sel_size = layer0_sel_count/8;
    const int layer0_z_size = layer0_z_count*sizeof(real);
    const int layer0_a_size = layer0_a_count*sizeof(real);
    const int layer1_x_size = layer1_x_count*sizeof(real);
    const int layer1_z2_size = layer1_z2_count*sizeof(real);
    const int layer1_sel_size = layer1_sel_count/8;
    const int layer1_z_size = layer1_z_count*sizeof(real);
    const int layer1_a_size = layer1_a_count*sizeof(real);
    const int layer2_x_size = layer2_x_count*sizeof(real);
    const int layer2_z_size = layer2_z_count*sizeof(real);
    const int layer2_a_size = layer2_a_count*sizeof(real);
    const int layer3_x_size = layer3_x_count*sizeof(real);
    const int layer3_z_size = layer3_z_count*sizeof(real);
    const int layer3_sm_size = layer3_sm_count*sizeof(real);
    const int layer3_x_grad_size = layer3_x_size;
    const int layer2_a_grad_size = layer2_a_size;
    const int layer2_z_grad_size = layer2_z_size;
    const int layer2_x_grad_size = layer2_x_size;
    const int layer1_a_grad_size = layer1_a_size;
    const int layer1_z2_grad_size = layer1_z2_size;
    const int layer1_x_grad_size = layer1_x_size;
    const int layer0_a_grad_size = layer0_a_size;
    const int layer0_z2_grad_size = layer0_z2_size;
    const int layer0_x_grad_size = layer0_x_size;

    const int train_set_x_size_iter = layer0_x_size;
    const int valid_set_x_size_iter = layer0_x_size;
    const int test_set_x_size_iter = layer0_x_size;
    const int train_set_x_size = n_train_batches*layer0_x_size;
    const int valid_set_x_size = n_valid_batches*layer0_x_size;
    const int test_set_x_size = n_test_batches*layer0_x_size;
    const int train_set_y_size_iter = layer3_pred_size;
    const int valid_set_y_size_iter = layer3_pred_size;
    const int test_set_y_size_iter = layer3_pred_size;
    const int train_set_y_size = n_train_batches*layer3_pred_size;
    const int valid_set_y_size = n_valid_batches*layer3_pred_size;
    const int test_set_y_size = n_test_batches*layer3_pred_size;

    const int train_set_x_offset = 4*384;
    const int valid_set_x_offset = train_set_x_offset+train_set_x_size;
    const int test_set_x_offset = valid_set_x_offset+valid_set_x_size;
    //const int layer0_x_offset = test_set_x_offset+test_set_x_size;
    const int layer0_z2_offset = test_set_x_offset;
    const int layer0_sel_offset = layer0_z2_offset+layer0_z2_size;
    const int layer0_z_offset = layer0_sel_offset+layer0_sel_size;
    const int layer0_a_offset = layer0_z_offset+layer0_z_size;
    const int layer1_x_offset = layer0_a_offset;
    const int layer1_z2_offset = layer1_x_offset+layer1_x_size;
    const int layer1_sel_offset = layer1_z2_offset+layer1_z2_size;
    const int layer1_z_offset = layer1_sel_offset+layer1_sel_size;
    const int layer1_a_offset = layer1_z_offset+layer1_z_size;
    const int layer2_x_offset = layer1_a_offset;
    const int layer2_z_offset = layer2_x_offset+layer2_x_size;
    const int layer2_a_offset = layer2_z_offset+layer2_z_size;
    const int layer3_x_offset = layer2_a_offset;
    const int layer3_z_offset = layer3_x_offset+layer3_x_size;
    const int layer3_sm_offset = layer3_z_offset+layer3_z_size;
    const int layer3_x_grad_offset = layer3_sm_offset+layer3_sm_size;
    const int layer2_a_grad_offset = layer3_x_grad_offset;
    const int layer2_z_grad_offset = layer2_a_grad_offset+layer2_a_grad_size;
    const int layer2_x_grad_offset = layer2_z_grad_offset+layer2_z_grad_size;
    const int layer1_a_grad_offset = layer2_x_grad_offset;
    const int layer1_z2_grad_offset = layer1_a_grad_offset+layer1_a_grad_size;
    const int layer1_x_grad_offset = layer1_z2_grad_offset+layer1_z2_grad_size;
    const int layer0_a_grad_offset = layer1_x_grad_offset;
    const int layer0_z2_grad_offset = layer0_a_grad_offset+layer0_a_grad_size;
    const int layer0_x_grad_offset = layer0_z2_grad_offset+layer0_z2_grad_size;
};

namespace{
    real* train_set_x;
    real* valid_set_x;
    real* test_set_x;
    cateType* train_set_y;
    cateType* valid_set_y;
    cateType* test_set_y;

    real* layer0_w;
    real* layer0_b;
    real* layer1_w;
    real* layer1_b;
    real* layer2_w;
    real* layer2_b;
    real* layer3_w;
    real* layer3_b;

    real* layer0_w_grad;
    real* layer0_b_grad;
    real* layer1_w_grad;
    real* layer1_b_grad;
    real* layer2_w_grad;
    real* layer2_b_grad;
    real* layer3_w_grad;
    real* layer3_b_grad;
    cateType* layer3_pred;

    enum kernel {
        K_fw_l0_conv,K_fw_l0_maxpool,K_fw_l1_conv,K_fw_l1_maxpool,K_fw_l2_mlp,K_fw_l3_softmax,
        K_bp_l3_softmax,K_bp_l2_mlp,K_bp_l1_maxpool,K_bp_l1_conv,K_bp_l0_maxpool,K_bp_l0_conv,
        K_TOTAL
    };
    max_file_t* max_files[K_TOTAL];
    max_engine_t* max_engines[K_TOTAL];
    int cur_engine;
};

void mark_timer(bool pt=false,int timer_id=0){
    assert((timer_id<timer_count) && (timer_id>=0));
    long s0 = old_timer[timer_id].tv_sec;
    long t0 = old_timer[timer_id].tv_usec;
    gettimeofday(&old_timer[timer_id],NULL);
    long um = (old_timer[timer_id].tv_sec-s0)*1000+(old_timer[timer_id].tv_usec-t0)/1000;
    if (pt){
        fprintf(stdout,"time used = %d.%03d s\n",um/1000,um%1000);
    }
}

template<typename T>
void alloc_read(const char* filename,T** ptr,int count){
    *ptr = (T*)malloc(sizeof(T)*count);
    FILE* inf = fopen(filename,"rb");
    fread(*ptr,sizeof(T),count,inf);
    fclose(inf);
}
void load_data(){
    alloc_read(train_set_x_filename,&train_set_x,n_train_batches*layer0_x_count);
    alloc_read(valid_set_x_filename,&valid_set_x,n_valid_batches*layer0_x_count);
    alloc_read(test_set_x_filename, &test_set_x, n_test_batches *layer0_x_count);
    alloc_read(train_set_y_filename,&train_set_y,n_train_batches*BATCH_SIZE);
    alloc_read(valid_set_y_filename,&valid_set_y,n_valid_batches*BATCH_SIZE);
    alloc_read(test_set_y_filename, &test_set_y ,n_test_batches *BATCH_SIZE);

    alloc_read(layer0_w_filename,&layer0_w,layer0_w_count);
    alloc_read(layer0_b_filename,&layer0_b,layer0_b_count);
    alloc_read(layer1_w_filename,&layer1_w,layer1_w_count);
    alloc_read(layer1_b_filename,&layer1_b,layer1_b_count);
    alloc_read(layer2_w_filename,&layer2_w,layer2_w_count);
    alloc_read(layer2_b_filename,&layer2_b,layer2_b_count);
    alloc_read(layer3_w_filename,&layer3_w,layer3_w_count);
    alloc_read(layer3_b_filename,&layer3_b,layer3_b_count);

    layer0_w_grad = (real*)malloc(layer0_w_grad_size);
    layer0_b_grad = (real*)malloc(layer0_b_grad_size);
    layer1_w_grad = (real*)malloc(layer1_w_grad_size);
    layer1_b_grad = (real*)malloc(layer1_b_grad_size);
    layer2_w_grad = (real*)malloc(layer2_w_grad_size);
    layer2_b_grad = (real*)malloc(layer2_b_grad_size);
    layer3_w_grad = (real*)malloc(layer3_w_grad_size);
    layer3_b_grad = (real*)malloc(layer3_b_grad_size);
    layer3_pred = (cateType*)malloc(layer3_pred_size);
}
void unload_data(){
    free(train_set_x);
    free(valid_set_x);
    free(test_set_x);
    free(train_set_y);
    free(valid_set_y);
    free(test_set_y);

    free(layer0_w);
    free(layer0_b);
    free(layer1_w);
    free(layer1_b);
    free(layer2_w);
    free(layer2_b);
    free(layer3_w);
    free(layer3_b);

    free(layer0_w_grad);
    free(layer0_b_grad);
    free(layer1_w_grad);
    free(layer1_b_grad);
    free(layer2_w_grad);
    free(layer2_b_grad);
    free(layer3_w_grad);
    free(layer3_b_grad);
    free(layer3_pred);
}
void load_engine(int engine_id){
    if (-1!=cur_engine){
        max_unload(max_engines[cur_engine]);
    }
    max_engines[engine_id] = max_load(max_files[engine_id],"*");
    cur_engine = engine_id;
}
void load_maxfiles(){
    fprintf(stdout,"Init maxfiles\n");
    max_files[K_fw_l0_conv] = CNN_FW_Conv_V0_DP_L0_0_init();
    max_files[K_fw_l0_maxpool] = CNN_FW_MaxPool_V0_DP_L0_0_init();
    max_files[K_fw_l1_conv] = CNN_FW_Conv_V0_DP_L1_0_init();
    max_files[K_fw_l1_maxpool] = CNN_FW_MaxPool_V0_DP_L1_0_init();
    max_files[K_fw_l2_mlp] = NULL;
    max_files[K_fw_l3_softmax] = CNN_FW_Softmax_V0_DP_L3_0_init();
    max_files[K_bp_l3_softmax] = CNN_BP_Softmax_V0_DP_L3_0_init();
    max_files[K_bp_l2_mlp] = NULL;
    max_files[K_bp_l1_maxpool] = CNN_BP_MaxPool_V0_DP_L1_0_init();
    max_files[K_bp_l1_conv] = CNN_BP_Conv_V0_DP_L1_0_init();
    max_files[K_bp_l0_maxpool] = CNN_BP_MaxPool_V0_DP_L0_0_init();
    max_files[K_bp_l0_conv] = CNN_BP_Conv_V0_DP_L0_0_init();
    for (int i=0;i<K_TOTAL;++i){
        max_engines[i] = NULL;
    }
    cur_engine = -1;

    int t = K_fw_l0_conv;
    load_engine(t);
    {
        fprintf(stdout,"Writing to LMem : train_set_x\n");
        max_actions_t* act;
        act = max_actions_init(max_files[t], "writeLMem");
        max_set_param_uint64t(act, "offset", train_set_x_offset);
        max_set_param_uint64t(act, "size", train_set_x_size);
        max_queue_input(act, "cpu_to_lmem_at_cpu", train_set_x, train_set_x_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
    }
    {
        fprintf(stdout,"Writing to LMem : valid_set_x\n");
        max_actions_t* act;
        act = max_actions_init(max_files[t], "writeLMem");
        max_set_param_uint64t(act, "offset", valid_set_x_offset);
        max_set_param_uint64t(act, "size", valid_set_x_size);
        max_queue_input(act, "cpu_to_lmem_at_cpu", valid_set_x, valid_set_x_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
    }
    {
        fprintf(stdout,"Writing to LMem : test_set_x\n");
        max_actions_t* act;
        act = max_actions_init(max_files[t], "writeLMem");
        max_set_param_uint64t(act, "offset", test_set_x_offset);
        max_set_param_uint64t(act, "size", test_set_x_size);
        max_queue_input(act, "cpu_to_lmem_at_cpu", test_set_x, test_set_x_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
    }
}
void unload_maxfiles(){
    if (-1!=cur_engine){
        max_unload(max_engines[cur_engine]);
        cur_engine = -1;
    }
    for (int i=0;i<K_TOTAL;++i){
        if (max_files[i]){
            max_file_free(max_files[i]);
        }
    }
}

void my_process(int data_x_offset,const cateType* data_y,int mb_idx,real learning_rate){
    {
        int t = K_fw_l0_conv;
        load_engine(t);
        printf("Running on DFE: fw_l0_conv");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", 1);
        max_set_param_uint64t(act, "no", NKERS[0]);
        max_queue_input(act, "b", layer0_b, layer0_b_size);
        max_queue_input(act, "w", layer0_w, layer0_w_size);
        max_set_param_uint64t(act, "x_offset", data_x_offset+mb_idx*layer0_x_size);
        max_set_param_uint64t(act, "z_offset", layer0_z2_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_fw_l0_maxpool;
        load_engine(t);
        printf("Running on DFE: fw_l0_maxpool");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "no", NKERS[0]);
        max_set_param_uint64t(act, "z2_offset", layer0_z2_offset);
        max_set_param_uint64t(act, "sel_offset", layer0_sel_offset);
        max_set_param_uint64t(act, "z_offset", layer0_z_offset);
        max_set_param_uint64t(act, "a_offset", layer0_a_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_fw_l1_conv;
        load_engine(t);
        printf("Running on DFE: fw_l1_conv");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", NKERS[0]);
        max_set_param_uint64t(act, "no", NKERS[1]);
        max_queue_input(act, "b", layer1_b, layer1_b_size);
        max_queue_input(act, "w", layer1_w, layer1_w_size);
        max_set_param_uint64t(act, "x_offset", layer1_x_offset);
        max_set_param_uint64t(act, "z_offset", layer1_z2_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_fw_l1_maxpool;
        load_engine(t);
        printf("Running on DFE: fw_l1_maxpool");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "no", NKERS[1]);
        max_set_param_uint64t(act, "z2_offset", layer1_z2_offset);
        max_set_param_uint64t(act, "sel_offset", layer1_sel_offset);
        max_set_param_uint64t(act, "z_offset", layer1_z_offset);
        max_set_param_uint64t(act, "a_offset", layer1_a_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_fw_l3_softmax;
        load_engine(t);
        printf("Running on DFE: fw_l3_softmax");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", NKERS[2]);
        max_set_param_uint64t(act, "x_offset", layer3_x_offset);
        max_queue_input(act, "w", layer3_w, layer3_w_size);
        max_queue_input(act, "b", layer3_b, layer3_b_size);
        max_set_param_uint64t(act, "softmax_offset", layer3_sm_offset);
        max_queue_output(act, "pred", layer3_pred, layer3_pred_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    //TODO: learning rate<0 exit
    {
        int t = K_bp_l3_softmax;
        load_engine(t);
        printf("Running on DFE: bp_l3_softmax");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", NKERS[2]);
        max_set_param_uint64t(act, "x_offset", layer3_x_offset);
        max_queue_input(act, "w", layer3_w, layer3_w_size);
        max_set_param_uint64t(act, "softmax_offset", layer3_sm_offset);
        max_queue_input(act, "std", data_y+mb_idx*layer3_pred_size, layer3_pred_size);
        max_queue_output(act, "w_grad", layer3_w_grad, layer3_w_grad_size);
        max_queue_output(act, "b_grad", layer3_b_grad, layer3_b_grad_size);
        max_set_param_uint64t(act, "x_grad_offset", layer3_x_grad_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_bp_l1_maxpool;
        load_engine(t);
        printf("Running on DFE: bp_l1_maxpool");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "no", NKERS[1]);
        max_set_param_uint64t(act, "a_grad_offset", layer1_a_grad_offset);
        max_set_param_uint64t(act, "z_offset", layer1_z_offset);
        max_set_param_uint64t(act, "sel_offset", layer1_sel_offset);
        max_set_param_uint64t(act, "z2_grad_offset", layer1_z2_grad_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_bp_l1_conv;
        load_engine(t);
        printf("Running on DFE: bp_l1_conv");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", NKERS[0]);
        max_set_param_uint64t(act, "no", NKERS[1]);
        max_set_param_uint64t(act, "z_grad_offset", layer1_z2_grad_offset);
        max_set_param_uint64t(act, "x_offset", layer1_x_offset);
        max_set_param_uint64t(act, "x_grad_offset", layer1_x_grad_offset);
        max_queue_input(act, "w", layer1_w, layer1_w_size);
        max_queue_output(act, "w_grad", layer1_w_grad, layer1_w_grad_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_bp_l0_maxpool;
        load_engine(t);
        printf("Running on DFE: bp_l0_maxpool");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "no", NKERS[0]);
        max_set_param_uint64t(act, "a_grad_offset", layer0_a_grad_offset);
        max_set_param_uint64t(act, "z_offset", layer0_z_offset);
        max_set_param_uint64t(act, "sel_offset", layer0_sel_offset);
        max_set_param_uint64t(act, "z2_grad_offset", layer0_z2_grad_offset);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
    {
        int t = K_bp_l0_conv;
        load_engine(t);
        printf("Running on DFE: bp_l0_conv");
        mark_timer(false,1);
        max_actions_t* act = max_actions_init(max_files[t], "default");
        max_set_param_uint64t(act, "ni", 1);
        max_set_param_uint64t(act, "no", NKERS[0]);
        max_set_param_uint64t(act, "z_grad_offset", layer0_z2_grad_offset);
        max_set_param_uint64t(act, "x_offset", data_x_offset+mb_idx*layer0_x_size);
        max_set_param_uint64t(act, "x_grad_offset", layer0_x_grad_offset);
        max_queue_input(act, "w", layer0_w, layer0_w_size);
        max_queue_output(act, "w_grad", layer0_w_grad, layer0_w_grad_size);
        max_run(max_engines[t], act);
        max_actions_free(act);
        mark_timer(true,1);
    }
}

int main(int argc,char* argv[]){
    const real learning_rate = 0.1;
    const int n_epochs = 1;

    fprintf(stdout,"[INFO] loading data\n");
    mark_timer();
    load_data();
    load_maxfiles();
    fprintf(stdout,"[INFO] load data done, ");
    mark_timer(true);

    fprintf(stdout,"[INFO] training\n");
    int patience = 10000;
    int patience_increase = 2;
    real improvement_threshold = 0.995;
    int validation_frequency = std::min(n_train_batches,patience/2);
    real best_validation_loss = std::numeric_limits<real>::max();
    int best_iter = 0;
    real mtls = 0.0;
    int epoch = 0;
    bool done_looping = false;
    std::vector<real> result_logs;
    while ((epoch<n_epochs) && (!done_looping)){
        for (int mb_idx=0;mb_idx<n_train_batches;++mb_idx){
            int cur_iter = epoch*n_train_batches+mb_idx;
            fprintf(stdout, "training @ epoch = [%d,%d/%d] iter = %d\n", epoch,mb_idx,n_train_batches,cur_iter);
            mark_timer();
            my_process(train_set_x_offset,train_set_y,mb_idx,learning_rate);
            mark_timer(true);
            //save_params();
            if ((cur_iter+1)%validation_frequency==0){
                // TODO: lzh
            }
            if (patience<=cur_iter){
                done_looping = true;
                break;
            }
        }
        ++epoch;
    }

    fprintf(stdout,"[INFO] unloading data\n");
    mark_timer();
    unload_data();
    unload_maxfiles();
    fprintf(stdout,"[INFO] unload data done, ");
    mark_timer(true);
    return 0;
}