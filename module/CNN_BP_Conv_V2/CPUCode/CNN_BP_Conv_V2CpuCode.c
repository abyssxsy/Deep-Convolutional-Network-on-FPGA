#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"
//typedef float real;
typedef double real;

int main(void)
{
	const int ni = 2;
	const int no = 2;
	const int k = 5;
	const int row = 12;
	const int col = 12;
	const int batch_size = 384;

	int x_size = ni*row*col*batch_size*sizeof(real);
	int z_grad_size = no*(row-k+1)*(col-k+1)*batch_size*sizeof(real);
	int x_grad_size = ni*row*col*batch_size*sizeof(real);
	int w_size = ni*no*k*k*sizeof(real);
	int w_grad_size = ni*no*k*k*batch_size*sizeof(real);
	int x_offset = 0;
	int z_grad_offset = x_size;
	int x_grad_offset = z_grad_offset+z_grad_size;

	real* x = (real*)malloc(x_size);
	real* w = (real*)malloc(w_size);
	real* z_grad = (real*)malloc(z_grad_size);
	real* x_grad = (real*)malloc(x_grad_size);
	real* w_grad = (real*)malloc(w_grad_size);

	max_file_t *maxfile = CNN_BP_Conv_V2_init();
	max_engine_t *engine = max_load(maxfile, "*");

	printf("Writing to LMem.\n");
	max_actions_t* act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", z_grad_offset);
	max_set_param_uint64t(act, "size", z_grad_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", z_grad, z_grad_size);
	max_run(engine, act);
	printf("Writing to LMem.\n");
	act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", x_offset);
	max_set_param_uint64t(act, "size", x_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", x, x_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ni", ni);
	max_set_param_uint64t(act, "no", no);
	max_set_param_uint64t(act, "z_grad_offset", z_grad_offset);
	max_set_param_uint64t(act, "x_offset", x_offset);
	max_set_param_uint64t(act, "x_grad_offset", x_grad_offset);
	max_queue_input(act, "w", w, w_size);
	max_queue_output(act, "w_grad", w_grad, w_grad_size);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", x_grad_offset);
	max_set_param_uint64t(act, "size", x_grad_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", x_grad, x_grad_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(x);
	free(w);
	free(z_grad);
	free(x_grad);
	free(w_grad);

	return 0;
}
