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
	const int row = 28;
	const int col = 28;
	const int batch_size = 384;

	int x_offset = 0;
	int x_size = ni*row*col*batch_size*sizeof(real);
	int z_offset = x_size;
	int z_size = no*(row-k+1)*(col-k+1)*batch_size*sizeof(real);

	int b_size = no*sizeof(real);
	int w_size = ni*no*k*k*sizeof(real);

	real* b = (real*)malloc(b_size);
	real* x = (real*)malloc(x_size);
	real* z = (real*)malloc(z_size);
	real* w = (real*)malloc(w_size);

	max_file_t *maxfile = CNN_FW_Conv_V0_DP_L0_0_init();
	max_engine_t *engine = max_load(maxfile, "*");
	max_actions_t* act;

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
	max_queue_input(act, "b", b, b_size);
	max_queue_input(act, "w", w, w_size);
	max_set_param_uint64t(act, "x_offset", x_offset);
	max_set_param_uint64t(act, "z_offset", z_offset);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", z_offset);
	max_set_param_uint64t(act, "size", z_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", z, z_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(b);
	free(x);
	free(z);
	free(w);

	return 0;
}
