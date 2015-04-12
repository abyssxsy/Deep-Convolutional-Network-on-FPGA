import os
import sys
import time
import gzip
import cPickle
import numpy
import shutil
import imp
import helper

modules_name = [
    'CNN_BP_Conv_V0_DP_L0_0',
    'CNN_BP_Conv_V0_DP_L1_0',
    'CNN_BP_MaxPool_V0_DP_L0_0',
    'CNN_BP_MaxPool_V0_DP_L1_0',
    'CNN_BP_Softmax_V0_DP_L3_0',
    'CNN_FW_Conv_V0_DP_L0_0',
    'CNN_FW_Conv_V0_DP_L1_0',
    'CNN_FW_MaxPool_V0_DP_L0_0',
    'CNN_FW_MaxPool_V0_DP_L1_0',
    'CNN_FW_Softmax_V0_DP_L3_0',
]
modules_info = [
    ('%s' %(x), '../../%(pjn)s/RunRules/DFE/maxfiles/%(pjn)s.h' %(dict(pjn=x)),  '../../%(pjn)s/RunRules/DFE/objects/maxfiles/slic_%(pjn)s.o' %(dict(pjn=x)),    )
    for x in modules_name
]

reserved_files = set([
    'conv_mlp.py','conv_mlp.pyc',
])

def load_original_datasets(filename='../data/mnist.pkl.gz'):
    inf = gzip.open(filename, 'rb')
    train_set,valid_set, test_set = cPickle.load(inf)
    inf.close()
    return train_set, valid_set, test_set

def transform_datasets(datasets,batch_size=384,dtype=numpy.float32):
    def strip_dataset(data_xy,batch_size,dtype):
        data_x, data_y = data_xy
        assert(data_x.shape[0]==data_y.shape[0])
        s0,s1 = data_x.shape
        s0 = (s0/batch_size)*batch_size
        dy = data_y[:s0].reshape(batch_size,s0/batch_size).transpose()
        dx = data_x[:s0,:].reshape(batch_size,s0*s1/batch_size).transpose().reshape(s0/batch_size,s1,batch_size)
        return numpy.array(dx,dtype=dtype),numpy.array(dy,dtype=dtype)
    return [strip_dataset(x,batch_size,dtype) for x in datasets]

def save_datasets(datasets,bin_filename='../data/data.bin',c_dir_name='./data/datasets'):
    with open(bin_filename,'wb') as ouf:
        cPickle.dump(datasets,ouf)
    if not os.path.isdir(c_dir_name):
        os.makedirs(c_dir_name)
    train_set, valid_set, test_set = datasets
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    train_x.tofile(os.path.join(c_dir_name,'train_x.bin'))
    train_y.tofile(os.path.join(c_dir_name,'train_y.bin'))
    valid_x.tofile(os.path.join(c_dir_name,'valid_x.bin'))
    valid_y.tofile(os.path.join(c_dir_name,'valid_y.bin'))
    test_x.tofile(os.path.join(c_dir_name,'test_x.bin'))
    test_y.tofile(os.path.join(c_dir_name,'test_y.bin'))

def load_datasets(filename='../data/data.bin'):
    with open(filename,'rb') as inf:
        ret = cPickle.load(inf)
    return ret

def gen_conv_params(rng,ni,no,factor,fan_in,fan_out,dtype=numpy.float32):
    W_bound = numpy.sqrt(6./(fan_in+fan_out))
    w_values = numpy.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=ni*no*factor),
        dtype=dtype
    )
    b_values = numpy.zeros(
        (no,),
        dtype=dtype
    )
    return w_values, b_values

def save_params(params=None,c_dir_name='./data/step_0',dtype=numpy.float32,nkerns=[20, 50, 500, 10]):
    if not os.path.isdir(c_dir_name):
        os.makedirs(c_dir_name)
    if params is None:
        rng = numpy.random.RandomState(23455)
        my_layer0_w, my_layer0_b = gen_conv_params(
            rng,
            ni = 1,
            no = nkerns[0],
            factor = 5*5,
            fan_in = 1*5*5,
            fan_out = nkerns[0]*5*5/(2*2),
            dtype = dtype
        )
        my_layer1_w, my_layer1_b = gen_conv_params(
            rng,
            ni = nkerns[0],
            no = nkerns[1],
            factor = 5*5,
            fan_in = nkerns[0]*5*5,
            fan_out = nkerns[1]*5*5/(2*2),
            dtype = dtype
        )
        my_layer2_w, my_layer2_b = gen_conv_params(
            rng,
            ni = nkerns[1]*4*4,
            no = nkerns[2],
            factor = 1,
            fan_in = nkerns[1]*4*4,
            fan_out = nkerns[2],
            dtype = dtype
        )
        my_layer3_w = numpy.zeros(
            shape = (nkerns[2],nkerns[3]),
            dtype = dtype
        )
        my_layer3_b = numpy.zeros(
            shape = (nkerns[3],),
            dtype = dtype
        )
    else:
        my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b = params
    my_layer0_w.tofile(os.path.join(c_dir_name,'layer0_w.bin'))
    my_layer0_b.tofile(os.path.join(c_dir_name,'layer0_b.bin'))
    my_layer1_w.tofile(os.path.join(c_dir_name,'layer1_w.bin'))
    my_layer1_b.tofile(os.path.join(c_dir_name,'layer1_b.bin'))
    my_layer2_w.tofile(os.path.join(c_dir_name,'layer2_w.bin'))
    my_layer2_b.tofile(os.path.join(c_dir_name,'layer2_b.bin'))
    my_layer3_w.tofile(os.path.join(c_dir_name,'layer3_w.bin'))
    my_layer3_b.tofile(os.path.join(c_dir_name,'layer3_b.bin'))

def load_params(c_dir_name='./data/step_0',dtype=numpy.float32):
    my_layer0_w = numpy.fromfile(os.path.join(c_dir_name,'layer0_w.bin'),dtype=dtype)
    my_layer0_b = numpy.fromfile(os.path.join(c_dir_name,'layer0_b.bin'),dtype=dtype)
    my_layer1_w = numpy.fromfile(os.path.join(c_dir_name,'layer1_w.bin'),dtype=dtype)
    my_layer1_b = numpy.fromfile(os.path.join(c_dir_name,'layer1_b.bin'),dtype=dtype)
    my_layer2_w = numpy.fromfile(os.path.join(c_dir_name,'layer2_w.bin'),dtype=dtype)
    my_layer2_b = numpy.fromfile(os.path.join(c_dir_name,'layer2_b.bin'),dtype=dtype)
    my_layer3_w = numpy.fromfile(os.path.join(c_dir_name,'layer3_w.bin'),dtype=dtype)
    my_layer3_b = numpy.fromfile(os.path.join(c_dir_name,'layer3_b.bin'),dtype=dtype)
    return (my_layer0_w,my_layer0_b,
            my_layer1_w,my_layer1_b,
            my_layer2_w,my_layer2_b,
            my_layer3_w,my_layer3_b)

def my_evaluate_lenet5(learning_rate=0.1,n_epochs=1,
    dataset_filename='../data/data.bin', params_c_dir_name='./data/step_0',
    nkerns=[20, 50, 500, 10], batch_size=384, dtype=numpy.float32):
    type_size = numpy.dtype(dtype).itemsize
    
    t0 = time.time()
    tl = [t0]

    print '[INFO] loading data'
    datasets = load_datasets(dataset_filename)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]

    my_layer0_filter_shape = (1, nkerns[0], 5, 5)
    my_layer0_image_shape = (batch_size, 28, 28, 24, 24, 12, 12)
    my_layer1_filter_shape = (nkerns[0], nkerns[1], 5, 5)
    my_layer1_image_shape = (batch_size, 12,12, 8,8, 4,4)
    my_layer2_shape = (batch_size, nkerns[1]*4*4,nkerns[2])
    my_layer3_shape = (batch_size, nkerns[2],nkerns[3])
    my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b = load_params(params_c_dir_name,dtype)
    my_layer0_w = my_layer0_w.reshape(1,nkerns[0],5,5)
    my_layer0_b = my_layer0_b.reshape(nkerns[0],)
    my_layer1_w = my_layer1_w.reshape(nkerns[0],nkerns[1],5,5)
    my_layer1_b = my_layer1_b.reshape(nkerns[1],)
    my_layer2_w = my_layer2_w.reshape(nkerns[1]*4*4,nkerns[2])
    my_layer2_b = my_layer2_b.reshape(nkerns[2],)
    my_layer3_w = my_layer3_w.reshape(nkerns[2],nkerns[3])
    my_layer3_b = my_layer3_b.reshape(nkerns[3],)

    t0 = time.time()
    print '[INFO] load data done, time = %f' %(t0-tl[-1])
    tl.append(t0)

    print '[INFO] training'
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    mtls = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False
    result_logs = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            print 'training @ epoch = [%d,%d/%d] iter = %d' %(epoch,minibatch_index,n_train_batches,iter)
            print 'Time delta : ',time.time()-t0; t0 = time.time()
            my_oa_input = train_set_x[minibatch_index].reshape(1,28,28,384)
            my_oa_output = train_set_y[minibatch_index]

            my_process = helper.my_process_0
            my_process(my_oa_input,my_oa_output,
                (my_layer0_filter_shape,my_layer0_image_shape,my_layer1_filter_shape,my_layer1_image_shape,my_layer2_shape,my_layer3_shape),
                my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b,
                learning_rate=learning_rate)
            save_params(
                params=(my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b),
                c_dir_name='./data/p_step_%d' %(iter)
            )
            if (iter+1) % validation_frequency == 0:
                my_vls = [my_process(valid_set_x[i].reshape(1,28,28,384),valid_set_y[i],
                    (my_layer0_filter_shape,my_layer0_image_shape,my_layer1_filter_shape,my_layer1_image_shape,my_layer2_shape,my_layer3_shape),
                    my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b,
                    learning_rate=None) for i in xrange(n_valid_batches)]
                mvls = numpy.mean(my_vls)
                result_logs.append((mvls,mtls))
                print('epoch %i, minibatch %i/%i, validation error %f %%' %(epoch, minibatch_index, n_train_batches, mvls*100.))
                if mvls < best_validation_loss:
                    if mvls<best_validation_loss*improvement_threshold:
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = mvls
                    best_iter = iter
                    my_tls = [my_process(test_set_x[i].reshape(1,28,28,384),test_set_y[i],
                        (my_layer0_filter_shape,my_layer0_image_shape,my_layer1_filter_shape,my_layer1_image_shape,my_layer2_shape,my_layer3_shape),
                        my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b,
                        learning_rate=None) for i in xrange(n_test_batches)]
                    mtls = numpy.mean(my_vls)
                    print(('\tepoch %i, minibatch %i/%i, test error of best model %f %%') %(epoch, minibatch_index + 1, n_train_batches,mtls*100.))
            if patience <= iter:
                done_looping = True
                break
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, with test performance %f %%' %(best_validation_loss*100., best_iter, mtls*100.))

def generate_files(modules_info=modules_info,header_filename='max_header.h'):
    lines = []
    for x in modules_info:
        name, header, obj_file = x
        lines.append('#include"%s"\n' %(header))
        lnnm = 'slic_%s.o' %(name)
        if os.path.lexists(lnnm):
            os.remove(lnnm)
        os.symlink(obj_file,lnnm)
    with open(header_filename,'w') as ouf:
        ouf.writelines(lines)

def prepare():
    tmp0 = load_original_datasets()
    tmp1 = transform_datasets(tmp0,dtype=numpy.float64)
    save_datasets(tmp1)
    save_params(dtype=numpy.float64)
    generate_files()

if __name__=='__main__':
    my_evaluate_lenet5(dtype=numpy.float64)
