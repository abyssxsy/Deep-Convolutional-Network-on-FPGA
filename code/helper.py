import numpy

def f_tanh(x):
    return numpy.tanh(x)
def f_tanh_grad(x):
    return 1./numpy.square(numpy.cosh(x))
def f_softmax(x):
    sm = numpy.zeros_like(x)
    for i in xrange(sm.shape[1]):
        sm[:,i] = numpy.exp(x[:,i] - numpy.max(x[:,i]))
        sm[:,i] *= 1.0 / numpy.sum(sm[:,i])
    return sm
def f_softmax_grad(x):
    sm = f_softmax(x)
    return sm * (1.-sm)
def check_same(x,y,name='',threshold_a=1e9,threshold_r=1e-4):
    if x.shape!=y.shape:
        print "Check_same [%s] failed, shape not match :" %(name),x.shape,y.shape
        return False
    max_abs_value = float(max(numpy.max(numpy.abs(x)),numpy.max(numpy.abs(y))))
    max_dif_value = float(numpy.max(numpy.abs(x-y)))
    rat = 0.0 if max_abs_value<min(1e-12,threshold_a) else max_dif_value/max_abs_value
    print "Check_same [%s], dif = %s, abs = %s, rat = %s" %(name,max_dif_value,max_abs_value,rat)
    return (max_dif_value<threshold_a) and (rat<threshold_r)

def forward_convpool(filter_shape,image_shape, my_w,my_b, my_input):
    ni_size, no_size, conv_kernel_size_r, conv_kernel_size_c = filter_shape
    bs_size, conv_input_r, conv_input_c, conv_output_r, conv_output_c, pooled_output_r, pooled_output_c = image_shape
    assert(my_w.shape==filter_shape)
    assert(my_b.shape==(no_size,))
    assert(my_input.shape==(ni_size, conv_input_r, conv_input_c, bs_size))
    conv_out_shape = (no_size, conv_output_r, conv_output_c, bs_size)
    my_conv_out = numpy.zeros(shape=conv_out_shape, dtype=numpy.float64)
    for ni in xrange(ni_size):
        for no in xrange(no_size):
            for dr in xrange(conv_kernel_size_r):
                for dc in xrange(conv_kernel_size_c):
                    for bi in xrange(bs_size):
                        my_conv_out[no,:,:,bi] += my_input[ni,dr:(dr+conv_output_r),dc:(dc+conv_output_c),bi]*my_w[ni,no,dr,dc]
    my_pooled_out_shape = (no_size,pooled_output_r,pooled_output_c,bs_size)
    my_pooled_out = numpy.zeros(shape=my_pooled_out_shape,dtype=numpy.float64)
    for no in xrange(no_size):
        for r in xrange(pooled_output_r):
            for c in xrange(pooled_output_c):
                for bi in xrange(bs_size):
                    my_pooled_out[no,r,c,bi] = numpy.max(my_conv_out[no,r+r:r+r+2,c+c:c+c+2,bi])
    my_before_tanh = my_pooled_out+numpy.repeat(my_b,pooled_output_r*pooled_output_r*bs_size).reshape(my_pooled_out_shape)
    my_output = numpy.tanh(my_before_tanh)
    return (my_conv_out,my_pooled_out,my_before_tanh,my_output)
def backward_convpool(filter_shape,image_shape, my_w, my_input,my_conv_out,my_pooled_out,my_before_tanh, my_output_grad):
    ni_size, no_size, conv_kernel_size_r, conv_kernel_size_c = filter_shape
    bs_size, conv_input_r, conv_input_c, conv_output_r, conv_output_c, pooled_output_r, pooled_output_c = image_shape
    assert(my_w.shape==filter_shape)
    assert(my_input.shape==(ni_size, conv_input_r, conv_input_c, bs_size))
    assert(my_conv_out.shape==(no_size, conv_output_r, conv_output_c, bs_size))
    assert(my_pooled_out.shape==(no_size,pooled_output_r,pooled_output_c,bs_size))
    assert(my_before_tanh.shape==my_pooled_out.shape)
    assert(my_output_grad.size==my_pooled_out.size)
    bp_z_pooled_grad = my_output_grad.reshape(my_pooled_out.shape)*f_tanh_grad(my_before_tanh)
    bp_z_grad = numpy.zeros(shape=my_conv_out.shape,dtype=numpy.float64)
    for no in xrange(no_size):
        for r in xrange(conv_output_r):
            for c in xrange(conv_output_c):
                for bi in xrange(bs_size):
                    bp_z_grad[no,r,c,bi] = bp_z_pooled_grad[no,r/2,c/2,bi] if my_conv_out[no,r,c,bi]==my_pooled_out[no,r/2,c/2,bi] else 0.
    bp_w_grad = numpy.zeros(shape=my_w.shape,dtype=numpy.float64)
    bp_x_grad = numpy.zeros(shape=my_input.shape,dtype=numpy.float64)
    for ni in xrange(ni_size):
        for no in xrange(no_size):
            for dr in xrange(conv_kernel_size_r):
                for dc in xrange(conv_kernel_size_c):
                    bp_w_grad[ni,no,dr,dc] += numpy.sum(bp_z_grad[no,:,:,:]*my_input[ni,dr:dr+conv_output_r,dc:dc+conv_output_c,:])
                    bp_x_grad[ni,dr:dr+conv_output_r,dc:dc+conv_output_c,:] += bp_z_grad[no,:,:,:]*my_w[ni,no,dr,dc]
    bp_b_grad = bp_z_pooled_grad.sum(axis=(1,2,3))
    return (bp_b_grad,bp_w_grad,bp_x_grad,bp_z_pooled_grad,bp_z_grad)
def my_forward(my_shapes,my_paras,my_oa_input,my_oa_output):
    my_layer0_filter_shape,my_layer0_image_shape,my_layer1_filter_shape,my_layer1_image_shape,my_layer2_shape,my_layer3_shape = my_shapes
    my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b = my_paras
    my_layer0_conv_out,my_layer0_pooled_out,my_layer0_before_tanh,my_layer0_output = forward_convpool(
        my_layer0_filter_shape,my_layer0_image_shape,
        my_layer0_w,my_layer0_b,my_oa_input
    )
    my_layer1_conv_out,my_layer1_pooled_out,my_layer1_before_tanh,my_layer1_output = forward_convpool(
        my_layer1_filter_shape,my_layer1_image_shape,
        my_layer1_w,my_layer1_b,my_layer0_output
    )
    bs,ni,no = my_layer2_shape
    my_layer2_input = my_layer1_output.reshape((ni,bs))
    my_layer2_lin_output = my_layer2_w.transpose().dot(my_layer2_input)+numpy.repeat(my_layer2_b,bs).reshape(no,bs)
    my_layer2_output = f_tanh(my_layer2_lin_output)
    bs,ni,no = my_layer3_shape
    my_layer3_lin_output = my_layer3_w.transpose().dot(my_layer2_output)+numpy.repeat(my_layer3_b,bs).reshape(no,bs)
    my_layer3_softmax = f_softmax(my_layer3_lin_output)
    my_layer3_log = numpy.log(numpy.asarray([my_layer3_softmax[my_oa_output[i],i] for i in xrange(bs)],dtype=numpy.float64))
    my_layer3_llh = -numpy.mean(my_layer3_log)
    my_layer3_y_pred = numpy.argmax(my_layer3_softmax,axis=0)
    return dict(
        my_layer0_conv_out = my_layer0_conv_out,
        my_layer0_pooled_out = my_layer0_pooled_out,
        my_layer0_before_tanh = my_layer0_before_tanh,
        my_layer0_output = my_layer0_output,
        my_layer1_conv_out = my_layer1_conv_out,
        my_layer1_pooled_out = my_layer1_pooled_out,
        my_layer1_before_tanh = my_layer1_before_tanh,
        my_layer1_output = my_layer1_output,
        my_layer2_input = my_layer2_input,
        my_layer2_lin_output = my_layer2_lin_output,
        my_layer2_output = my_layer2_output,
        my_layer3_softmax = my_layer3_softmax,
        my_layer3_log = my_layer3_log,
        my_layer3_llh = my_layer3_llh,
        my_layer3_y_pred = my_layer3_y_pred,
    )
def my_backward(my_shapes,my_paras,for_res_set,my_oa_input,my_oa_output):
    my_layer0_filter_shape,my_layer0_image_shape,my_layer1_filter_shape,my_layer1_image_shape,my_layer2_shape,my_layer3_shape = my_shapes
    my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b = my_paras
    my_layer3_softmax = for_res_set['my_layer3_softmax']
    my_layer2_output = for_res_set['my_layer2_output']
    my_layer2_lin_output = for_res_set['my_layer2_lin_output']
    my_layer2_input = for_res_set['my_layer2_input']
    my_layer0_output = for_res_set['my_layer0_output']
    my_layer1_conv_out = for_res_set['my_layer1_conv_out']
    my_layer1_pooled_out = for_res_set['my_layer1_pooled_out']
    my_layer1_before_tanh = for_res_set['my_layer1_before_tanh']
    my_layer0_conv_out = for_res_set['my_layer0_conv_out']
    my_layer0_pooled_out = for_res_set['my_layer0_pooled_out']
    my_layer0_before_tanh = for_res_set['my_layer0_before_tanh']
    bs,ni,no = my_layer3_shape
    bp_layer3_b_grad_x = numpy.ones(shape=(no,bs),dtype=numpy.float64)
    bp_layer3_b_grad_y = numpy.array([[1 if my_oa_output[iy]==iv else 0 for iy in xrange(bs)] for iv in xrange(no)],dtype=numpy.float64)
    bp_layer3_b_grad_p = my_layer3_softmax
    bp_layer3_b_grad = (bp_layer3_b_grad_x*(bp_layer3_b_grad_p-bp_layer3_b_grad_y)).mean(axis=1)
    bp_layer3_w_grad_x = numpy.tile(my_layer2_output,no).reshape(ni,no,bs)
    bp_layer3_w_grad_y = numpy.tile(bp_layer3_b_grad_y,(ni,1)).reshape(ni,no,bs)
    bp_layer3_w_grad_p = numpy.tile(my_layer3_softmax,(ni,1)).reshape(ni,no,bs)
    bp_layer3_w_grad = (bp_layer3_w_grad_x*(bp_layer3_w_grad_p-bp_layer3_w_grad_y)).mean(axis=2)
    bp_layer3_z_grad = (bp_layer3_b_grad_p-bp_layer3_b_grad_y)/bs
    bp_layer3_x_grad = my_layer3_w.dot(bp_layer3_z_grad)
    bp_layer2_z_grad = bp_layer3_x_grad*f_tanh_grad(my_layer2_lin_output)
    bp_layer2_b_grad = bp_layer2_z_grad.sum(axis=1)
    bp_layer2_w_grad = my_layer2_input.dot(bp_layer2_z_grad.transpose())
    bp_layer2_x_grad = my_layer2_w.dot(bp_layer2_z_grad)
    bp_layer1_output_grad = bp_layer2_x_grad.reshape(my_layer1_filter_shape[1],my_layer1_image_shape[-2],my_layer1_image_shape[-1],bs)
    res_bc1 = backward_convpool(my_layer1_filter_shape,my_layer1_image_shape,
        my_layer1_w,
        my_layer0_output,my_layer1_conv_out,my_layer1_pooled_out,my_layer1_before_tanh,
        bp_layer1_output_grad
    )
    bp_layer1_b_grad,bp_layer1_w_grad,bp_layer1_x_grad,bp_layer1_z_pooled_grad,bp_layer1_z_grad = res_bc1[:5]
    bp_layer0_output_grad = bp_layer1_x_grad
    res_bc0 = backward_convpool(my_layer0_filter_shape,my_layer0_image_shape,
        my_layer0_w,
        my_oa_input,my_layer0_conv_out,my_layer0_pooled_out,my_layer0_before_tanh,
        bp_layer0_output_grad
    )
    bp_layer0_b_grad,bp_layer0_w_grad,bp_layer0_x_grad,bp_layer0_z_pooled_grad,bp_layer0_z_grad = res_bc0[:5]
    return (bp_layer0_w_grad,bp_layer0_b_grad,bp_layer1_w_grad,bp_layer1_b_grad,bp_layer2_w_grad,bp_layer2_b_grad,bp_layer3_w_grad,bp_layer3_b_grad)
def my_process_0(my_oa_input,my_oa_output,my_shapes,
    my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b,
    learning_rate=None):
    my_paras = (my_layer0_w,my_layer0_b,my_layer1_w,my_layer1_b,my_layer2_w,my_layer2_b,my_layer3_w,my_layer3_b)
    res0 = my_forward(my_shapes,my_paras,my_oa_input,my_oa_output)
    res1 = my_backward(my_shapes,my_paras,res0,my_oa_input,my_oa_output)
    bp_layer0_w_grad,bp_layer0_b_grad,bp_layer1_w_grad,bp_layer1_b_grad,bp_layer2_w_grad,bp_layer2_b_grad,bp_layer3_w_grad,bp_layer3_b_grad = res1
    if learning_rate is not None:
        my_layer3_w -= bp_layer3_w_grad*learning_rate
        my_layer3_b -= bp_layer3_b_grad*learning_rate
        my_layer2_w -= bp_layer2_w_grad*learning_rate
        my_layer2_b -= bp_layer2_b_grad*learning_rate
        my_layer1_w -= bp_layer1_w_grad*learning_rate
        my_layer1_b -= bp_layer1_b_grad*learning_rate
        my_layer0_w -= bp_layer0_w_grad*learning_rate
        my_layer0_b -= bp_layer0_b_grad*learning_rate
    my_layer3_softmax = res0['my_layer3_softmax']
    y_pred = numpy.argmax(my_layer3_softmax, axis=1)
    cost = numpy.mean(numpy.asarray(y_pred!=my_oa_output,dtype=numpy.float64))
    return cost
