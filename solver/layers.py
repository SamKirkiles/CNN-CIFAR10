
import numpy as np

def conv_forward_naive(x,weight,b,parameters):

	pad = parameters['pad']
	stride = parameters['stride']

	(m, n_h, n_w, n_C_prev) = x.shape
	(f,f, n_C_prev, n_C) = weight.shape

	n_H = int(1 + (n_h + 2 * pad - f) / stride)
	n_W = int(1 + (n_w + 2 * pad - f) / stride)
	    
	x_prev_pad = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)

	Z = np.zeros((m, n_H,n_W,n_C))

	caches = (x,weight,b,pad,stride)
	        
	for i in range(m):
	    for h in range(n_H):
	        for w in range(n_W):
	            for c in range(n_C):

	                    vert_start = h*stride
	                    vert_end = vert_start + f
	                    horiz_start = w * stride
	                    horiz_end = horiz_start + f

	                    x_slice = x_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
	                    Z[i,h,w,c] = np.sum(np.multiply(x_slice, weight[:,:,:,c]))

	return Z + b[None,None,None,:], caches

def conv_back_naive(dout,cache):

	x,w_filter,b,pad,stride = cache

	(m, n_h, n_w, n_C_prev) = x.shape
	(f,f, n_C_prev, n_C) = w_filter.shape

	n_H = int(1 + (n_h + 2 * pad - f) / stride)
	n_W = int(1 + (n_w + 2 * pad - f) / stride)

	a_prev_pad = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)

	dw = np.zeros(w_filter.shape,dtype=np.float32)
	dx = np.zeros(x.shape,dtype=np.float32)

	for h in range(f):
	    for w in range(f):
	        for p in range(n_C_prev):
	            for c in range(n_C):

	                # go through all the individual positions that this filter affected and multiply by their dout
	                a_slice = a_prev_pad[:,h:h + n_H * stride:stride,w:w + n_W * stride:stride,p]
	                 
	                dw[h,w,p,c] = np.sum(a_slice * dout[:,:,:,c])
	                
	# TODO: put back in dout to get correct gradient
	dx_pad = np.pad(dx, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
	    
	for i in range(m):        
	    for h_output in range(n_H):
	        for w_output in range(n_W):
	            for g in range(n_C):
	                        
	                vert_start = h_output*stride
	                vert_end = vert_start + f
	                horiz_start = w_output * stride
	                horiz_end = horiz_start + f
	                dx_pad[i,vert_start:vert_end,horiz_start:horiz_end,:] += w_filter[:,:,:,g] * dout[i,h_output,w_output,g]
	                                
	            
	dx = dx_pad[:,pad:pad+n_h,pad:pad+n_w,:]
	            
	db = np.sum(dout,axis=(0,1,2))

	return dx,dw,db

def relu(x):
    return np.maximum(0, x)

def relu_back(x,dout):
    dx = np.array(dout, copy=True)
    dx[x <= 0] = 0

    return dx


def max_pooling(prev_layer, filter_size=2):

	(m, n_H_prev, n_W_prev, channels) = prev_layer.shape

	stride = 2


	# with max pooling I dont want overlapping filters so make stride = filter size
	n_H = int((n_H_prev - filter_size)/filter_size + 1)
	n_W = int((n_W_prev - filter_size)/filter_size + 1)

	pooling = np.zeros((m,n_H,n_W,channels))

	for i in range(m):
	    for h in range(n_H):
	        for w in range(n_W):
	            for c in range(channels):
	                vert_start = h*filter_size
	                vert_end = vert_start + filter_size
	                horiz_start = w*filter_size
	                horiz_end = horiz_start + filter_size
	            
	                prev_slice = prev_layer[i,vert_start:vert_end,horiz_start:horiz_end,c]
	                
	                pooling[i,h,w,c] = np.max(prev_slice)

	caches = (pooling,prev_layer,filter_size)                    
	                
	return pooling, caches


def max_pooling_back(dout, caches):
        
    pool, prev, filter_size = caches

    (m, n_H, n_W, channels) = pool.shape
    (m_prev, n_prev_H, n_prev_W, channels_prev) = prev.shape
    
    empty = np.zeros((m, n_prev_H, n_prev_W, channels))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(channels):
                    
                    vert_start = h*filter_size
                    vert_end = vert_start + filter_size
                    horiz_start = w*filter_size
                    horiz_end = horiz_start + filter_size
                                        
                    mask = prev[i,vert_start:vert_end,horiz_start:horiz_end,c] == pool[i,h,w,c]
     
                    empty[i,vert_start:vert_end,horiz_start:horiz_end,c] = mask * dout[i,h,w,c]
                    
    return empty



def fully_connected(prev_layer, w,b):
    fc = prev_layer.dot(w) + b
    caches = {'input':prev_layer,'weights':w,'bias':b}
    return fc, caches

def fully_connected_backward(dout,caches):
    x_input = caches['input']
    w = caches['weights']
    b = caches['bias']

    da = (w.dot(dout.T)).T
    dw = x_input.T.dot(dout)
    db = np.sum(dout,axis=0)

    return da,dw,db

def softmax_cost(y, y_hat):
    return -np.sum(y * np.log(y_hat),axis=1)


def softmax(z):
	return np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)

def softmax_back(softmax, Y):    
    return (softmax-Y)/softmax.shape[0]


