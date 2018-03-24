import numpy as np

def im2col_flat(x,field_height,field_width,padding,stride):
    
    #takes x and reshapes into a volume where each receptive field is a column vector
        
    N,H, W, C = x.shape
    
    n_H = int(((H - field_height + 2 * padding)/stride)+1)
    n_W = int(((W - field_width + 2 * padding)/stride)+1)
    
    p = padding
    x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')

    # Create indicies for each position of our convolution
        
    filters_h = np.repeat(np.arange(field_height),field_width)
    
    height = stride * np.repeat(np.arange(n_H),n_W).reshape(-1,1)
    height_index = filters_h + height
    
    i = np.tile(height_index,C)
    
    filters_w = np.tile(np.arange(field_width),field_height)
    width = stride * np.tile(np.arange(n_W),n_H).reshape(-1,1)
    width_index = filters_w + width

    j = np.tile(width_index,C)

    k = np.repeat(np.arange(C),field_height*field_width)
    
    
    return x_padded[:,i,j,k],(i,j,k)


def conv_fast(x,w_filter,b,parameters):
    

    padding = parameters["pad"]
    stride = parameters["stride"]

    cache = (x,w_filter,b,padding,stride)

    
    N,H, W, C = x.shape
    
    assert (H + 2 * padding - w_filter.shape[0]) % stride == 0
    assert (W + 2 * padding - w_filter.shape[1]) % stride == 0

    n_H = int(((H - w_filter.shape[0]+ 2 * padding)/stride)+1)
    n_W = int(((W - w_filter.shape[1] + 2 * padding)/stride)+1)

    
    flat,dims = im2col_flat(x,w_filter.shape[0],w_filter.shape[1],padding,stride)    
    filter_flat = w_filter[:,:,:,:].reshape(-1,w_filter.shape[2],w_filter.shape[3]).T.reshape(w_filter.shape[3],-1).T
    
    #automatic broadcasting will handle this for us 
    
    conv = flat.dot(filter_flat)
                 
    #now the final reshape
    conv = conv.reshape(N,n_H,n_W,w_filter.shape[3])
    return conv + b[None,None,None,:], cache


def col2im_flat(x_shape,dims,col,padding,stride):
    
    #we want to ad d together all of the ones at each position
    
    n, h, w, c = x_shape
    
    padded_output = np.zeros((n,h + 2 * padding, w + 2 *padding,c))    
    #create index array that tells index for each training example
    dims3 = np.zeros(col.shape,dtype=int)
    # repeat index into training examples for number of receptive fields
    dims3 = np.repeat(np.arange(col.shape[0])[:,None],col.shape[1],axis=1)

    # Add together the filter positions that each filter has influence over
    np.add.at(padded_output,[dims3[:,:,None],dims[0],dims[1],dims[2]],col)
    
    out = padded_output[:,padding:padding+h,padding:padding+w,:]
    
    return out


def conv_fast_back(dout,cache):


    x,w_filter,b,padding,stride = cache

    
    f_h, f_w, c, f =  w_filter.shape
    
    n, H, W, c = x.shape

    n_H = int(((H - w_filter.shape[0]+ 2 * padding)/stride)+1)
    n_W = int(((W - w_filter.shape[1] + 2 * padding)/stride)+1)
        
    # dw
    flat,dims = im2col_flat(x,f_h,f_w,padding,stride)
    
    dout_reshape = dout.reshape(dout.shape[0],dout.shape[1]*dout.shape[2],dout.shape[3])
    dout_reshape = np.repeat(dout_reshape[:,:,None,:],flat.shape[2],axis=2)
    flat = np.repeat(flat[:,:,:,None],w_filter.shape[3],axis=3)
    dw = flat * dout_reshape
    dw= np.sum(dw,axis=(0,1))
    dw = dw.reshape(c,f_h,f_w,f)
    dw = np.moveaxis(dw,0,2)
    
    # Finding dx is simply taking the flattened filter matrix and reshaping it into the
    # input shape 
    
    # TODO: This is not outputting the correct value
    doutr = dout.reshape(dout.shape[0],dout.shape[1]*dout.shape[2],dout.shape[3])
    doutr = np.moveaxis(doutr,2,1)
    #reshape filter into a flat column vector 
    filter_flat = w_filter.reshape(-1,w_filter.shape[2],w_filter.shape[3]).T.reshape(w_filter.shape[3],-1)
    # repeat for the number of receptive fields
    filter_flat = np.repeat(filter_flat[:,None,:],n_H*n_W,axis=1)
    
    # automatic broadcasting will take care of this
    filter_flat = np.repeat(filter_flat[None,:,:,:],n,axis=0)
    filter_flat = filter_flat * doutr[:,:,:,None]

    filter_flat = np.sum(filter_flat,axis=1)
    #Repeat for the number of training examples
    
    # Now take our flattened filter volume and transform it back into the size of the input image    
    dx = col2im_flat(x.shape,dims,filter_flat,padding,stride)
    
    db = np.sum(dout,axis=(0,1,2))
    
    return dx,dw,db 
