import os
import theano
import theano.tensor as tensor
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous

class PoolInvOp(GpuOp):

    def __init__(self, size):
        self.size = size

    def __eq__(self, other):
        return type(self) == type(other) and self.size == other.size

    def __hash__(self):
        return hash((type(self), self.size[0], self.size[1]))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, fmap):    
        fmap = as_cuda_ndarray_variable(fmap)
        assert fmap.ndim == 4
        return theano.Apply(self, [fmap], [fmap.type()])

    def infer_shape(self, node, in_shapes):
        b,f,h,w = in_shapes[0]
        return [(b, f, h*self.size[1], w*self.size[0])]

    def grad(self, inputs, output_grads):
        dy, = output_grads
        return [PoolInvGradOp(self.size)(dy)]

    def c_support_code(self):
        size_w = self.size[0]
        size_h = self.size[1]
        return """
        static __global__ void k_pool_inv_%(size_w)ix%(size_h)i(float* fmap, size_t fs0, size_t fs1, size_t fs2, size_t fs3, 
                                                                float* r, size_t rs0, size_t rs1, size_t rs2, size_t rs3, 
                                                                size_t bs, size_t fn, size_t h, size_t w){

            size_t index = blockIdx.x*blockDim.x + threadIdx.x;
            if (index >= (bs*h*w))
                return;

            int b,y,x;
            b = index / (h*w);
            index -= b*(h*w);
            y = index / w;
            index -= y*w;
            x = index;
            //printf("b=%%i,y=%%i,x=%%i\\n",b,y,x);

            const size_t size_w = %(size_w)i;
            const size_t size_h = %(size_h)i;
            for(size_t f=0; f < fn; f++){
                for(size_t ry = y*size_h; ry < (y*size_h + size_h); ry++){
                    for(size_t rx = x*size_w; rx < (x*size_w + size_w); rx++){
                        r[b*rs0 + f*rs1 + ry*rs2 + rx*rs3] = fmap[b*fs0 + f*fs1 + y*fs2 + x*fs3];
                    }
                }
            }
        };
        """%locals()
        
    def c_code(self, node, name, inputs, output, sub):
        fmap, = inputs
        result, = output
        fail = sub['fail']
        size_w = self.size[0]
        size_h = self.size[1]
        return """
        size_t batch_size = CudaNdarray_HOST_DIMS(%(fmap)s)[0];        
        size_t feature_num = CudaNdarray_HOST_DIMS(%(fmap)s)[1];        
        size_t height = CudaNdarray_HOST_DIMS(%(fmap)s)[2];        
        size_t width = CudaNdarray_HOST_DIMS(%(fmap)s)[3];        

        //choose grid / block dim
        size_t total_threads = batch_size*height*width;
        size_t threads_per_block = 1024;
        size_t grid_num = std::ceil((double)total_threads / threads_per_block);
        dim3 grid_dim(grid_num, 1, 1);
        dim3 block_dim(threads_per_block, 1, 1); 
        cudaError_t err;
        
        const size_t size_w = %(size_w)i;
        const size_t size_h = %(size_h)i;
        int dims[] = {batch_size, feature_num, height*size_h, width*size_w};
        if (CudaNdarray_prep_output(&%(result)s, 4, dims) != 0){
            PyErr_Format(PyExc_RuntimeError, "Failed to allocate output array");
            %(fail)s
        }

        //printf("result shape =(%%i,%%i,%%i,%%i)", CudaNdarray_HOST_DIMS(%(result)s)[0], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[1], CudaNdarray_HOST_DIMS(%(result)s)[2], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[3]);
        //printf("batch_size %%i, feature_num=%%i, grid_size=%%i, sample_num=%%i\\n", batch_size, feature_num, grid_size, sample_num);
        k_pool_inv_%(size_w)ix%(size_h)i<<<grid_dim, block_dim>>>(
            CudaNdarray_DEV_DATA(%(fmap)s), 
            CudaNdarray_HOST_STRIDES(%(fmap)s)[0],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[1],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[2],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[3],
            CudaNdarray_DEV_DATA(%(result)s), 
            CudaNdarray_HOST_STRIDES(%(result)s)[0],
            CudaNdarray_HOST_STRIDES(%(result)s)[1],
            CudaNdarray_HOST_STRIDES(%(result)s)[2],
            CudaNdarray_HOST_STRIDES(%(result)s)[3], 
            batch_size, feature_num, height, width);
        
        CNDA_THREAD_SYNC;             
        err = cudaGetLastError();
        if (err != cudaSuccess){
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s:\\n", cudaGetErrorString(err));
            %(fail)s
        }
        """%locals()

    def c_code_cache_version(self):
        return (0,5)

class PoolInvGradOp(GpuOp):
    def __init__(self, size):
        self.size = size

    def __eq__(self, other):
        return type(self) == type(other) and self.size == other.size

    def __hash__(self):
        return hash((type(self), self.size[0], self.size[1]))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy):    
        dy = as_cuda_ndarray_variable(dy)
        assert dy.ndim == 4
        return theano.Apply(self, [dy], [dy.type()])

    def c_support_code(self):
        size_w = self.size[0]
        size_h = self.size[1]
        return """
        static __global__ void k_pool_inv_grad_%(size_w)ix%(size_h)i(float* fmap, size_t fs0, size_t fs1, size_t fs2, size_t fs3, 
                                                                     float* r, size_t rs0, size_t rs1, size_t rs2, size_t rs3, 
                                                                     size_t bs, size_t fn, size_t h, size_t w){

            size_t index = blockIdx.x*blockDim.x + threadIdx.x;
            if (index >= (bs*h*w))
                return;

            int b,y,x;
            b = index / (h*w);
            index -= b*(h*w);
            y = index / w;
            index -= y*w;
            x = index;

            const size_t size_w = %(size_w)i;
            const size_t size_h = %(size_h)i;
            for(size_t f=0; f < fn; f++){
                r[b*rs0 + f*rs1 + y*rs2 + x*rs3] = 0.0;
                for(size_t ry = y*size_h; ry < (y*size_h + size_h); ry++){
                    for(size_t rx = x*size_w; rx < (x*size_w + size_w); rx++){
                        r[b*rs0 + f*rs1 + y*rs2 + x*rs3] += fmap[b*fs0 + f*fs1 + ry*fs2 + rx*fs3];
                    }
                }
            }
        };
        """%locals()
        
    def c_code(self, node, name, inputs, output, sub):
        dy, = inputs
        result, = output
        fail = sub['fail']
        size_w = self.size[0]
        size_h = self.size[1]
        return """
        size_t batch_size = CudaNdarray_HOST_DIMS(%(dy)s)[0];        
        size_t feature_num = CudaNdarray_HOST_DIMS(%(dy)s)[1];        
        size_t height = CudaNdarray_HOST_DIMS(%(dy)s)[2] / %(size_h)i;        
        size_t width = CudaNdarray_HOST_DIMS(%(dy)s)[3] / %(size_w)i;        

        //choose grid / block dim
        size_t total_threads = batch_size*height*width;
        size_t threads_per_block = 1024;
        size_t grid_num = std::ceil((double)total_threads / threads_per_block);
        dim3 grid_dim(grid_num, 1, 1);
        dim3 block_dim(threads_per_block, 1, 1); 
        cudaError_t err;
        
        int dims[] = {batch_size, feature_num, height, width};
        if (CudaNdarray_prep_output(&%(result)s, 4, dims) != 0){
            PyErr_Format(PyExc_RuntimeError, "Failed to allocate output array");
            %(fail)s
        }

        k_pool_inv_grad_%(size_w)ix%(size_h)i<<<grid_dim, block_dim>>>(
            CudaNdarray_DEV_DATA(%(dy)s), 
            CudaNdarray_HOST_STRIDES(%(dy)s)[0],
            CudaNdarray_HOST_STRIDES(%(dy)s)[1],
            CudaNdarray_HOST_STRIDES(%(dy)s)[2],
            CudaNdarray_HOST_STRIDES(%(dy)s)[3],
            CudaNdarray_DEV_DATA(%(result)s), 
            CudaNdarray_HOST_STRIDES(%(result)s)[0],
            CudaNdarray_HOST_STRIDES(%(result)s)[1],
            CudaNdarray_HOST_STRIDES(%(result)s)[2],
            CudaNdarray_HOST_STRIDES(%(result)s)[3], 
            batch_size, feature_num, height, width);
        
        CNDA_THREAD_SYNC;             
        err = cudaGetLastError();
        if (err != cudaSuccess){
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s:\\n", cudaGetErrorString(err));
            %(fail)s
        }
        """%locals()

    def c_code_cache_version(self):
        return (1,0)
    
    

