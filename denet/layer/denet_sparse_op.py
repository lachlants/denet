import os
import theano
import theano.tensor as tensor
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous

class DeNetSparseOp(GpuOp):

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __eq__(self, other):
        return type(self) == type(other) and self.grid_size == other.grid_size

    def __hash__(self):
        return hash((type(self), self.grid_size))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, fmap, bbox):    
        fmap = as_cuda_ndarray_variable(fmap)
        bbox = as_cuda_ndarray_variable(bbox)

        assert fmap.ndim == 4
        assert bbox.ndim == 4
        return theano.Apply(self, [fmap, bbox], [fmap.type()])

    def infer_shape(self, node, in_shapes):
        b,f,_,_ = in_shapes[0]
        _,j,i,_ = in_shapes[1]
        return [(b,f* self.grid_size* self.grid_size+2,j,i)]

    #HACK! return zeros instead of grad_undefined so that training can still use op in switch statement
    def grad(self, inputs, output_grads):
        fmap, bbox = inputs
        dy, = output_grads
        return [DeNetSparseGradOp(self.grid_size)(fmap, bbox, dy), theano.gradient.grad_undefined(self, 1, inputs[1])]

    def c_support_code(self):
        return """
        static __global__ void k_sparse_sample%i(float* fmap, size_t fs0, size_t fs1, size_t fs2, size_t fs3, 
                                                float* bbox, size_t bs0, size_t bs1, size_t bs2, size_t bs3,
                                                float* r, size_t rs0, size_t rs1, size_t rs2, size_t rs3,
                                                size_t fn, size_t h, size_t w, size_t sn, size_t bs){

            size_t index = blockIdx.x*blockDim.x + threadIdx.x;
            if (index >= (bs*sn*sn))
                return;

            int b,j,i;
            b = index / (sn*sn);
            index -= b*sn*sn;
            j = index / sn;
            index -= j*sn;
            i = index;
            //printf("b=%%i,j=%%i,i=%%i,bs=%%i\\n",b,j,i,bs);
        
            const size_t gs = %i;
            size_t bbox_offset = b*bs0 + j*bs1 + i*bs2;
            float bbox_x0 = bbox[bbox_offset + 0*bs3];
            float bbox_y0 = bbox[bbox_offset + 1*bs3];
            float bbox_x1 = bbox[bbox_offset + 2*bs3];
            float bbox_y1 = bbox[bbox_offset + 3*bs3];
            float bbox_h = bbox_y1 - bbox_y0;
            float bbox_w = bbox_x1 - bbox_x0;
            float k = 1.0f / (gs-1);
            
            size_t ff = 0;
            for(size_t yi = 0; yi < gs; yi++){
                float y = bbox_y0 + yi*bbox_h*k;
                size_t ys = lroundf(max(0.0f, min(h - 1.0f, y*h)));
                for(size_t xi = 0; xi < gs; xi++){
                    float x = bbox_x0 + xi*bbox_w*k;
                    size_t xs = lroundf(max(0.0f, min(w - 1.0f, x*w)));
                    //printf("%%f,%%f = %%z,%%z\\n", x, y, xs, ys);
                    for(size_t f=0; f < fn; f++){
                        r[b*rs0 + ff*rs1 + j*rs2 + i*rs3] = fmap[b*fs0 + f*fs1 + ys*fs2 + xs*fs3];
                        ff++;
                    }
                }
            }
            r[b*rs0 + ff*rs1 + j*rs2 + i*rs3] = bbox_h;
            r[b*rs0 + (ff+1)*rs1 + j*rs2 + i*rs3] = bbox_w;
        };
        """%(self.grid_size, self.grid_size)
        
    def c_code(self, node, name, inputs, output, sub):
        fmap, bbox = inputs
        result, = output
        fail = sub['fail']
        grid_size = self.grid_size
        return """
        size_t batch_size = CudaNdarray_HOST_DIMS(%(fmap)s)[0];        
        size_t feature_num = CudaNdarray_HOST_DIMS(%(fmap)s)[1];        
        size_t height = CudaNdarray_HOST_DIMS(%(fmap)s)[2];        
        size_t width = CudaNdarray_HOST_DIMS(%(fmap)s)[3];        
        size_t sample_num = CudaNdarray_HOST_DIMS(%(bbox)s)[1]; 

        //choose grid / block dim
        size_t total_threads = batch_size*sample_num*sample_num;
        size_t threads_per_block = 1024;
        size_t grid_num = std::ceil((double)total_threads / threads_per_block);
        dim3 grid_dim(grid_num, 1, 1);
        dim3 block_dim(threads_per_block, 1, 1); 
        cudaError_t err;
        
        size_t grid_size = %(grid_size)i;
        int dims[] = {batch_size, feature_num*grid_size*grid_size + 2, sample_num, sample_num};
        if (CudaNdarray_prep_output(&%(result)s, 4, dims) != 0){
            PyErr_Format(PyExc_RuntimeError, "Failed to allocate output array");
            %(fail)s
        }

        //printf("result shape =(%%i,%%i,%%i,%%i)", CudaNdarray_HOST_DIMS(%(result)s)[0], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[1], CudaNdarray_HOST_DIMS(%(result)s)[2], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[3]);
        //printf("batch_size %%i, feature_num=%%i, grid_size=%%i, sample_num=%%i\\n", batch_size, feature_num, grid_size, sample_num);
        k_sparse_sample%(grid_size)i<<<grid_dim, block_dim>>>(
            CudaNdarray_DEV_DATA(%(fmap)s), 
            CudaNdarray_HOST_STRIDES(%(fmap)s)[0],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[1],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[2],
            CudaNdarray_HOST_STRIDES(%(fmap)s)[3],
            CudaNdarray_DEV_DATA(%(bbox)s), 
            CudaNdarray_HOST_STRIDES(%(bbox)s)[0],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[1],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[2],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[3],
            CudaNdarray_DEV_DATA(%(result)s), 
            CudaNdarray_HOST_STRIDES(%(result)s)[0],
            CudaNdarray_HOST_STRIDES(%(result)s)[1],
            CudaNdarray_HOST_STRIDES(%(result)s)[2],
            CudaNdarray_HOST_STRIDES(%(result)s)[3],
            feature_num, height, width, sample_num, batch_size);
        
        CNDA_THREAD_SYNC;             
        err = cudaGetLastError();
        if (err != cudaSuccess){
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s:\\n", cudaGetErrorString(err));
            %(fail)s
        }
        """%locals()

    def c_code_cache_version(self):
        return (200,103)

class DeNetSparseGradOp(GpuOp):
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __eq__(self, other):
        return type(self) == type(other) and self.grid_size == other.grid_size

    def __hash__(self):
        return hash((type(self), self.grid_size))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, fmap, bbox, dy):    
        fmap = as_cuda_ndarray_variable(fmap)
        bbox = as_cuda_ndarray_variable(bbox)
        dy = as_cuda_ndarray_variable(dy)

        assert bbox.ndim == 4 and dy.ndim == 4
        return theano.Apply(self, [fmap, bbox, dy], [fmap.type()])

    def c_support_code(self):
        return """
        static __global__ void k_sparse_sample_grad%i(float* df, size_t fs0, size_t fs1, size_t fs2, size_t fs3, 
                                                      float* bbox, size_t bs0, size_t bs1, size_t bs2, size_t bs3,
                                                      float* r, size_t rs0, size_t rs1, size_t rs2, size_t rs3,
                                                      size_t fn, size_t h, size_t w, size_t sn, size_t bs){

            size_t index = blockIdx.x*blockDim.x + threadIdx.x;
            if (index >= (bs*sn*sn))
                return;

            int b,j,i;
            b = index / (sn*sn);
            index -= b*sn*sn;
            j = index / sn;
            index -= j*sn;
            i = index;
            //printf("b=%%i,j=%%i,i=%%i,bs=%%i\\n",b,j,i,bs);
        
            const size_t gs = %i;
            size_t bbox_offset = b*bs0 + j*bs1 + i*bs2;
            float bbox_x0 = bbox[bbox_offset + 0*bs3];
            float bbox_y0 = bbox[bbox_offset + 1*bs3];
            float bbox_x1 = bbox[bbox_offset + 2*bs3];
            float bbox_y1 = bbox[bbox_offset + 3*bs3];
            float bbox_h = bbox_y1 - bbox_y0;
            float bbox_w = bbox_x1 - bbox_x0;
            float k = 1.0f / (gs-1);
            
            size_t ff = 0;
            for(size_t yi = 0; yi < gs; yi++){
                float y = bbox_y0 + yi*bbox_h*k;
                size_t ys = lroundf(max(0.0f, min(h - 1.0f, y*h)));
                for(size_t xi = 0; xi < gs; xi++){
                    float x = bbox_x0 + xi*bbox_w*k;
                    size_t xs = lroundf(max(0.0f, min(w - 1.0f, x*w)));
                    //printf("%%f,%%f = %%z,%%z\\n", x, y, xs, ys);
                    for(size_t f=0; f < fn; f++){
                        atomicAdd(r + b*rs0 + f*rs1 + ys*rs2 + xs*rs3, df[b*fs0 + ff*fs1 + j*fs2 + i*fs3]);
                        ff++;
                    }
                }
            }
        };
        """%(self.grid_size, self.grid_size)
        
    def c_code(self, node, name, inputs, output, sub):
        fmap, bbox, dy = inputs
        result, = output
        fail = sub['fail']
        grid_size = self.grid_size
        return """
        size_t batch_size = CudaNdarray_HOST_DIMS(%(fmap)s)[0];        
        size_t feature_num = CudaNdarray_HOST_DIMS(%(fmap)s)[1];        
        size_t height = CudaNdarray_HOST_DIMS(%(fmap)s)[2];        
        size_t width = CudaNdarray_HOST_DIMS(%(fmap)s)[3];        
        size_t sample_num = CudaNdarray_HOST_DIMS(%(bbox)s)[1]; 

        //choose grid / block dim
        size_t total_threads = batch_size*sample_num*sample_num;
        size_t threads_per_block = 1024;
        size_t grid_num = std::ceil((double)total_threads / threads_per_block);
        dim3 grid_dim(grid_num, 1, 1);
        dim3 block_dim(threads_per_block, 1, 1); 
        cudaError_t err;
        
        size_t grid_size = %(grid_size)i;
        int dims[] = {batch_size, feature_num, height, width};
        if (CudaNdarray_prep_output(&%(result)s, 4, dims) != 0){
            PyErr_Format(PyExc_RuntimeError, "Failed to allocate output array");
            %(fail)s
        }

        if (cudaMemset(%(result)s->devdata, 0, 4*batch_size*feature_num*height*width) != cudaSuccess){
            PyErr_Format(PyExc_RuntimeError, "Failed to zero output array");
            %(fail)s
        }

        //printf("result shape =(%%i,%%i,%%i,%%i)", CudaNdarray_HOST_DIMS(%(result)s)[0], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[1], CudaNdarray_HOST_DIMS(%(result)s)[2], 
        //    CudaNdarray_HOST_DIMS(%(result)s)[3]);
        //printf("batch_size %%i, feature_num=%%i, grid_size=%%i, sample_num=%%i\\n", batch_size, feature_num, grid_size, sample_num);
        k_sparse_sample_grad%(grid_size)i<<<grid_dim, block_dim>>>(
            CudaNdarray_DEV_DATA(%(dy)s), 
            CudaNdarray_HOST_STRIDES(%(dy)s)[0],
            CudaNdarray_HOST_STRIDES(%(dy)s)[1],
            CudaNdarray_HOST_STRIDES(%(dy)s)[2],
            CudaNdarray_HOST_STRIDES(%(dy)s)[3],
            CudaNdarray_DEV_DATA(%(bbox)s), 
            CudaNdarray_HOST_STRIDES(%(bbox)s)[0],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[1],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[2],
            CudaNdarray_HOST_STRIDES(%(bbox)s)[3],
            CudaNdarray_DEV_DATA(%(result)s), 
            CudaNdarray_HOST_STRIDES(%(result)s)[0],
            CudaNdarray_HOST_STRIDES(%(result)s)[1],
            CudaNdarray_HOST_STRIDES(%(result)s)[2],
            CudaNdarray_HOST_STRIDES(%(result)s)[3],
            feature_num, height, width, sample_num, batch_size);
        
        CNDA_THREAD_SYNC;             
        err = cudaGetLastError();
        if (err != cudaSuccess){
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s:\\n", cudaGetErrorString(err));
            %(fail)s
        }
        """%locals()

    def c_code_cache_version(self):
        return (0,4)
    
    

