#include <Python.h>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <list>
#include <algorithm>
#include <thread>
#include <cstdio>
#include <chrono>
#include <atomic>

#include "numpy/arrayobject.h"
#include "theano_mod_helper.h"


//basic logging 
FILE* log_file = NULL;
static PyObject* init_logging(PyObject *self, PyObject *args){
    char* fname; //ndarray
    if (!PyArg_ParseTuple(args, "s", &fname))
        return NULL;

    log_file = fopen(fname, "w");
    setbuf(log_file, NULL);

    Py_RETURN_NONE;
}

#define LOG_PRINT(...) fprintf(log_file, ##__VA_ARGS__);fflush(log_file)

//basic timer
class Timer {
public:
    Timer(bool start = true){ if (start) this->reset();}
    void mark() {this->times.push_back(std::clock());}
    void reset(){this->times.resize(0); this->times.push_back(std::clock());}
    double current_ms() const {return 1000*(double)(std::clock() - this->times[0]) / CLOCKS_PER_SEC;}
    std::vector<double> delta_ms() const {
        std::vector<double> dt;
        for(size_t i=0; i < (this->times.size()-1); i++){
            dt.push_back(1000 * (double)(this->times[i+1] - this->times[i]) / CLOCKS_PER_SEC);
        }
        return dt;
    };

    std::vector<double> times;
};

typedef std::tuple<npy_intp, npy_intp, float> CornerType;
struct SampleType {
    SampleType(float pr=0, float x0=0, float y0=0, float x1=0, float y1=0){
        v[0] = pr; v[1] = x0; v[2] = y0; v[3] = x1; v[4] = y1;
    }

    const float& pr() const {return this->v[0];}
    float& pr() {return this->v[0];}

    const float& x0() const {return this->v[1];}
    float& x0() {return this->v[1];}

    const float& y0() const {return this->v[2];}
    float& y0() {return this->v[2];}

    const float& x1() const {return this->v[3];}
    float& x1() {return this->v[3];}

    const float& y1() const {return this->v[4];}
    float& y1() {return this->v[4];}

    float width() const {return this->v[3]-this->v[1];}
    float height() const {return this->v[4]-this->v[2];}
    float area() const {return this->width()*this->height();}
    float v[5];

    bool operator<(const SampleType& rhs) const {return this->pr() > rhs.pr();}

    void update_bounds(const SampleType& s) {
        this->pr() = std::max(s.pr(), this->pr());
        this->x0() = std::min(s.x0(), this->x0());
        this->y0() = std::min(s.y0(), this->y0());
        this->x1() = std::max(s.x1(), this->x1());
        this->y1() = std::max(s.y1(), this->y1());
    }

    static float overlap(const SampleType& s0, const SampleType& s1){

        float dx = std::max(0.0f, std::min(s0.x1(), s1.x1()) - std::max(s0.x0(), s1.x0()));
        float dy = std::max(0.0f, std::min(s0.y1(), s1.y1()) - std::max(s0.y0(), s1.y0()));
        return dx*dy;
    }

    static float overlap_rel(const SampleType& s0, const SampleType& s1){
        return SampleType::overlap(s0,s1) / s0.area();
    }
    
    static float overlap_iou(const SampleType& s0, const SampleType& s1){
        float ai = SampleType::overlap(s0, s1);
        float au = s0.area() + s1.area() - ai;
        return ai / au;
    }
};

class ClusterType {
public:
    ClusterType(){};
    ClusterType(const SampleType& s){
        this->bbox = s;
        this->sv_list.push_back(new std::vector<SampleType>(1, s));
    }

    ~ClusterType(){
        for(std::vector<SampleType> *sv : this->sv_list)
            delete sv;
    }

    void merge(ClusterType& c){
        this->bbox.update_bounds(c.bbox);
        for(std::vector<SampleType> *sv : c.sv_list)
            this->sv_list.push_back(sv);
        c.sv_list.resize(0);
    }

    void add_sample(const SampleType& s){
        this->sv_list[0]->push_back(s);
        this->bbox.update_bounds(s);
    }

    //check if sample overlaps this cluster
    bool overlap(const SampleType& sample_i, const float& threshold) const {

        if (SampleType::overlap(sample_i, this->bbox) == 0)
            return false;

        for(const std::vector<SampleType> *sv : this->sv_list){
            for(const SampleType& sample_j : *sv){
                if (SampleType::overlap_iou(sample_i, sample_j) > threshold)
                    return true;
            }
        }
        return false;
    }

    //get all samples associated with cluster
    std::vector<SampleType> get_samples() const {
        std::vector<SampleType> samples;
        for(std::vector<SampleType> *sv : this->sv_list)
            samples.insert(samples.end(), sv->begin(), sv->end());
        return samples;
    }

    size_t get_sample_num() const {
        size_t num = 0;
        for(std::vector<SampleType> *sv : this->sv_list)
            num += sv->size();
        return num;
    }

    SampleType bbox;
    std::vector< std::vector<SampleType>* > sv_list;
};

void apply_cluster(std::vector<SampleType>& samples, float threshold, size_t input_num, size_t output_num){

    typedef std::list<ClusterType*>::iterator ClusterListIt;

    Timer timer;

    //restrict number of samples input into clustering
    if (samples.size() > input_num){
        std::partial_sort(samples.begin(), samples.begin() + input_num, samples.end());
        samples.resize(input_num);
    }
    timer.mark();

    std::list<ClusterType*> cluster_list;
    for(const SampleType& sample_i : samples){
        
        //find overlapping clusters
        std::vector<ClusterListIt> overlap;
        for(ClusterListIt it=cluster_list.begin(); it != cluster_list.end(); it++)
            if ((*it)->overlap(sample_i, threshold))
                overlap.push_back(it);
        
        if (overlap.size() > 0){

            //merge cluster_list and add sample
            ClusterType* cluster = *overlap.back();
            overlap.pop_back();

            //add new sample
            cluster->add_sample(sample_i);

            //merge other clusters into cluster and erase
            for(const ClusterListIt& it : overlap){
                ClusterType* cluster_it = *it;
                cluster->merge(*cluster_it);
                cluster_list.erase(it);
                delete cluster_it;
            }

        } else {
            //create new cluster
            cluster_list.push_back(new ClusterType(sample_i));
        }
    }

    //limit number of clusters by one with smallest number of elements
    if (cluster_list.size() > output_num){
        cluster_list.sort([](const ClusterType* lhs, const ClusterType* rhs){return (lhs->get_sample_num() > rhs->get_sample_num());});

        size_t n=0;
        for(ClusterListIt it = cluster_list.begin(); it != cluster_list.end(); it++){
            if (n > output_num)
                delete *it;
            n++;
        }
        cluster_list.resize(output_num);
    }
    timer.mark();

    //rebuild samples
    double cluster_ratio = (double)(output_num - cluster_list.size()) / (samples.size() - cluster_list.size());
    size_t sample_num = samples.size();
    samples.resize(0);
    for(ClusterListIt it=cluster_list.begin(); it != cluster_list.end(); it++){
        std::vector<SampleType> it_samples = (*it)->get_samples();
        size_t n = 1 + std::floor(it_samples.size()*cluster_ratio);
        std::partial_sort(it_samples.begin(), it_samples.begin() + n, it_samples.end());
        samples.insert(samples.end(), it_samples.begin(), it_samples.begin() + n);
        delete *it;
    }
    timer.mark();

    LOG_PRINT("Found %zu clusters from %zu samples - outputted %zu samples (cluster ratio=%.1f%%)\n", cluster_list.size(), sample_num, 
              samples.size(), cluster_ratio*100);

    std::vector<double> delta_ms = timer.delta_ms();
    LOG_PRINT("Timing (ms) - input %.0f, cluster %.0f, build %.0f\n", delta_ms[0], delta_ms[1], delta_ms[2]);
}


//unoptimized NMS implementation
void apply_nms(std::vector<SampleType>& samples, float threshold, size_t max_size){

    //limit to maximum size
    if (samples.size() > max_size){
        std::partial_sort(samples.begin(), samples.begin() + max_size, samples.end());
        samples.resize(max_size);
    }

    std::vector<SampleType> samples_nms;
    for(const SampleType& sample_i : samples){

        bool supress = false;
        for(const SampleType& sample_j : samples){
            if ((sample_i.pr() < sample_j.pr()) && (SampleType::overlap_iou(sample_i, sample_j) > threshold)){
                supress = true;
                break;
            }
        }        

        if (!supress)
            samples_nms.push_back(sample_i);
    }
    samples = samples_nms;
}

void get_sample(PyArrayObject* corner_pr, npy_intp b, npy_intp x0, npy_intp y0, npy_intp x1, npy_intp y1, std::vector<SampleType>& samples){
    
    npy_intp height = PyArray_DIM(corner_pr, 3);
    npy_intp width = PyArray_DIM(corner_pr, 4);

    npy_intp f00[] = {b, 0, 0, y0, x0};
    npy_intp f01[] = {b, 0, 1, y0, x1};
    npy_intp f10[] = {b, 0, 2, y1, x0};
    npy_intp f11[] = {b, 0, 3, y1, x1};
    float pr_f = 0;
    pr_f += *(float*)PyArray_GetPtr(corner_pr, f00);
    pr_f += *(float*)PyArray_GetPtr(corner_pr, f01);
    pr_f += *(float*)PyArray_GetPtr(corner_pr, f10);
    pr_f += *(float*)PyArray_GetPtr(corner_pr, f11);
        
    npy_intp t00[] = {b, 1, 0, y0, x0};
    npy_intp t01[] = {b, 1, 1, y0, x1};
    npy_intp t10[] = {b, 1, 2, y1, x0};
    npy_intp t11[] = {b, 1, 3, y1, x1};
    float pr_t = 0;
    pr_t += *(float*)PyArray_GetPtr(corner_pr, t00);
    pr_t += *(float*)PyArray_GetPtr(corner_pr, t01);
    pr_t += *(float*)PyArray_GetPtr(corner_pr, t10);
    pr_t += *(float*)PyArray_GetPtr(corner_pr, t11);

    //handle center if provided
    if (PyArray_DIM(corner_pr, 2) == 5){
        npy_intp cx = (x0+x1)/2;
        npy_intp cy = (y0+y1)/2;
        npy_intp f[] = {b, 0, 4, cy, cx};
        npy_intp t[] = {b, 1, 4, cy, cx};
        pr_f += *(float*)PyArray_GetPtr(corner_pr, f);
        pr_t += *(float*)PyArray_GetPtr(corner_pr, t);
    }

    float pr = 1.0 / (1.0 + std::exp(fabs(pr_f - pr_t)));
    samples.push_back(SampleType(pr, (double)x0 / width, (double)y0 / height, (double)(x1+1) / width, (double)(y1+1) / height));
};

//simple bounding box hash (requires width / height < 65535)
inline uint64_t get_bbox_hash(npy_intp x0, npy_intp y0, npy_intp x1, npy_intp y1){
    uint64_t hash = 0;
    hash |= ((uint64_t)x0 << 48);
    hash |= ((uint64_t)y0 << 32);
    hash |= ((uint64_t)x1 << 16);
    hash |= ((uint64_t)y1 << 0);
    return hash;
}

//combine corner combinations to produce samples
std::vector<SampleType> search_corners(PyArrayObject* corner_pr, npy_intp b, const std::vector<std::vector<CornerType>>& corner_list){

    //print corners
    // LOG_PRINT("Found corners: ");
    // for(int ci=0; ci < corner_list.size(); ci++)
    //     LOG_PRINT("%zu ", corner_list[ci].size());
    // LOG_PRINT("\n");

    std::vector<SampleType> samples;
    std::unordered_map<uint64_t, bool> unique_bboxs;
    npy_intp height = PyArray_DIM(corner_pr, 3);
    npy_intp width = PyArray_DIM(corner_pr, 4);

    //search top left + bottom right
    const std::vector<CornerType>& top_left = corner_list[0];
    const std::vector<CornerType>& bottom_right = corner_list[3];
    for(const CornerType& tl : top_left){
        npy_intp x0 = std::get<0>(tl);
        npy_intp y0 = std::get<1>(tl);
        for(const CornerType& br : bottom_right){
            npy_intp x1 = std::get<0>(br);
            npy_intp y1 = std::get<1>(br);
            if ((x1 <= x0) || (y1 <= y0))
                continue;

            uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
            if (unique_bboxs.count(hash) == 0){
                get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                unique_bboxs[hash] = true;
            }
        }
    }
    // LOG_PRINT("%i - Found samples: %zu\n", b, samples.size());

    //search top right + bottom left
    const std::vector<CornerType>& top_right = corner_list[1];
    const std::vector<CornerType>& bottom_left = corner_list[2];
    for(const CornerType& tr : top_right){
        npy_intp x1 = std::get<0>(tr);
        npy_intp y0 = std::get<1>(tr);
        for(const CornerType& bl : bottom_left){
            npy_intp x0 = std::get<0>(bl);
            npy_intp y1 = std::get<1>(bl);
            if ((x1 <= x0) || (y1 <= y0))
                continue;
            
            uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
            if (unique_bboxs.count(hash) == 0){
                get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                unique_bboxs[hash] = true;
            }
        }
    }
    // LOG_PRINT("%i - Found samples: %zu\n", b, samples.size());
    
    //search centers if provided
    if (corner_list.size() == 5){
        const std::vector<CornerType>& center = corner_list[4];

        size_t ci=0;
        for(const CornerType& c : center){

            int cx = (int)std::get<0>(c);
            int cy = (int)std::get<1>(c);

            LOG_PRINT("%li - Process Center (%zu): %i,%i\n", b, ci, cx, cy);
            ci += 1;

            //center + top left
            for(const CornerType& tl : top_left){
                int x0 = (int)std::get<0>(tl);
                int y0 = (int)std::get<1>(tl);
                int x1 = x0 + 2*(cx-x0);
                int y1 = y0 + 2*(cy-y0);

                if ((x0 < 0) || (y0 < 0) || (x1 >= width) || (y1 >= height) || (x1 <= x0) || (y1 <= y0))
                    continue;

                LOG_PRINT("%li: C (%i,%i) + TL (%i,%i) = (%i,%i,%i,%i)\n", b, cx,cy,x0,y0,x0, y0, x1, y1);

                uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
                if (unique_bboxs.count(hash) == 0){
                    get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                    unique_bboxs[hash] = true;
                }
            }


            //center + top right
            for(const CornerType& tr : top_right){
                int x1 = (int)std::get<0>(tr);
                int y0 = (int)std::get<1>(tr);
                int x0 = x1 - 2*(x1-cx);
                int y1 = y0 + 2*(cy-y0);

                if ((x0 < 0) || (y0 < 0) || (x1 >= width) || (y1 >= height) || (x1 <= x0) || (y1 <= y0))
                    continue;

                LOG_PRINT("%li: C (%i,%i) + TR (%i,%i) = (%i,%i,%i,%i)\n", b, cx,cy, x1, y0, x0, y0, x1, y1);

                uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
                if (unique_bboxs.count(hash) == 0){
                    get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                    unique_bboxs[hash] = true;
                }
            }

            //center + bottom left
            for(const CornerType& bl : bottom_left){
                int x0 = (int)std::get<0>(bl);
                int y1 = (int)std::get<1>(bl);
                int x1 = x0 + 2*(cx-x0);
                int y0 = y1 - 2*(y1-cy);

                if ((x0 < 0) || (y0 < 0) || (x1 >= width) || (y1 >= height) || (x1 <= x0) || (y1 <= y0))
                    continue;

                LOG_PRINT("%i: C (%i,%i) + BL (%i,%i) = (%i,%i,%i,%i)\n", b, cx,cy, x0, y1, x0, y0, x1, y1);

                uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
                if (unique_bboxs.count(hash) == 0){
                    get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                    unique_bboxs[hash] = true;
                }
            }

            //center + bottom right
            for(const CornerType& br : bottom_right){
                int x1 = (int)std::get<0>(br);
                int y1 = (int)std::get<1>(br);
                int x0 = x1 - 2*(x1-cx);
                int y0 = y1 - 2*(y1-cy);

                if ((x0 < 0) || (y0 < 0) || (x1 >= width) || (y1 >= height) || (x1 <= x0) || (y1 <= y0))
                    continue;

                LOG_PRINT("%li: C (%i,%i) + BR (%i,%i) = (%i,%i,%i,%i)\n", b, cx,cy, x1, y1, x0, y0, x1, y1);

                uint64_t hash = get_bbox_hash(x0,y0,x1,y1);
                if (unique_bboxs.count(hash) == 0){
                    get_sample(corner_pr, b, x0, y0, x1, y1, samples);
                    unique_bboxs[hash] = true;
                }
            }
        }

        LOG_PRINT("%li - Found samples: %zu\n", b, samples.size());
    }

    return samples;
}

//get maximum pr in local region
inline float get_local_max(PyArrayObject* corner_pr, npy_intp b, npy_intp ci, npy_intp y, npy_intp x, int local_max){
    npy_intp x0 = std::max<npy_intp>(0, x - local_max);
    npy_intp y0 = std::max<npy_intp>(0, y - local_max);
    npy_intp x1 = std::min<npy_intp>(PyArray_DIM(corner_pr,4)-1, x + local_max);
    npy_intp y1 = std::min<npy_intp>(PyArray_DIM(corner_pr,3)-1, y + local_max);
    float max_pr = -100000;
    for(npy_intp yy = y0; yy < y1; yy++){
        for(npy_intp xx = x0; xx < x1; xx++){
            npy_intp index[] = {b, 1, ci, yy, xx};
            max_pr = std::max<float>(max_pr, *(float*)PyArray_GetPtr(corner_pr, index));
        }
    }
    return max_pr;
}

void run_build_samples(PyArrayObject* corner_pr, float corner_threshold, int sample_num, int max_corners, int local_max, 
                       float cluster_threshold, npy_intp batch_index, std::vector<SampleType>* output_samples, 
                       std::atomic<bool>* check_done){

    size_t sample_count = sample_num*sample_num;
    npy_intp batch_size = PyArray_DIM(corner_pr, 0);
    npy_intp corner_num = PyArray_DIM(corner_pr, 2);
    npy_intp height = PyArray_DIM(corner_pr, 3);
    npy_intp width = PyArray_DIM(corner_pr, 4);

    size_t cluster_snum = 10*sample_count;
    size_t nms_max_size = 4*sample_count;

    //find corners for sample
    std::clock_t time = std::clock();
    std::vector<std::vector<CornerType> > corner_list(corner_num, std::vector<CornerType>());

    float threshold = std::log(corner_threshold);
    for(int ci=0; ci < corner_num; ci++){
        
        //find corners
        for(npy_intp y=0; y < height; y++){
            for(npy_intp x=0; x < width; x++){
                npy_intp index[] = {batch_index, 1, ci, y, x};
                float log_pr = *(float*)PyArray_GetPtr(corner_pr, index);
                if (log_pr > threshold){

                    //check local max if required
                    if ((local_max > 0) && (log_pr < get_local_max(corner_pr, batch_index, ci, y, x, local_max)))
                        continue;

                    corner_list[ci].push_back(std::make_tuple(x,y,log_pr));
                }
            }
        }

        //select maximum corners
        if (corner_list[ci].size() > max_corners){
            std::partial_sort(corner_list[ci].begin(), corner_list[ci].begin() + max_corners, corner_list[ci].end(),
                              [](const CornerType& lhs, const CornerType& rhs){return std::get<2>(lhs) > std::get<2>(rhs);});
            corner_list[ci].resize(max_corners);
        }
    }
    double corner_time = (double)(std::clock() - time) / CLOCKS_PER_SEC;

    //find samples in identified corners
    time = std::clock();
    std::vector<SampleType> samples = search_corners(corner_pr, batch_index, corner_list);
    double search_time = (double)(std::clock() - time) / CLOCKS_PER_SEC;

    //perform clustering if neccessary
    time = std::clock();
    if ((samples.size() > sample_count) && (cluster_threshold < 1.0))
        apply_cluster(samples, cluster_threshold, cluster_snum, sample_count);
    double cluster_time = (double)(std::clock() - time) / CLOCKS_PER_SEC;

    //get max
    time = std::clock();
    std::partial_sort(samples.begin(), samples.begin() + std::min<size_t>(samples.size(), sample_count), samples.end());
    if (samples.size() > sample_count)
        samples.resize(sample_count);
    
    double sort_time = (double)(std::clock() - time) / CLOCKS_PER_SEC;
    for(const SampleType& s : samples)
        output_samples->push_back(s);
    
    *check_done = true;
    // LOG_PRINT("%i Timing (ms) - corner: %.0f, search: %.0f, sort: %.0f, cluster: %.0f\n", batch_index, 1000*corner_time, 1000*search_time, 1000*sort_time, 1000*cluster_time);
}

static PyObject* build_samples(PyObject *self, PyObject *args) {

    //get args    
    std::clock_t time = std::clock();
    int thread_num; //
    PyObject* corner_pr_obj; //ndarray
    float corner_threshold; //dict of list of tuple
    int sample_num; //
    int max_corners; //
    int local_max; //
    float cluster_threshold;
    if (!PyArg_ParseTuple(args, "iOfiiif", &thread_num, &corner_pr_obj, &corner_threshold, &sample_num, &max_corners, &local_max, &cluster_threshold))
        return NULL;
    
    PyArrayObject* corner_pr = (PyArrayObject*)PyArray_FROM_OTF(corner_pr_obj, NPY_FLOAT, NPY_IN_ARRAY);
    if (corner_pr==NULL)
        return NULL;

    npy_intp batch_size = PyArray_DIM(corner_pr, 0);
    PyObject* py_sample_list = PyList_New(batch_size);
    if (thread_num == 1){

        std::atomic<bool> check_done;
        std::vector<SampleType> samples;
        samples.reserve(sample_num*sample_num);
        for(npy_intp batch_index=0; batch_index < batch_size; batch_index++){
            run_build_samples(corner_pr, corner_threshold, sample_num, max_corners, local_max, cluster_threshold, batch_index, &samples, &check_done);

            PyObject* py_samples = PyList_New(samples.size());
            for(size_t i=0; i < samples.size(); i++) {
                PyObject* py_sample = Py_BuildValue("(f(ffff))", samples[i].v[0], samples[i].v[1], samples[i].v[2], samples[i].v[3], samples[i].v[4]);
                PyList_SET_ITEM(py_samples, i, py_sample);
            }
            PyList_SET_ITEM(py_sample_list, batch_index, py_samples);
            samples.clear();
        }

    } else {

        typedef std::tuple<std::thread*, int, int, std::atomic<bool>*> WorkerType;

        std::vector<std::vector<SampleType> > output_samples(thread_num, std::vector<SampleType>());
        for(std::vector<SampleType>& samples: output_samples)
            samples.reserve(sample_num*sample_num);

        std::list<int> thread_indexs;
        for(int i=0; i < thread_num; i++)
            thread_indexs.push_back(i);
        
        std::list<WorkerType> workers;
        npy_intp index = 0;
        while((index < batch_size) || workers.size() > 0){

            //spawn new threads:
            while((thread_indexs.size() > 0) && (index < batch_size)){

                int thread_index = thread_indexs.front();
                int batch_index = index;
                std::atomic<bool> *check_done = new std::atomic<bool>(false);
                std::thread *build_thread = new std::thread(run_build_samples, corner_pr,  corner_threshold, sample_num, max_corners, local_max,
                                                            cluster_threshold, batch_index, &output_samples[thread_index], check_done);

                workers.push_back(WorkerType(build_thread, thread_index, batch_index, check_done));
                thread_indexs.pop_front();
                index++;
            }
        
            //clean up finished threads
            std::list<WorkerType>::iterator it = workers.begin();
            while(it != workers.end()){

                std::thread *build_thread = std::get<0>(*it);
                int thread_index = std::get<1>(*it);
                int batch_index = std::get<2>(*it);
                std::atomic<bool> *check_done = std::get<3>(*it);

                if (*check_done){

                     //wait for thread to finish
                    build_thread->join();
                    
                    //insert samples into return value
                    std::vector<SampleType>& samples = output_samples[thread_index];
                    PyObject* py_samples = PyList_New(samples.size());
                    for(size_t i=0; i < samples.size(); i++) {
                        PyObject* py_sample = Py_BuildValue("(f(ffff))", samples[i].v[0], samples[i].v[1], samples[i].v[2], samples[i].v[3], samples[i].v[4]);
                        PyList_SET_ITEM(py_samples, i, py_sample);
                    }
                    PyList_SET_ITEM(py_sample_list, batch_index, py_samples);

                    //clean up
                    samples.clear();
                    thread_indexs.push_back(thread_index);
                    delete build_thread;
                    delete check_done;
                    it = workers.erase(it);

                } else {
                    it++;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    Py_DECREF(corner_pr);
    double build_time = (double)(std::clock() - time) / CLOCKS_PER_SEC;
    LOG_PRINT("Build Samples took %.0f ms\n", 1000*build_time);
    return py_sample_list;
};

static PyObject* build_bbox_array(PyObject *self, PyObject *args) {
    
    PyObject* bbox_obj; //ndarray
    PyObject* bbox_samples; //
    if (!PyArg_ParseTuple(args, "OO", &bbox_samples, &bbox_obj))
        return NULL;

    PyArrayObject* bbox = (PyArrayObject*)PyArray_FROM_OTF(bbox_obj, NPY_FLOAT, NPY_INOUT_ARRAY);
    if (bbox == NULL)
        return NULL;
    
    npy_intp batch_size = PyArray_DIM(bbox, 0);
    npy_intp sample_num = PyArray_DIM(bbox, 1);

    for(npy_intp b=0; b < batch_size; b++){
        PyObject* samples = PyList_GET_ITEM(bbox_samples, b);
        for(npy_intp i=0; i < PyList_GET_SIZE(samples); i++){

            PyObject* s = PyList_GET_ITEM(samples, i);
            PyObject* s_bbox = PyTuple_GET_ITEM(s, 1);
            float *v = (float*)PyArray_GETPTR4(bbox, b, i/sample_num, i%sample_num, 0);
            for(int n=0; n < 4; n++){
                v[n] = (float)PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(s_bbox, n));
            }
        }
    }

    Py_DECREF(bbox);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
        {"init_logging", init_logging, METH_VARARGS, ""},
        {"build_samples", build_samples, METH_VARARGS, ""},
        {"build_bbox_array", build_bbox_array, METH_VARARGS, ""},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "denet_sparse",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC PyInit_denet_sparse(void){
    import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
};
