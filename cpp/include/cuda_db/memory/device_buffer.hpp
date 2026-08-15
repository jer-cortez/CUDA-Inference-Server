#pragma once
#include <vector>
#include <cstddef>


namespace cuda_db { 

struct DeviceBuffer { 
    void* ptr;
    size_t size;
    size_t capacity;
    uint32_t flag;
    bool is_mapped = false;
    
};

}