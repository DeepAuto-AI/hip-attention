#include <ATen/ATen.h>
#include <torch/extension.h>

// write function that will be called from python
torch::Tensor tensor_from_pointer(
    const long ptr,
    const long size,
    const long elem_size,
    const long device_index
) {
    // printf("%ld\n", ptr);
    // printf("%ld\n", size);
    // printf("%ld\n", elem_size);

    auto dtype = torch::kUInt32;
    if (elem_size == 16) {
        dtype = torch::kUInt16;
    }
    else if (elem_size == 32) {
        dtype = torch::kUInt32;
    }
    else if (elem_size == 64) {
        dtype = torch::kUInt64;
    }
    else if (elem_size == 8) {
        dtype = torch::kByte;
    }
    else {
        dtype = torch::kUInt32;
    }

    auto options = torch::TensorOptions();
    if (device_index >= 0) {
        options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_index);
    } else {
        options = torch::TensorOptions().dtype(dtype);
    }
    torch::Tensor tharray = torch::from_blob((void *) ptr, {size, }, options);

    return tharray;
}

TORCH_LIBRARY(hip_attn, m) {
    m.def("tensor_from_pointer", &tensor_from_pointer);
}
