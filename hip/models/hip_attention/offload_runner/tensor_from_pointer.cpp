#include <ATen/ATen.h>
#include <torch/extension.h>

// write function that will be called from python
torch::Tensor tensor_from_pointer(
    const long ptr, 
    const long size, 
    const long elem_size,
    const long device_index
) {
    printf("%ld\n", ptr);
    printf("%ld\n", size);
    printf("%ld\n", elem_size);

    auto dtype = torch::kFloat32;
    if (elem_size == 16) {
        dtype = torch::kFloat16;
    }
    else if (elem_size == 32) {
        dtype = torch::kFloat32;
    }
    else if (elem_size == 64) {
        dtype = torch::kFloat64;
    }
    else if (elem_size == 8) {
        dtype = torch::kByte;
    }
    else {
        dtype = torch::kFloat32;
    }

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_index);
    torch::Tensor tharray = torch::from_blob((void *) ptr, {size, }, options);

    return tharray;
}

// bind function by PYBIND11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_from_pointer", &tensor_from_pointer, "tensor_from_pointer (CUDA)");
}