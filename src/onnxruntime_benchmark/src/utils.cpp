#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <exception>
#include <map>
#include <string>

template<class T>
Ort::Value create_tensor_from_image(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

template<class T>
Ort::Value create_tensor_from_binary(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

Ort::Value get_image_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto precision = tensor_descr.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_image<float>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_image<int32_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_image<int64_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_image<bool>(files, tensor_descr, batch_size);
    }
    else {
        throw std::runtime_error("Unsuported tensor precision: " + get_precision_str(precision));
    }
}

Ort::Value get_binary_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto precision = tensor_descr.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_binary<float>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_binary<int32_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_binary<int64_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_binary<bool>(files, tensor_descr, batch_size);
    }
    else {
        throw std::runtime_error("Unsuported tensor precision: " + get_precision_str(precision));
    }
}

DataPrecision get_data_precision(ONNXTensorElementDataType type) {
    if (onnx_dtype_to_precision_map.count(type) > 0) {
        return onnx_dtype_to_precision_map.at(type);
    }
    else {
        throw std::runtime_error("ConvertToDataPrecision: does not support element type " + std::to_string(type));
    }
}

std::string get_precision_str(DataPrecision p) {
    for (auto& [key, val] : precision_to_str_map) {
        if (val == p) {
            return key;
        }
    }
    return "UNKNOWN";
}
