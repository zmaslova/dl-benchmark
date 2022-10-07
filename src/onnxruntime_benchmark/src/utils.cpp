#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <exception>
#include <map>
#include <string>

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