#include "utils.hpp"

#include "inputs_preparation.hpp"
#include "onnxruntime_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

utils::DataPrecision utils::get_data_precision(ONNXTensorElementDataType type) {
    if (onnx_dtype_to_precision_map.count(type) > 0) {
        return onnx_dtype_to_precision_map.at(type);
    }
    else {
        throw std::invalid_argument("Does not support element data type " + std::to_string(type));
    }
}

std::string utils::get_precision_str(DataPrecision p) {
    if (precision_to_str_map.count(p) > 0) {
        return precision_to_str_map.at(p);
    }
    return "UNKNOWN";
}

std::string utils::guess_layout_from_shape(const std::vector<int64> &shape) {
    if (shape.size() == 2) {
        return "NC";
    }
    if (shape.size() == 3) {
        return shape[0] > 4 && shape[2] <= 4 ? "HWC" : "CHW";
    }
    if (shape.size() == 4) {
        return shape[1] > 4 && shape[3] <= 4 ? "NHWC" : "NCHW";
    }
    throw std::invalid_argument("Unsupported shape with size " + std::to_string(shape.size()));
}

std::string utils::format_double(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
};
