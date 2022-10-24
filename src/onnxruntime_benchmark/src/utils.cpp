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
        throw std::invalid_argument("does not support element type " + std::to_string(type));
    }
}

std::string utils::get_precision_str(DataPrecision p) {
    for (auto &[key, val] : precision_to_str_map) {
        if (val == p) {
            return key;
        }
    }
    return "UNKNOWN";
}

void utils::set_batch_size(InputsInfo &inputs_info, int batch_size) {
    for (auto &[_, input_descr] : inputs_info) {
        input_descr.tensor_descr.set_batch(batch_size);
    }
}

int utils::get_batch_size(const InputsInfo &inputs_info) {
    int batch_size = 0;
    for (auto &[name, info] : inputs_info) {
        auto &tensor_descr = info.tensor_descr;
        std::size_t batch_index = tensor_descr.layout.find("N");
        if (batch_index != std::string::npos) {
            if (batch_size == 0) {
                batch_size = tensor_descr.shape[batch_index];
            }
            else if (batch_size != tensor_descr.shape[batch_index]) {
                throw std::logic_error("Batch size is different for different inputs!");
            }
        }
    }
    if (batch_size == 0) {
        logger::warn << "Batch dimension not found, batch is set to 1" << logger::endl;
        batch_size = 1;
    }
    return batch_size;
}

std::string utils::guess_layout_from_shape(std::vector<int64> &shape) {
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
