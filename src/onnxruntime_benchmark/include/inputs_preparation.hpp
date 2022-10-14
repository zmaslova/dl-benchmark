#pragma once
#include "utils.hpp"
#include "onnxruntime_model.hpp"

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct InputDescr {
    ONNXTensorDescr tensor_descr;
    std::vector<std::string> files;
    std::vector<float> mean;
    std::vector<float> scale;
};

using InputsInfo = std::map<std::string, InputDescr>;

InputsInfo get_inputs_info(const std::map<std::string, std::vector<std::string>> &input_files,
                           const std::vector<ONNXTensorDescr> &model_inputs,
                           const std::string &layout_string,
                           const std::string &shape_string,
                           const std::string &mean_string,
                           const std::string &scale_string);

std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo &inputs_info,
                                                       int batch_size,
                                                       int tensors_num = 1);
