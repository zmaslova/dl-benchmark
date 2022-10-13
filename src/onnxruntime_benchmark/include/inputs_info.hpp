#pragma once
#include "tensors_handler.hpp"
#include <map>
#include <string>
#include <vector>

// struct InputDescr {
//     ONNXTensorDescr tdesc;
//     std::vector<std::string> files;
//     std::vector<float> mean;
//     std::vector<float> scale;
//     size_t batch;
// };

// using InputsInfo = std::map<std::string, InputDescr>;

// InputsInfo get_inputs_info(const std::vector<ONNXTensorDescr>& model_inputs, const std::string& layout_string,
//     const std::string& shape_string, const std::string& mean_string, const std::string& scale_string, int batch_size);

// std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo& inputs_info, int tensors_num = 1);
