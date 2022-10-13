#pragma once
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ONNXTensorDescr {
    std::string name;
    std::vector<int64_t> shape;
    std::string layout;
    DataPrecision precision;
    ONNXTensorElementDataType elem_type;

    bool is_image() const;
    bool is_dynamic() const;
    int64_t get_dimension_by_layout(char ch) const;
    int64_t channels() const;
    int64_t width() const;
    int64_t height() const;
    void set_batch(int64_t batch_size);
};

struct InputDescr {
    ONNXTensorDescr tdesc;
    std::vector<std::string> files;
    std::vector<float> mean;
    std::vector<float> scale;
};

using InputsInfo = std::map<std::string, InputDescr>;

size_t get_batch_size(const InputsInfo& inputs_info);

Ort::Value get_binary_tensor(const InputDescr& input_descr, int batch_size, int start_index);

Ort::Value get_image_tensor(const InputDescr& input_descr, int batch_size, int start_index);

Ort::Value get_random_tensor(const InputDescr& input_descr, int batch_size);

std::string guess_layout_from_shape(std::vector<int64_t>& shape);

InputsInfo get_inputs_info(const std::map<std::string, std::vector<std::string>>& input_files, const std::vector<ONNXTensorDescr>& model_inputs, const std::string& layout_string,
    const std::string& shape_string, const std::string& mean_string, const std::string& scale_string);

std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo& inputs_info, int batch_size, int tensors_num = 1);

