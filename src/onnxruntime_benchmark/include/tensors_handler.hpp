#pragma once
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <map>
#include <string>
#include <vector>

struct ONNXTensorDescr {
   std::string name;
   std::vector<int64_t> shape;
   std::string layout;
   DataPrecision precision;
   ONNXTensorElementDataType elem_type;
   const std::vector<float> mean;
   const std::vector<float> scale;
};

Ort::Value get_image_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor, int batch_size);

Ort::Value get_binary_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor, int batch_size);