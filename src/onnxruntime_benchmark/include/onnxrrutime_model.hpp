#pragma once
#include "logger.hpp"
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ONNXModel {
 private:

    int batch_size;
    int num_threads;
    std::string layout;
    std::vector<float> mean;
    std::vector<float> scale;

    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_names;
    std::vector<DataPrecision> input_data_precisions;
    std::vector<ONNXTensorElementDataType> input_data_types;
    std::vector<std::vector<int64_t>> input_shapes;

    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<const char*> output_names;

    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;

    void read_model(const std::string model);
    void get_input_output_info();
 public:
    ONNXModel(const std::string& model_file, int batch_size,const std::string& layout_string,
       const std::string& mean_string, const std::string& scale_string, int num_threads);
    void prepare_input_tensors(const std::vector<std::string>& input_files);
    void infer();
};
