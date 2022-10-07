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
    std::vector<char*> input_names;
    std::vector<char*> output_names;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    ONNXTensorElementDataType data_precision;
    int batch_size; 
    int num_threads;

    void read_model(const std::string model);
    void get_input_output_info();

 public:
    ONNXModel(const std::string& model_file, int num_threads, int batch_size); 
    void prepare_input_tensors(const std::vector<std::string>& input_files);
    void infer();
};
