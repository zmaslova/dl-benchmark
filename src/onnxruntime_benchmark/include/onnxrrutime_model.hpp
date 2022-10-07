#pragma once

#include "logger.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <map>
#include <memory>
#include <string>

class ONNXModel {
 private:
    std::vector<char*> input_names;
    std::vector<char*> output_names;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    ONNXTensorElementDataType data_precision;
    int num_threads = 4;

 public:
    ONNXModel(int num_threads) : num_threads(num_threads) {}; 
    void read_model(const std::string model);
    void prepare_input_tensors(std::vector<cv::Mat> imgs);
    void infer();
};
