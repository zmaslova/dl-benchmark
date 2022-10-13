#pragma once
#include "logger.hpp"
#include "tensors_handler.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ONNXModel {
 private:
    struct IOInfo {
        std::vector<const char*> input_names;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<DataPrecision> input_data_precisions;
        std::vector<ONNXTensorElementDataType> input_data_types;
        std::vector<std::vector<int64_t>> input_shapes;

        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    } io;

    int nthreads;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;

    void fill_inputs_outputs_info();

 public:
    ONNXModel(int nthreads);
    void read_model(const std::string& model);
    std::vector<ONNXTensorDescr> get_input_tensors_info() const;
    void run(const std::vector<Ort::Value>& input_tensors) const;
};
