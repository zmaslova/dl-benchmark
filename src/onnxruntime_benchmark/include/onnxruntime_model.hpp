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

struct ONNXTensorDescr {
    std::string name;
    std::vector<int64_t> shape;
    std::string layout;
    ONNXTensorElementDataType elem_type;

    bool is_image() const;
    bool is_dynamic() const;
    bool has_batch() const;
    bool is_dynamic_batch() const;
    int64_t get_dimension_by_layout(char ch) const;
    int64_t channels() const;
    int64_t width() const;
    int64_t height() const;
    void set_batch(int64_t batch_size);
};

class ONNXModel {
private:
    struct IOInfo {
        std::vector<const char *> input_names;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<ONNXTensorElementDataType> input_data_types;
        std::vector<std::vector<int64_t>> input_shapes;

        std::vector<const char *> output_names;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    } io;

    int nthreads;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;

    void fill_inputs_outputs_info();

public:
    ONNXModel(int nthreads);
    void read_model(const std::string &model);
    std::vector<ONNXTensorDescr> get_input_tensors_info() const;
    void run(const std::vector<Ort::Value> &input_tensors) const;
};
