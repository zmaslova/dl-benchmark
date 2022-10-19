#pragma once
#include "logger.hpp"
#include "utils.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using HighresClock = std::chrono::high_resolution_clock;

struct ONNXTensorDescr {
    std::string name;
    std::vector<int64_t> shape;
    std::string layout;
    ONNXTensorElementDataType type;

    bool is_image() const;
    bool is_dynamic() const;
    bool has_batch() const;
    bool is_dynamic_batch() const;
    int64_t get_dimension_by_layout(char ch) const;
    int64_t channels() const;
    int64_t width() const;
    int64_t height() const;
    void set_batch(int batch_size);
};

using IOTensorsInfo = std::pair<std::vector<ONNXTensorDescr>, std::vector<ONNXTensorDescr>>;

class ONNXModel {
private:
    struct IOInfo {
        std::vector<const char *> input_names;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<ONNXTensorElementDataType> input_data_types;
        std::vector<std::vector<int64_t>> input_shapes;

        std::vector<const char *> output_names;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
        std::vector<ONNXTensorElementDataType> output_data_types;
        std::vector<std::vector<int64_t>> output_shapes;
    } io;

    int nthreads;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session;

    // general time measurments;
    HighresClock::time_point total_start_time;
    HighresClock::time_point total_end_time;

    // time measuremnt for one inference
    HighresClock::time_point infer_start_time;
    std::vector<double> latencies;

public:
    ONNXModel(int nthreads);
    void fill_inputs_outputs_info();
    void reset_timers();
    std::vector<double> get_latencies();
    double get_total_time_ms();
    void read_model(const std::string &model);
    IOTensorsInfo get_io_tensors_info() const;
    void run(const std::vector<Ort::Value> &input_tensors);
};
