#include "onnxruntime_model.hpp"

#include "args_handler.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

bool ONNXTensorDescr::is_image() const {
    return (layout == "NCHW" || layout == "NHWC" || layout == "CHW" || layout == "HWC") && channels() == 3;
}

bool ONNXTensorDescr::is_dynamic() const {
    return std::find(shape.begin(), shape.end(), -1) != shape.end();
}

void ONNXTensorDescr::set_batch(int64_t batch_size) {
    std::size_t batch_index = layout.find("N");
    if (batch_index != std::string::npos) {
        shape[batch_index] = batch_size;
    }
}

int64_t ONNXTensorDescr::get_dimension_by_layout(char ch) const {
    size_t pos = layout.find(ch);
    if (pos == std::string::npos) {
        throw std::invalid_argument("Can't get " + std::string(ch, 1) + " from layout " + layout);
    }
    return shape.at(pos);
}

int64_t ONNXTensorDescr::channels() const {
    return get_dimension_by_layout('C');
}

int64_t ONNXTensorDescr::width() const {
    return get_dimension_by_layout('W');
}

int64_t ONNXTensorDescr::height() const {
    return get_dimension_by_layout('H');
}

ONNXModel::ONNXModel(int nthreads) : nthreads(nthreads) {}

void ONNXModel::read_model(const std::string &model_path) {

    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ORT Bench");
    Ort::SessionOptions session_options;
    // Profile enabling?
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // if (device_type == DLBenchDevice::CPU) { // log device and nthreads
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // Parallel
    if (nthreads > 0) {
        session_options.SetIntraOpNumThreads(nthreads);
    }
    logger::info << "Reading model " << model_path << logger::endl;
    session = std::make_shared<Ort::Session>(*env, model_path.c_str(), session_options);
    logger::info << "Device: CPU" << logger::endl;
    logger::info << "\tNumber of threads: " << (nthreads != 0 ? std::to_string(nthreads) : "DEFAULT") << logger::endl;
    fill_inputs_outputs_info();
}

void ONNXModel::fill_inputs_outputs_info() {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    // Get input from model
    logger::info << "Model inputs:" << logger::endl;
    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        // get input name
        auto input_name = session->GetInputNameAllocated(i, allocator);
        io.input_names.emplace_back(input_name.get());
        io.input_names_ptr.push_back(std::move(input_name));
        // get input type
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        io.input_data_types.push_back(type);

        // get input shapes/dims
        auto input_node_shape = tensor_info.GetShape();
        io.input_shapes.push_back(input_node_shape);

        // log inputs infor
        logger::info << "\t" << io.input_names[i] << ": " << get_precision_str(get_data_precision(type)) << " "
                     << shape_string(tensor_info.GetShape()) << logger::endl;
    }

    // Get outputs from model
    logger::info << "Model outputs:" << logger::endl;
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        // get output name
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        io.output_names.push_back(output_name.get());
        io.output_names_ptr.push_back(std::move(output_name));

        // get output type
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();

        // log outputs info
        logger::info << "\t" << io.output_names[i] << ": " << get_precision_str(get_data_precision(type)) << " "
                     << shape_string(tensor_info.GetShape()) << logger::endl;
    }
}

std::vector<ONNXTensorDescr> ONNXModel::get_input_tensors_info() const {
    std::vector<ONNXTensorDescr> input_tensors_info;
    for (int i = 0; i < io.input_names.size(); ++i) {
        input_tensors_info.push_back({std::string(io.input_names[i]),
                                      io.input_shapes[i],
                                      "",
                                      io.input_data_types[i]});
    }
    return input_tensors_info;
}

void check_output(const std::vector<Ort::Value> &output_tensors, int batch_size) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        logger::debug << "Output tensor #" << i << logger::endl;
        const float *floatarr = output_tensors[i].GetTensorData<float>();
        for (int b = 0; b < batch_size; ++b) {
            logger::debug << "Batch #" << b << logger::endl;
            std::vector<float> res(floatarr + b * 1000, floatarr + b * 1000 + 1000);
            std::vector<int> idx(1000);
            std::iota(idx.begin(), idx.end(), 0);
            auto max = std::max_element(res.begin(), res.end());
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&res](int l, int r) {
                return res[l] > res[r];
            });
            for (size_t j = 0; j < 5; ++j) {
                logger::debug << "id: " << idx[j] << " score " << res[idx[j]] << logger::endl;
            }
        }
    }
}

void ONNXModel::run(const std::vector<Ort::Value> &input_tensors) const {
    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       io.input_names.data(),
                                       input_tensors.data(),
                                       io.input_names.size(),
                                       io.output_names.data(),
                                       io.output_names.size());
    check_output(output_tensors, 3);
}
