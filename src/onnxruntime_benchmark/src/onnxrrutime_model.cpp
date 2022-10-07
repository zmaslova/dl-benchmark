#include "onnxrrutime_model.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>


void ONNXModel::read_model(const std::string model_path) {
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ORT Bench");
    Ort::SessionOptions session_options;
    // Profile enabling?
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // if (device_type == DLBenchDevice::CPU) {
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (num_threads > 0) {
        session_options.SetIntraOpNumThreads(num_threads);
    }
    session =  std::make_shared<Ort::Session>(*env, model_path.c_str(), session_options);

}

void ONNXModel::prepare_input_tensors(std::vector<cv::Mat> imgs) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    // Get input from net
    logger::info << "Model inputs:" << logger::endl;
    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        // get input name
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        // get input type
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // get input shapes/dims
        auto input_node_dims = tensor_info.GetShape();

        logger::info << "\t" << input_names[i] << ": " << get_precision_str(get_data_precision(type)) << " ";
        logger::info << "[";
        for (size_t j = 0; j < input_node_dims.size() - 1; ++j) {
             logger::info << input_node_dims[j] << ","; 
        }
        logger::info << input_node_dims.back() <<  "]" << logger::endl;

    }

    // Get outputs from 
    logger::info << "Model outputs:" << logger::endl;
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        // get output name
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        // get output type
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // get output shapes/dims
        auto output_node_dims = tensor_info.GetShape();

        logger::info << "\t" << output_names[i] << ": " << get_precision_str(get_data_precision(type)) << " ";
        logger::info << "[";
        for (size_t j = 0; j < output_node_dims.size() - 1; ++j) {
             logger::info << output_node_dims[j] << ","; 
        }
        logger::info << output_node_dims.back() <<  "]" << logger::endl;

    }
}