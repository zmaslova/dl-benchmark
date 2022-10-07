#include "onnxrrutime_model.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>
#include <vector>

ONNXModel::ONNXModel(const std::string& model_file, int num_threads, int batch_size)
    : num_threads(num_threads), batch_size(batch_size) {
    read_model(model_file);
    get_input_output_info();
}

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

void ONNXModel::get_input_output_info() {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    // Get input from model
    logger::info << "Model inputs:" << logger::endl;
    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        // get input name
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));
        // get input type
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        input_data_types.push_back(type);
        input_data_precisions.push_back(get_data_precision(type));

        // get input shapes/dims
        auto input_node_shape = tensor_info.GetShape();
        input_shapes.push_back(input_node_shape);
        logger::info << "\t" << input_names[i] << ": " << get_precision_str(get_data_precision(type)) << " ";
        print_dims(input_node_shape, logger::info);
        logger::info << logger::endl;
    }

    if (batch_size == 0) {
        batch_size = input_shapes[0][0]; // TODO: extend to support various layouts (with or without batch)
    }

    // Get outputs from model
    logger::info << "Model outputs:" << logger::endl;
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        // get output name
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_names_ptr.push_back(std::move(output_name));
        // get output type
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // get output shapes/dims
        auto output_node_shape = tensor_info.GetShape();
        logger::info << "\t" << output_names[i] << ": " << get_precision_str(get_data_precision(type)) << " ";
        print_dims(output_node_shape, logger::info);
        logger::info << logger::endl;
    }

    //input_tensors.emplace_back(get_tensor_from_image())
}

void ONNXModel::prepare_input_tensors(const std::vector<std::string>& input_files) {
    input_tensors.emplace_back(get_image_tensor(input_files, {input_names[0], input_shapes[0], input_data_precisions[0], input_data_types[0]}, batch_size));
}

void ONNXModel::infer() {
    logger::info << input_names[0] << logger::endl;
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < 5; i++) {
        std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
    }
}
