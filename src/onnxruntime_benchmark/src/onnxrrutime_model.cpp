#include "onnxrrutime_model.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

ONNXModel::ONNXModel(const std::string& model_file, const std::string& layout_string, const std::string& shape_string,
    const std::string& mean_string, const std::string& scale_string, int batch_size, int num_threads)
    : batch_size(batch_size), num_threads(num_threads), mean(3, 0), scale(3, 1), dynamic_input(false) {

    input_layouts_map = parse_shape_or_layout_string(layout_string);
    for (const auto& [input_name, shape] : parse_shape_or_layout_string(shape_string)) {
        input_shapes_map.emplace(input_name, string_to_vec<long>(shape, ','));
    }

    if (!mean_string.empty()) {
        mean = string_to_vec<float>(mean_string, ' ');
        if (mean.size() != 3) {
            throw std::logic_error("Meast must have 3 values, one value per channel, bug given: " + mean_string);
        }
    }
    if (!scale_string.empty()) {
        scale = string_to_vec<float>(scale_string, ' ');
        if (scale.size() != 3) {
            throw std::logic_error("Meast must have 3 values, one value per channel, bug given: " + scale_string);
        }
    }

    read_model(model_file);
    get_input_output_info();
}

void ONNXModel::read_model(const std::string model_path) {
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ORT Bench");
    Ort::SessionOptions session_options;
    // Profile enabling?
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // if (device_type == DLBenchDevice::CPU) { // log device and nthreads
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // Parallel

    if (num_threads > 0) {
        session_options.SetIntraOpNumThreads(num_threads);
        logger::info << "Number of threads: " <<  num_threads << logger::endl;

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

    // check for dynamic input
    for (auto& shape : input_shapes) {
        if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
            dynamic_input = true;
        }
    }

    if (dynamic_input) {
        logger::info << "Model has dynamic input dimensions" << logger::endl;
    }

    if (batch_size == 0) {
        batch_size = input_shapes[0][0]; // TODO: extend to support various layouts (with or without batch)
    }
    else if (dynamic_input) {
        input_shapes[0][0] = batch_size;
        input_shapes[0][2] = 224;
        input_shapes[0][3] = 224;
        logger::info << "Set batch to " << std::to_string(batch_size) << logger::endl;
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

void ONNXModel::prepare_input_tensors(const std::map<std::string, std::vector<std::string>>& input_files) {
    input_tensors.emplace_back(get_image_tensor(input_files.begin()->second, {input_names[0], input_shapes[0], "", input_data_precisions[0], input_data_types[0], mean, scale}, batch_size));
}

void check_output(const std::vector<Ort::Value>& output_tensors, int batch_size) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        logger::debug << "Output tensor #" << i << logger::endl;
        const float* floatarr = output_tensors[i].GetTensorData<float>();
        for (int b = 0; b < batch_size; ++b) {
            logger::debug << "Batch #" << b << logger::endl;
            std::vector<float> res(floatarr + b*1000, floatarr +  b*1000 + 1000);
            std::vector<int> idx(1000);
            std::iota(idx.begin(), idx.end(), 0);
            auto max = std::max_element(res.begin(), res.end());
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&res](int l, int r){
                return  res[l] > res[r];
            });
            for (size_t j = 0; j < 5; ++j) {
                logger::debug << "id: " << idx[j] << " score " << res[idx[j]] << logger::endl;
            }
        }
    }
}

void ONNXModel::infer() {
    logger::info << input_names[0] << logger::endl;
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
    check_output(output_tensors, batch_size);
}
