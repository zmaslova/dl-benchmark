#include "inputs_info.hpp"
#include "tensors_handler.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <numeric>
#include <string>
#include <vector>

size_t get_batch_size(const InputsInfo& inputs_info) {
   size_t batch_size = 0;
    for (auto& [name, info] : inputs_info) {
        auto& tdesc = info.tdesc;
        std::size_t batch_index = tdesc.layout.find("N");
        if (batch_index != std::string::npos) {
            if (batch_size == 0) {
                batch_size = tdesc.shape[batch_index];
            }
            else if (batch_size != tdesc.shape[batch_index]) {
                throw std::logic_error("Batch size is different for different inputs!");
            }
        }
    }
    if (batch_size == 0) {
        logger::warn << "Batch dimension not found, batch is set to 1" << logger::endl;
        batch_size = 1;
    }
    return batch_size;
}


std::string guess_layout_from_shape(std::vector<int64>& shape) {
    if (shape.size() == 2) {
        return "NC";
    }
    if (shape.size() == 3) {
        return shape[0] > 4 && shape[2] <= 4 ? "HWC" : "CHW";
    }
    if (shape.size() == 4) {
        return shape[1] > 4 && shape[3] <= 4 ? "NHWC" : "NCHW";
    }
    throw std::invalid_argument("Unsupported shape with size " + std::to_string(shape.size()));
}

bool ONNXTensorDescr::is_image() const {
    return (layout == "NCHW" ||
            layout == "NHWC" ||
            layout == "CHW" ||
            layout == "HWC")
            && channels() == 3;

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

template <typename T>
const T get_mat_value(const cv::Mat& mat, size_t h, size_t w, size_t c) {
    switch (mat.type()) {
        case CV_8UC1:  return (T)mat.at<uchar>(h, w);
        case CV_8UC3:  return (T)mat.at<cv::Vec3b>(h, w)[c];
        case CV_32FC1: return (T)mat.at<float>(h, w);
        case CV_32FC3: return (T)mat.at<cv::Vec3f>(h, w)[c];
    }
    throw std::runtime_error("cv::Mat type is not recognized");
};

cv::Mat centerSquareCrop(const cv::Mat& image) {
    if (image.cols >= image.rows) {
        return image(cv::Rect((image.cols - image.rows) / 2, 0, image.rows, image.rows));
    }
    return image(cv::Rect(0, (image.rows - image.cols) / 2, image.cols, image.cols));
}

template<class T>
Ort::Value create_random_tensor(const InputDescr& input_descr) {
    logger::info << "\t" << "random tensor" << logger::endl;
    auto tensor_descr = input_descr.tdesc;
    auto allocator = Ort::AllocatorWithDefaultOptions();
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

template<class T>
Ort::Value create_tensor_from_image(const InputDescr& input_descr, int batch_size, int start_index) {
    auto tensor_descr = input_descr.tdesc;
    const auto& files = input_descr.files;

    auto allocator = Ort::AllocatorWithDefaultOptions();
    if (files.size() < batch_size) {
        logger::warn << "Number of input files less than batch size. Some files will be duplicated" << logger::endl;
    }
    //size_t tensor_size = std::accumulate(tensor_descr.shape.begin(), tensor_descr.shape.end(), 1, std::multiplies<int64_t>());
    tensor_descr.set_batch(batch_size);
    auto tensor = Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
    auto tensor_data = tensor.GetTensorMutableData<T>();

    size_t channels = tensor_descr.shape[1];
    size_t width = tensor_descr.shape[2];
    size_t height = tensor_descr.shape[3];
    cv::Mat img;

    for (size_t b = 0; b < batch_size; ++b) {
        logger::info << "\t\t" << files[(start_index + b) % files.size()] << logger::endl;
        img = cv::imread(files[(start_index + b) % files.size()]);
        cv::Mat tmp = centerSquareCrop(img);
        cv::resize(tmp, tmp, cv::Size(width, height));
        cv::cvtColor(tmp, tmp,
                 cv::ColorConversionCodes::COLOR_BGR2RGB); // ?
        // cv::imshow("Test", tmp);
        // cv::waitKey(0);
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    size_t offset = b * channels * width * height + (ch * width * height + h * width + w);
                    tensor_data[offset] = (get_mat_value<T>(tmp, h, w, ch) - static_cast<T>(input_descr.mean[ch]))
                        / static_cast<T>(input_descr.scale[ch]);;
                }
            }
        }
    }

    logger::info << "}" << logger::endl;
    return tensor;
}

template<class T>
Ort::Value create_tensor_from_binary(const InputDescr& input_descr, int batch_size, int start_index) {
    auto tensor_descr = input_descr.tdesc;
    auto allocator = Ort::AllocatorWithDefaultOptions();
    logger::info << "\t" << "random tensor" << logger::endl;
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

Ort::Value get_image_tensor(const InputDescr& input_descr, int batch_size, int start_index) {
    auto precision = input_descr.tdesc.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_image<float>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_image<int32_t>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_image<int64_t>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_image<bool>(input_descr, batch_size, start_index);
    }

    throw std::invalid_argument("Unsuported tensor precision: " + get_precision_str(precision));
}

Ort::Value get_binary_tensor(const InputDescr& input_descr, int batch_size, int start_index) {
    auto precision = input_descr.tdesc.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_binary<float>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_binary<int32_t>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_binary<int64_t>(input_descr, batch_size, start_index);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_binary<bool>(input_descr, batch_size, start_index);
    }
    throw std::invalid_argument("Unsuported tensor precision: " + get_precision_str(precision));
}

Ort::Value get_random_tensor(const InputDescr& input_descr, int batch_size) {
    auto precision = input_descr.tdesc.precision;
    if (precision == DataPrecision::FP32) {
        return create_random_tensor<float>(input_descr);
    }
    else if (precision == DataPrecision::S32) {
        return create_random_tensor<int32_t>(input_descr);
    }
    else if (precision == DataPrecision::S64) {
        return create_random_tensor<int64_t>(input_descr);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_random_tensor<bool>(input_descr);
    }
    throw std::invalid_argument("Unsuported tensor precision: " + get_precision_str(precision));
}

InputsInfo get_inputs_info(const std::map<std::string, std::vector<std::string>>& input_files, const std::vector<ONNXTensorDescr>& model_inputs, const std::string& layout_string,
    const std::string& shape_string, const std::string& mean_string, const std::string& scale_string) {
    // parse input layouts and input shapes
    std::map<std::string, std::string> input_layouts = parse_shape_or_layout_string(layout_string);
    std::map<std::string, std::vector<int64_t>> input_shapes;
    for (const auto& [input_name, shape] : parse_shape_or_layout_string(shape_string)) {
        input_shapes.emplace(input_name, string_to_vec<long>(shape, ','));
    }

    // parse mean and scale
    std::vector<float> mean = {0.f, 0.f, 0.f};
    if (!mean_string.empty()) {
        mean = string_to_vec<float>(mean_string, ' ');
        if (mean.size() != 3) {
            throw std::logic_error("Mean must have 3 values, one value per channel, bug given: " + mean_string);
        }
    }
    std::vector<float> scale = {1.f, 1.f, 1.f};
    if (!scale_string.empty()) {
        scale = string_to_vec<float>(scale_string, ' ');
        if (scale.size() != 3) {
            throw std::logic_error("Scale must have 3 values, one value per channel, bug given: " + scale_string);
        }
    }

    // Check dynamic inputs
    bool is_dynamic_input = std::any_of(model_inputs.begin(), model_inputs.end(), [](const auto& tdesc) {
                                                                                    return tdesc.is_dynamic();
                                                                                });
    bool is_dynamic_batch = std::any_of(model_inputs.begin(), model_inputs.end(), [](const auto& tdesc) {
                                                                                    return tdesc.shape[0] == -1;
                                                                                });

    if (is_dynamic_input && !is_dynamic_batch && input_shapes.empty()) {
        throw std::logic_error("Shapes must be specified explicitly for models with dynamic inputs.");
    }

    InputsInfo input_info;
    for (const auto& input : model_inputs) {
        InputDescr input_descr;
        input_descr.tdesc = input;
        auto& tdesc = input_descr.tdesc;

        std::string name = input.name;
        if (input_files.count(name) > 0) {
            input_descr.files = input_files.at(name);
        }
        else if (input_files.count("") > 0 && input_files.size() == 1) { // case with 1 input wihtout specifying name
            input_descr.files = input_files.at("");
        }
        else if (input_files.size() > 1) {
            throw std::invalid_argument("Input name " + name + " not found in the names provided with -i argument.");
        }

        auto& shape = tdesc.shape;
        if (!input_shapes.empty() && is_dynamic_input) {
            if (input_shapes.count(name) > 0) {
                shape = input_shapes.at(name);
            }
            else if (input_shapes.count("") > 0 && input_shapes.size() == 1) { // case with 1 input wihtout specifying name
                shape = input_shapes.at("");
            }
            else if (input_shapes.size() > 1) {
                throw std::invalid_argument("Input name " + name + " not found in the names provided with -shapes argument.");
            }
        }
        else if (!is_dynamic_batch) {
            logger::warn << "Model inputs is static, -shape option will be ignored!" << logger::endl;
        }

        auto& layout = tdesc.layout;
        if (!input_layouts.empty()) {
            if (input_layouts.count(name) > 0) {
                layout = input_layouts.at(name);
            }
            if (input_layouts.count("") > 0 && input_layouts.size() == 1) {
                layout = input_layouts.at("");
            }
            else if (input_layouts.size() > 1) {
                throw std::invalid_argument("Input name " + name + " not found in the names provided with -layout argument.");
            }
        }

        input_descr.mean = mean;
        input_descr.scale = scale;
        input_info.emplace(name, input_descr);
    }

    if (input_layouts.empty()) {
        logger::warn << "Layout will be detected automatically, as it wasn't provided explicitly." << logger::endl;
        for (auto& [name, input_descr] : input_info) {
            input_descr.tdesc.layout = guess_layout_from_shape(input_descr.tdesc.shape);
        }
    }
    return input_info;
}

std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo& inputs_info, int batch_size, int tensors_num) { //inputs_info.begin()->second.batch;
    std::vector<std::vector<Ort::Value>> tensors(tensors_num);
    int start_file_index = 0;
    for (int n = 0; n < tensors_num; ++n) {
        logger::info << "Input config " << n << logger::endl;
        for (const auto& [name, input_descr] : inputs_info) {
            const auto& tensor_descr = input_descr.tdesc;
            logger::info << " \t" << name << " (" << tensor_descr.layout  <<
                " " << get_precision_str(get_data_precision(tensor_descr.elem_type)) <<
                " " << shape_string(tensor_descr.shape) << ")" << " {" << logger::endl;
            if (input_descr.files.empty()) {
                tensors[n].push_back(get_random_tensor(input_descr, batch_size));
            }
            else if (tensor_descr.is_image()) {
                tensors[n].push_back(get_image_tensor(input_descr, batch_size, start_file_index));
            }
            else {
                tensors[n].push_back(get_binary_tensor(input_descr, batch_size, start_file_index));
            }
        }
        start_file_index += batch_size;
    }
    return tensors;
}
