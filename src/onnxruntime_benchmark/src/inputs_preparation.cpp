#include "inputs_preparation.hpp"

#include "args_handler.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

template <typename T>
using UniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

cv::Mat read_image(const std::string &img_path, size_t height, size_t width) {
    auto img = cv::imread(img_path);
    cv::resize(img, img, cv::Size(width, height));
    return img;
}

template <typename T>
const T get_mat_value(const cv::Mat &mat, size_t h, size_t w, size_t c) {
    switch (mat.type()) {
        case CV_8UC1:
            return static_cast<T>(mat.at<uchar>(h, w));
        case CV_8UC3:
            return static_cast<T>(mat.at<cv::Vec3b>(h, w)[c]);
        case CV_32FC1:
            return static_cast<T>(mat.at<float>(h, w));
        case CV_32FC3:
            return static_cast<T>(mat.at<cv::Vec3f>(h, w)[c]);
    }
    throw std::runtime_error("cv::Mat type is not recognized");
};

template <class T, class T2>
Ort::Value create_random_tensor(const InputDescr &input_descr,
                                T rand_min = std::numeric_limits<uint8_t>::min(),
                                T rand_max = std::numeric_limits<uint8_t>::max()) {
    logger::info << "\tRandomly generated data" << logger::endl;
    auto tensor_descr = input_descr.tensor_descr;

    auto allocator = Ort::AllocatorWithDefaultOptions();
    auto tensor =
        Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.type);
    auto tensor_data = tensor.GetTensorMutableData<T>();
    int64_t tensor_size =
        std::accumulate(tensor_descr.shape.begin(), tensor_descr.shape.end(), 1, std::multiplies<int64_t>());

    std::mt19937 gen(0);
    UniformDistribution<T2> distribution(rand_min, rand_max);
    for (int64_t i = 0; i < tensor_size; ++i) {
        tensor_data[i] = static_cast<T>(distribution(gen));
    }
    return tensor;
}

template <class T>
Ort::Value create_tensor_from_image(const InputDescr &input_descr, int batch_size, int start_index) {
    auto tensor_descr = input_descr.tensor_descr;
    const auto &files = input_descr.files;

    auto allocator = Ort::AllocatorWithDefaultOptions();
    auto tensor =
        Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.type);
    auto tensor_data = tensor.GetTensorMutableData<T>();

    size_t channels = tensor_descr.shape[1];
    size_t width = tensor_descr.shape[2];
    size_t height = tensor_descr.shape[3];

    for (int b = 0; b < batch_size; ++b) {
        const auto &file_path = files[(start_index + b) % files.size()];
        logger::info << "\t\t" << file_path << logger::endl;
        cv::Mat img = read_image(file_path, height, width);
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    size_t offset = b * channels * width * height + (ch * width * height + h * width + w);
                    tensor_data[offset] = (get_mat_value<T>(img, h, w, ch) - static_cast<T>(input_descr.mean[ch])) /
                                          static_cast<T>(input_descr.scale[ch]);
                }
            }
        }
    }
    return tensor;
}

template <class T>
Ort::Value create_tensor_from_binary(const InputDescr &input_descr, int batch_size, int start_index) {
    auto tensor_descr = input_descr.tensor_descr;
    const auto &files = input_descr.files;

    size_t tensor_size =
        std::accumulate(tensor_descr.shape.begin(), tensor_descr.shape.end(), 1, std::multiplies<int64_t>());
    auto allocator = Ort::AllocatorWithDefaultOptions();
    auto tensor =
        Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.type);
    auto tensor_data = tensor.GetTensorMutableData<char>();
    for (int b = 0; b < batch_size; ++b) {
        size_t input_id = (start_index + b) % files.size();
        const auto &file_path = files[input_id];
        logger::info << "\t\t" << file_path << logger::endl;

        std::ifstream binary_file(file_path, std::ios_base::binary | std::ios_base::ate);
        if (!binary_file) {
            throw std::runtime_error("Can't open " + file_path);
        }

        auto file_size = static_cast<std::size_t>(binary_file.tellg());
        auto input_size = tensor_size * sizeof(T) / batch_size;
        if (file_size != input_size) {
            throw std::invalid_argument("File " + file_path + " contains " + std::to_string(file_size) +
                                        " bytes but the mdoel expects " + std::to_string(input_size));
        }

        binary_file.seekg(0, std::ios_base::beg);
        if (!binary_file.good()) {
            throw std::runtime_error("Can't read " + file_path);
        }

        if (tensor_descr.layout != "CN") {
            binary_file.read(&tensor_data[b * input_size], input_size);
        }
        else {
            for (int i = 0; i < tensor_descr.channels(); ++i) {
                binary_file.read(&tensor_data[(i * batch_size + b) * sizeof(T)], sizeof(T));
            }
        }
    }

    return tensor;
}

Ort::Value get_tensor_from_image(const InputDescr &input_descr, int batch_size, int start_index) {
    auto precision = utils::get_data_precision(input_descr.tensor_descr.type);
    if (precision == utils::DataPrecision::FP32) {
        return create_tensor_from_image<float>(input_descr, batch_size, start_index);
    }
    else if (precision == utils::DataPrecision::S32) {
        return create_tensor_from_image<int32_t>(input_descr, batch_size, start_index);
    }
    else if (precision == utils::DataPrecision::S64) {
        return create_tensor_from_image<int64_t>(input_descr, batch_size, start_index);
    }

    throw std::invalid_argument("Unsuported tensor precision: " + utils::get_precision_str(precision));
}

Ort::Value get_tensor_from_binary(const InputDescr &input_descr, int batch_size, int start_index) {
    auto precision = utils::get_data_precision(input_descr.tensor_descr.type);
    if (precision == utils::DataPrecision::FP32) {
        return create_tensor_from_binary<float>(input_descr, batch_size, start_index);
    }
    else if (precision == utils::DataPrecision::S32) {
        return create_tensor_from_binary<int32_t>(input_descr, batch_size, start_index);
    }
    else if (precision == utils::DataPrecision::S64) {
        return create_tensor_from_binary<int64_t>(input_descr, batch_size, start_index);
    }
    else if (precision == utils::DataPrecision::BOOL) {
        return create_tensor_from_binary<uint8_t>(input_descr, batch_size, start_index);
    }
    throw std::invalid_argument("Unsuported tensor precision: " + utils::get_precision_str(precision));
}

Ort::Value get_random_tensor(const InputDescr &input_descr) {
    auto precision = utils::get_data_precision(input_descr.tensor_descr.type);
    if (precision == utils::DataPrecision::FP32) {
        return create_random_tensor<float, float>(input_descr);
    }
    else if (precision == utils::DataPrecision::S32) {
        return create_random_tensor<int32_t, int32_t>(input_descr);
    }
    else if (precision == utils::DataPrecision::S64) {
        return create_random_tensor<int64_t, int64_t>(input_descr);
    }
    else if (precision == utils::DataPrecision::BOOL) {
        return create_random_tensor<uint8_t, uint32_t>(input_descr, 0, 1);
    }
    throw std::invalid_argument("Unsuported tensor precision: " + utils::get_precision_str(precision));
}

std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo &inputs_info, int batch_size, int tensors_num) {
    std::vector<std::vector<Ort::Value>> tensors(tensors_num);
    int start_file_index = 0;
    for (int i = 0; i < tensors_num; ++i) {
        logger::info << "Input config " << i << logger::endl;
        for (const auto &[name, input_descr] : inputs_info) {
            const auto &tensor_descr = input_descr.tensor_descr;
            logger::info << " \t" << name << " (" << tensor_descr.layout << " "
                         << utils::get_precision_str(utils::get_data_precision(tensor_descr.type)) << " "
                         << shape_string(tensor_descr.shape) << ")" << logger::endl;

            if (!input_descr.files.empty() && static_cast<int>(input_descr.files.size()) < batch_size) {
                logger::warn << "\tNumber of input files is less than batch size. Some files will be duplicated."
                             << logger::endl;
            }
            if (input_descr.files.empty()) {
                tensors[i].push_back(get_random_tensor(input_descr));
            }
            else if (tensor_descr.is_image()) {
                tensors[i].push_back(get_tensor_from_image(input_descr, batch_size, start_file_index));
            }
            else {
                tensors[i].push_back(get_tensor_from_binary(input_descr, batch_size, start_file_index));
            }
        }
        start_file_index += batch_size;
    }
    return tensors;
}

InputsInfo get_inputs_info(const std::map<std::string, std::vector<std::string>> &input_files,
                           const std::vector<ONNXTensorDescr> &model_inputs,
                           const std::string &layout_string,
                           const std::string &shape_string,
                           const std::string &mean_string,
                           const std::string &scale_string) {
    // parse input layouts and input shapes
    std::map<std::string, std::string> input_layouts = parse_shape_or_layout_string(layout_string);
    std::map<std::string, std::vector<int64_t>> input_shapes;
    for (const auto &[input_name, shape] : parse_shape_or_layout_string(shape_string)) {
        input_shapes.emplace(input_name, string_to_vec<long>(shape, ','));
    }

    // parse mean and scale
    std::vector<float> mean = {0.f, 0.f, 0.f};
    if (!mean_string.empty()) {
        mean = string_to_vec<float>(mean_string, ' ');
        if (mean.size() != 3) {
            throw std::logic_error("Mean must have 3 values, one value per channel, but given: " + mean_string);
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
    bool is_dynamic_input = std::any_of(model_inputs.begin(), model_inputs.end(), [](const auto &tensor_descr) {
        return tensor_descr.is_dynamic();
    });
    bool is_dynamic_batch = std::any_of(model_inputs.begin(), model_inputs.end(), [](const auto &tensor_descr) {
        return tensor_descr.shape[0] == -1;
    });

    if (is_dynamic_input && !is_dynamic_batch && input_shapes.empty()) {
        throw std::logic_error("Shapes must be specified explicitly for models with dynamic input shapes.");
    }

    InputsInfo input_info;
    for (const auto &input : model_inputs) {
        InputDescr input_descr;
        input_descr.tensor_descr = input;
        auto &tensor_descr = input_descr.tensor_descr;

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

        auto &shape = tensor_descr.shape;
        if (!input_shapes.empty() && is_dynamic_input) {
            if (input_shapes.count(name) > 0) {
                shape = input_shapes.at(name);
            }
            else if (input_shapes.count("") > 0 && input_shapes.size() == 1) { // handle case without specifying name
                shape = input_shapes.at("");
            }
            else if (input_shapes.size() > 1) {
                throw std::invalid_argument("Input name " + name +
                                            " not found in the names provided with -shape argument.");
            }
        }
        else if (!input_shapes.empty() && !is_dynamic_batch) {
            logger::warn << "Model inputs are static, -shape option will be ignored!" << logger::endl;
        }

        auto &layout = tensor_descr.layout;
        if (!input_layouts.empty()) {
            if (input_layouts.count(name) > 0) {
                layout = input_layouts.at(name);
            }
            if (input_layouts.count("") > 0 && input_layouts.size() == 1) {
                layout = input_layouts.at("");
            }
            else if (input_layouts.size() > 1) {
                throw std::invalid_argument("Input name " + name +
                                            " not found in the names provided with -layout argument.");
            }
        }

        input_descr.mean = mean;
        input_descr.scale = scale;
        input_info.emplace(name, input_descr);
    }

    if (input_layouts.empty()) {
        logger::warn << "Layout will be detected automatically, as it wasn't provided explicitly." << logger::endl;
        for (auto &[name, input_descr] : input_info) {
            input_descr.tensor_descr.layout = utils::guess_layout_from_shape(input_descr.tensor_descr.shape);
        }
    }
    return input_info;
}
