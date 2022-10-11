#include "logger.hpp"
#include "tensors_handler.hpp"
#include <utils.hpp>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <numeric>
#include <string>
#include <vector>

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
Ort::Value create_random_tensor(const ONNXTensorDescr& tensor_descr, size_t batch_size) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

template<class T>
Ort::Value create_tensor_from_image(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, size_t batch_size) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    if (files.size() < batch_size) {
        logger::warn << "Number of input files less than batch size. Some files will be duplicated" << logger::endl;
    }
    size_t tensor_size = std::accumulate(tensor_descr.shape.begin(), tensor_descr.shape.end(), 1, std::multiplies<int64_t>());
    auto tensor = Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
    auto tensor_data = tensor.GetTensorMutableData<T>();

    size_t channels = tensor_descr.shape[1];
    size_t width = tensor_descr.shape[2];
    size_t height = tensor_descr.shape[3];
    cv::Mat img;


    for (size_t b = 0; b < batch_size; ++b) {
        img = cv::imread(files[b % files.size()]);
        cv::Mat tmp = centerSquareCrop(img);
        cv::resize(tmp, tmp, cv::Size(width, height));
        cv::cvtColor(tmp, tmp,
                 cv::ColorConversionCodes::COLOR_BGR2RGB); // ?
        cv::imshow("Test", tmp);
        cv::waitKey(0);
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    size_t offset = b * channels * width * height + (ch * width * height + h * width + w);
                    tensor_data[offset] = (get_mat_value<T>(tmp, h, w, ch) - static_cast<T>(tensor_descr.mean[ch]))
                    / static_cast<T>(tensor_descr.scale[ch]);;
                }
            }
        }
    }
    // auto ch = img.channels();
    // std::vector<uchar> imgdata(img.data, img.data + img.total());
    // T* t = tensor.GetTensorMutableData<T>();
    // std::vector<T> tmp(t, t + tensor_size);
    return tensor;
}

template<class T>
Ort::Value create_tensor_from_binary(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    return Ort::Value::CreateTensor(allocator, tensor_descr.shape.data(), tensor_descr.shape.size(), tensor_descr.elem_type);
}

Ort::Value get_image_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto precision = tensor_descr.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_image<float>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_image<int32_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_image<int64_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_image<bool>(files, tensor_descr, batch_size);
    }

    throw std::runtime_error("Unsuported tensor precision: " + get_precision_str(precision));
}

Ort::Value get_binary_tensor(const std::vector<std::string>& files, const ONNXTensorDescr& tensor_descr, int batch_size) {
    auto precision = tensor_descr.precision;
    if (precision == DataPrecision::FP32) {
        return create_tensor_from_binary<float>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S32) {
        return create_tensor_from_binary<int32_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::S64) {
        return create_tensor_from_binary<int64_t>(files, tensor_descr, batch_size);
    }
    else if (precision == DataPrecision::BOOL) {
        return create_tensor_from_binary<bool>(files, tensor_descr, batch_size);
    }

    throw std::runtime_error("Unsuported tensor precision: " + get_precision_str(precision));
}

