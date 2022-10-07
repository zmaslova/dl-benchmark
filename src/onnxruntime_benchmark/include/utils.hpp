#pragma once
#include "logger.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <vector>

enum class DataPrecision : unsigned int {
    FP32 = 0,
    FP11,
    FP16,
    U8,
    S8,
    S16,
    S32,
    S64,
    BOOL,
    MIXED,
    UNKNOWN
};

const std::map<ONNXTensorElementDataType, DataPrecision> onnx_dtype_to_precision_map = {
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,     DataPrecision::FP32    },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,   DataPrecision::FP16    },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,     DataPrecision::U8      },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,      DataPrecision::S8      },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,     DataPrecision::S32     },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,     DataPrecision::S64     },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,      DataPrecision::BOOL    },
    { ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, DataPrecision::UNKNOWN }
};

static const std::map<std::string, DataPrecision> precision_to_str_map = {
    { "FP32",   DataPrecision::FP32 },
    { "FP16",   DataPrecision::FP16 },
    { "FP11",   DataPrecision::FP11 },
    { "U8",     DataPrecision::U8   },
    { "S8",     DataPrecision::S8   },
    { "S16",    DataPrecision::S16  },
    { "S32",    DataPrecision::S32  },
    { "S64",    DataPrecision::S64  },
    { "INT8",   DataPrecision::S8   },
    { "INT16",  DataPrecision::S16  },
    { "INT32",  DataPrecision::S32  },
    { "INT64",  DataPrecision::S64  },
    { "BOOL",   DataPrecision::BOOL },
    { "MIXED",  DataPrecision::MIXED },
};

struct ONNXTensorDescr {
   std::string name;
   std::vector<int64_t> shape;
   DataPrecision precision;
   ONNXTensorElementDataType elem_type;
};


std::vector<std::string> parse_input_args();

DataPrecision get_data_precision(ONNXTensorElementDataType type);

std::string get_precision_str(DataPrecision p);

Ort::Value get_image_tensor(const std::vector<std::string>& files, ONNXTensorDescr& tensor, int batch_size);

Ort::Value get_binary_tensor(const std::vector<std::string>& files, ONNXTensorDescr& tensor, int batch_size);

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

template<class OutStream> 
void print_dims(std::vector<int64_t> dims, OutStream s) {
    s << "[";
    for (size_t j = 0; j < dims.size() - 1; ++j) {
        s << dims[j] << ","; 
    }
    s << dims.back() <<  "]";
}

static inline void catcher() noexcept {
    if (std::current_exception()) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            logger::err << error.what() << logger::endl;
        } catch (...) {
            logger::err << "Non-exception object thrown" << logger::endl;
        }
        std::exit(1);
    }
    std::abort();
}
