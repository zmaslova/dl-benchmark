#pragma once
#include "logger.hpp"
#include <onnxruntime_cxx_api.h>
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

std::map<std::string, std::vector<std::string>> parse_input_files_arguments(const std::vector<std::string>& args, size_t max_files = 20);

std::map<std::string, std::string> parse_shape_or_layout_string(const std::string& parameter_string);

std::vector<int64_t> parse_shape_string();

DataPrecision get_data_precision(ONNXTensorElementDataType type);

std::string get_precision_str(DataPrecision p);

std::vector<float> string_to_vec(const std::string& mean_scale);


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
