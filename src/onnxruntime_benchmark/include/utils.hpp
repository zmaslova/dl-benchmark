#pragma once
#include "logger.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <exception>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct InputDescr;
using InputsInfo = std::map<std::string, InputDescr>;

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
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, DataPrecision::FP32},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, DataPrecision::FP16},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, DataPrecision::U8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, DataPrecision::S8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, DataPrecision::S32},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, DataPrecision::S64},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, DataPrecision::BOOL},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, DataPrecision::UNKNOWN}};

static const std::map<std::string, DataPrecision> precision_to_str_map = {
    {"FP32", DataPrecision::FP32},
    {"FP16", DataPrecision::FP16},
    {"FP11", DataPrecision::FP11},
    {"U8", DataPrecision::U8},
    {"S8", DataPrecision::S8},
    {"S16", DataPrecision::S16},
    {"S32", DataPrecision::S32},
    {"S64", DataPrecision::S64},
    {"INT8", DataPrecision::S8},
    {"INT16", DataPrecision::S16},
    {"INT32", DataPrecision::S32},
    {"INT64", DataPrecision::S64},
    {"BOOL", DataPrecision::BOOL},
    {"MIXED", DataPrecision::MIXED},
};

DataPrecision get_data_precision(ONNXTensorElementDataType type);

std::string get_precision_str(DataPrecision p);

int get_batch_size(const InputsInfo &inputs_info);

void set_batch_size(InputsInfo &inputs_info, int batch_size);

std::string guess_layout_from_shape(std::vector<int64_t> &shape);

static inline void catcher() noexcept {
    if (std::current_exception()) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception &error) {
            logger::err << error.what() << logger::endl;
        } catch (...) {
            logger::err << "Non-exception object thrown" << logger::endl;
        }
        std::exit(1);
    }
    std::abort();
}
