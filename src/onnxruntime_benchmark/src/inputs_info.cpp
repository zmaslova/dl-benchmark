#include "inputs_info.hpp"
#include "tensors_handler.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// std::string guess_layout_from_shape(std::vector<int64_t>& shape) {
//     if (shape.size() == 2) {
//         return "NC";
//     }
//     if (shape.size() == 3) {
//         return shape[0] > 4 && shape[2] <= 4 ? "HWC" : "CHW";
//     }
//     if (shape.size() == 4) {
//         return shape[1] > 4 && shape[3] <= 4 ? "NHWC" : "NCHW";
//     }
//     throw std::invalid_argument("Unsupported shape with size " + std::to_string(shape.size()));
// }

// InputsInfo get_inputs_info(const std::vector<ONNXTensorDescr>& model_inputs, std::map<std::string, std::vector<std::string>> input_files,
//     const std::string& layout_string, const std::string& shape_string, const std::string& mean_string, const std::string& scale_string, int batch_size) {
//     // parse input layouts and input shapes
//     std::map<std::string, std::string> input_layouts = parse_shape_or_layout_string(layout_string);
//     std::map<std::string, std::vector<int64_t>> input_shapes;
//     for (const auto& [input_name, shape] : parse_shape_or_layout_string(shape_string)) {
//         input_shapes.emplace(input_name, string_to_vec<long>(shape, ','));
//     }

//     // parse mean and scale
//     std::vector<float> mean = {0.f, 0.f, 0.f};
//     if (!mean_string.empty()) {
//         mean = string_to_vec<float>(mean_string, ' ');
//         if (mean.size() != 3) {
//             throw std::logic_error("Mean must have 3 values, one value per channel, bug given: " + mean_string);
//         }
//     }
//     std::vector<float> scale = {1.f, 1.f, 1.f};
//     if (!scale_string.empty()) {
//         scale = string_to_vec<float>(scale_string, ' ');
//         if (scale.size() != 3) {
//             throw std::logic_error("Scale must have 3 values, one value per channel, bug given: " + scale_string);
//         }
//     }

//     // Check dynamic inputs
//     bool dynamic_model = std::any_of(model_inputs.begin(), model_inputs.end(), [](const auto& tdesc) {
//                                                                                     return tdesc.is_dynamic();
//                                                                                 });

//     if (dynamic_model && input_shapes.empty()) {
//         throw std::logic_error("Shapes must be specified explicitly for models with dynamic inputs.");
//     }


//     InputsInfo input_info;
//     for (const auto& input : model_inputs) {
//         InputDescr input_descr;
//         input_descr.tdesc = input;
//         auto& tdesc = input_descr.tdesc;

//         std::string name = input.name;
//         if (input_files.count(name) > 0) {
//             input_descr.files = input_files.at(name);
//         }
//         else if (input_files.count("") > 0 && input_files.size() == 1) { // case with 1 input wihtout specifying name
//             input_descr.files = input_files.at("");
//         }
//         else if (input_files.size() > 1) {
//             throw std::invalid_argument("Input name " + name + " not found in the names provided with -i argument.");
//         }

//         auto& shape = tdesc.shape;
//         if (!input_shapes.empty()) {
//             if (input_shapes.count(name) > 0) {
//                 shape = input_shapes.at(name);
//             }
//             else if (input_shapes.count("") > 0 && input_shapes.size() == 1) { // case with 1 input wihtout specifying name
//                 shape = input_shapes.at("");
//             }
//             else if (input_shapes.size() > 1) {
//                 throw std::invalid_argument("Input name " + name + " not found in the names provided with -shapes argument.");
//             }
//         }

//         auto& layout = tdesc.layout;
//         if (!input_layouts.empty()) {
//             if (input_layouts.count(name) > 0) {
//                 layout = input_layouts.at(name);
//             }
//             if (input_layouts.count("") > 0 && input_layouts.size() == 1) {
//                 layout = input_layouts.at("");
//             }
//             else if (input_layouts.size() > 1) {
//                 throw std::invalid_argument("Input name " + name + " not found in the names provided with -layout argument.");
//             }
//         }
//         input_info.emplace(name, input_descr);
//     }

//     if (input_layouts.empty()) {
//         logger::warn << "Layout will be detected automatically, as it wasn't provided explicitly." << logger::endl;
//         for (auto& [name, input_descr] : input_info) {
//             input_descr.tdesc.layout = guess_layout_from_shape(input_descr.tdesc.shape);
//         }
//     }
//     return input_info;
// }

// std::vector<std::vector<Ort::Value>> get_input_tensors(const InputsInfo& inputs_info, int tensors_num) {
//     int batch = inputs_info.begin()->second.batch;
//     std::vector<std::vector<Ort::Value>> tensors(tensors_num);
//     int start_file_index = 0;
//     for (int n = 0; n < tensors_num; ++n) {
//         logger::info << "Input config " << n << logger::endl;
//         for (const auto& [name, input_descr] : inputs_info) {
//             const auto& tensor_descr = input_descr.tdesc;
//             logger::info << name << " (" << tensor_descr.layout  <<
//                 " " << get_precision_str(get_data_precision(tensor_descr.elem_type)) <<
//                 " " << shape_string(tensor_descr.shape) << ")" << logger::endl;
//             if (input_descr.files.empty()) {
//                 tensors[n].push_back(get_random_tensor(input_descr));
//             }
//             else if (tensor_descr.is_image()) {
//                 tensors[n].push_back(get_image_tensor(input_descr, start_file_index));
//             }
//             else {
//                 tensors[n].push_back(get_binary_tensor(input_descr, start_file_index));
//             }
//         }
//         start_index += batch
//     }
//     return tensors;
// }
