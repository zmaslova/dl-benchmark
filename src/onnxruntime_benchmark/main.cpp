#include "onnxrrutime_model.hpp"
#include "utils.hpp"
#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace {
constexpr char help_msg[] = "show the help message and exit";
DEFINE_bool(h, false, help_msg);

constexpr char model_msg[] = "path to an .onnx file with a trained model";
DEFINE_string(m, "", model_msg);

constexpr char input_msg[] = "path to an input to process. The input must be an image and/or binaries, a folder of images and/or binaries";
DEFINE_string(i, "", input_msg);

constexpr char batch_size_msg[] = "batch size value. If not provided, batch size value is determined from the model";
DEFINE_uint32(b, 0, batch_size_msg);

constexpr char shape_msg[] = "shape for network input";
DEFINE_string(shape, "", shape_msg);

constexpr char layout_msg[] = "layout for network input";
DEFINE_string(layout, "", layout_msg);

constexpr char threads_num_msg[] = "number of threads.";
DEFINE_uint32(nthreads, 0, threads_num_msg);

constexpr char iterations_num_msg[] = "number of iterations. If not provided, default time limit is set";
DEFINE_uint32(niter, 0, iterations_num_msg);

constexpr char time_msg[] = "time limit for inference in seconds";
DEFINE_uint32(t, 0, time_msg);

constexpr char save_report_msg[] = "save report in JSON format.";
DEFINE_bool(save_report, false, save_report_msg);

constexpr char report_folder_msg[] = "destination folder for report.";
DEFINE_string(report_folder, "", report_folder_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout << "onnxruntime_benchmark"
                  << "\nOptions:"
                  << "\n\t[-h]                                           " << help_msg
                  << "\n\t[-help]                                      print help on all arguments"
                  << "\n\t -m <MODEL FILE>                             " << model_msg
                  << "\n\t -i <INPUT>                                  " << input_msg
                  << "\n\t[-b <NUMBER>]                                " << batch_size_msg
                  << "\n\t[-shape <[N,C,H,W]>]                         " << shape_msg
                  << "\n\t[-layout <[NCHW]>]                           " << layout_msg
                  << "\n\t[-nthreads <NUMBER>]                         " << threads_num_msg
                  << "\n\t[-niter <NUMBER>]                            " << iterations_num_msg
                  << "\n\t[-t <NUMBER>]                                " << time_msg
                  << "\n\t[-save_report]                               " << save_report_msg
                  << "\n\t[-report_folder <PATH>]                      " << report_folder_msg
                  << "\n";
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
}
}

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);
    logger::info << "Parsing input arguments" << logger::endl;
    parse(argc, argv);
    logger::info << FLAGS_m << logger::endl;
    logger::info << FLAGS_i << logger::endl;
    auto img = cv::imread(FLAGS_i);
    //cv::imshow("test", img);
    //cv::waitKey(0);
    ONNXModel model(FLAGS_nthreads);
    model.read_model(FLAGS_m);
    model.prepare_input_tensors({img});
    return 0;
}
