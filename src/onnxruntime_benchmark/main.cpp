#include "args_handler.hpp"
#include "inputs_preparation.hpp"
#include "onnxruntime_model.hpp"
#include "statistics.hpp"
#include "utils.hpp"

#include <gflags/gflags.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr char help_msg[] = "show the help message and exit";
DEFINE_bool(h, false, help_msg);

constexpr char model_msg[] = "path to an .onnx file with a trained model";
DEFINE_string(m, "", model_msg);

constexpr char input_msg[] =
    "path to an input to process. The input must be an image and/or binaries, a folder of images and/or binaries";
DEFINE_string(i, "", input_msg);

constexpr char batch_size_msg[] = "batch size value. If not provided, batch size value is determined from the model";
DEFINE_uint32(b, 0, batch_size_msg);

constexpr char shape_msg[] = "shape for network input";
DEFINE_string(shape, "", shape_msg);
//
constexpr char layout_msg[] = "layout for network input";
DEFINE_string(layout, "", layout_msg);

constexpr char input_mean_msg[] =
    "Mean values per channel for input image.\n"
    "                                                     Applicable only for models with one image input.\n"
    "                                                     Example: -mean 123.675 116.28 103.53";
DEFINE_string(mean, "", input_mean_msg);

constexpr char input_scale_msg[] =
    "Scale values per channel for input image.\n"
    "                                                     Applicable only for models with one image input.\n"
    "                                                     Example: -scale 58.395 57.12 57.375";
DEFINE_string(scale, "", input_scale_msg);

constexpr char threads_num_msg[] = "number of threads.";
DEFINE_uint32(nthreads, 0, threads_num_msg);

constexpr char tensors_num_msg[] = "number of input tensors to inference. If not provided, default value is set";
DEFINE_uint32(ntensors, 0, tensors_num_msg);

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
                  << "\n\t[-h]                                         " << help_msg
                  << "\n\t[-help]                                      print help on all arguments"
                  << "\n\t -m <MODEL FILE>                             " << model_msg
                  << "\n\t[-i <INPUT>]                                 " << input_msg
                  << "\n\t[-b <NUMBER>]                                " << batch_size_msg
                  << "\n\t[-shape <[N,C,H,W]>]                         " << shape_msg
                  << "\n\t[-layout <[NCHW]>]                           " << layout_msg
                  << "\n\t[-mean <R G B>]                              " << input_mean_msg
                  << "\n\t[-scale <R G B>]                             " << input_scale_msg
                  << "\n\t[-nthreads <NUMBER>]                         " << threads_num_msg
                  << "\n\t[-ntensors <NUMBER>]                          " << tensors_num_msg
                  << "\n\t[-niter <NUMBER>]                            " << iterations_num_msg
                  << "\n\t[-t <NUMBER>]                                " << time_msg
                  << "\n\t[-save_report]                               " << save_report_msg
                  << "\n\t[-report_folder <PATH>]                      " << report_folder_msg << "\n";
        exit(0);
    }
    if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
}

static void log_step(const std::string optional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> steps = {{1, "Parsing and validating input arguments"},
                                                        {2, "Loading ONNX Runtime"},
                                                        {3, "Reading model files"},
                                                        {4, "Configuring input of the model"},
                                                        {5, "Setting execution parameters"},
                                                        {6, "Creating input tensors"},
                                                        {7, "Measuring model performance"},
                                                        {8, "Saving statistics report"}};

    step_id++;
    if (steps.count(step_id) == 0) {
        throw std::invalid_argument("Invalid number of step " + std::to_string(step_id) +
                                    " was provided, number of the step should be less than " +
                                    std::to_string(steps.size()));
    }

    std::cout << "[Step " << step_id << "/" << steps.size() << "] " << steps.at(step_id)
              << (optional_info.empty() ? "" : " (" + optional_info + ")") << std::endl;
}

void catcher() noexcept {
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

} // namespace

int main(int argc, char *argv[]) {
    std::set_terminate(catcher);
    log_step();
    logger::info << "Parsing input arguments" << logger::endl;
    parse(argc, argv);

    logger::info << "Checking input files" << logger::endl;
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    auto input_files = parse_input_files_arguments(gflags::GetArgvs());
    log_step();
    logger::info << "ONNX Runtime version: " << OrtGetApiBase()->GetVersionString() << logger::endl;
    log_step();
    ONNXModel model(FLAGS_nthreads);
    logger::info << "Reading model " << FLAGS_m << logger::endl;
    auto start_time = HighresClock::now();
    model.read_model(FLAGS_m);
    auto read_model_time = ns_to_ms(HighresClock::now() - start_time);
    logger::info << "Read model took " << format_double(read_model_time) << " ms" << logger::endl;
    logger::info << "Model inputs/outputs info" << logger::endl;
    model.fill_inputs_outputs_info();
    auto io_tensors_info = model.get_io_tensors_info();
    log_model_inputs_outputs(io_tensors_info);

    log_step();
    auto inputs_info =
        get_inputs_info(input_files, io_tensors_info.first, FLAGS_layout, FLAGS_shape, FLAGS_mean, FLAGS_scale);

    // determine batch size
    int batch_size = get_batch_size(inputs_info);
    if (batch_size == -1 && FLAGS_b > 0) {
        batch_size = FLAGS_b;
    }
    else if (batch_size == -1) {
        throw std::logic_error("Model has dynamic batch size, but -b option wasn't provided.");
    }
    else if (FLAGS_b > 0) {
        throw std::logic_error("Can't set batch for model with static batch dimension.");
    }

    // setting batch
    set_batch_size(inputs_info, batch_size);
    logger::info << "Set batch to " << batch_size << logger::endl;

    log_step();
    // number of input tensors to infer (analogue of infer requests from benchmark_app)
    int num_tensors = FLAGS_ntensors;
    if (FLAGS_ntensors == 0) {
        num_tensors = 1;
    }

    // set and align iterations limit
    int64_t num_iterations = FLAGS_niter;
    if (num_iterations > 0) {
        num_iterations = ((num_iterations + num_tensors - 1) / num_tensors) * num_tensors;
        if (FLAGS_niter != num_iterations) {
            logger::warn << "Provided number of iterations " << FLAGS_niter << " was changed to " << num_iterations
                         << " to be aligned with number of tensors " << num_tensors << logger::endl;
        }
    }

    // set time limit
    uint32_t time_limit_sec = 0;
    if (FLAGS_t != 0) {
        time_limit_sec = FLAGS_t;
    }
    else if (FLAGS_niter == 0) {
        time_limit_sec = 60;
        logger::warn << "Default time limit is set: " << time_limit_sec << " seconds " << logger::endl;
    }
    uint64_t time_limit_ns = sec_to_ns(time_limit_sec);

    log_step();
    auto tensors = get_input_tensors(inputs_info, batch_size, num_tensors);

    log_step();
    // warm up before benhcmarking
    model.run(tensors[0]);
    logger::info << "Warming up inference took " << format_double(model.get_latencies()[0]) << " ms" << logger::endl;
    model.reset_timers();

    int64_t iteration = 0;
    start_time = HighresClock::now();
    auto uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
    while ((num_iterations != 0 && iteration < num_iterations) ||
           (time_limit_ns != 0 && static_cast<uint64_t>(uptime) < time_limit_ns)) {
        model.run(tensors[iteration % tensors.size()]);
        ++iteration;
        uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
    }

    log_step();
    Metrics metrics(model.get_latencies(), batch_size);
    double total_time = model.get_total_time_ms();

    // Performance metrics report
    logger::info << "Count: " << iteration << " iterations" << logger::endl;
    logger::info << "Duration: " << format_double(total_time) << " ms" << logger::endl;
    logger::info << "Latency:" << logger::endl;
    logger::info << "\tMedian   " << format_double(metrics.median) << " ms" << logger::endl;
    logger::info << "\tAverage: " << format_double(metrics.avg) << " ms" << logger::endl;
    logger::info << "\tMin:     " << format_double(metrics.min) << " ms" << logger::endl;
    logger::info << "\tMax:     " << format_double(metrics.max) << " ms" << logger::endl;
    logger::info << "Throughput: " << format_double(metrics.fps) << " FPS" << logger::endl;

    return 0;
}
