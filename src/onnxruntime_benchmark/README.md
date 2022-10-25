# ONNX Runtime Benchmark
The tool allows to measure deep learning models inference performance with [ONNX Runtime](https://github.com/microsoft/onnxruntime). This implementation uses [OpenVINO Benchmark C++ tool](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/benchmark_app) as a reference and stick to its testing methodology, thus performance results comparison of both tools is valid.

## Prerequisites
The tool was tested on Ubuntu 20.04 (64-bit) with default GCC* 9.4.0
1. CMake 3.13 or higher
2. GCC 9.4 or higher

## Build ONNX Runtime
1. Clone repository, checkout to the latest stable release and update submodules:
```
git clone  https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.13.1
git submodule update --init --recursive
```
2. Create `build` directory:
```
mkdir build && cd build
```
3. Configure it with `cmake`:
```
cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -Donnxruntime_BUILD_FOR_NATIVE_MACHINE=ON -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_USE_AUTOML=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_FULL_PROTOBUF=ON -Donnxruntime_USE_OPENMP=ON ../cmake
```
4. Build and install project:
```
cmake --build . --target install --config Release -- -j$(nproc --all)
```


## Build ONNX Runtime Benchmark
To build the tool you need to have an installation of [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [OpenCV](https://github.com/opencv/opencv). Set the following environment variables so that cmake can find them during configuration step:
* `ORT_INSTALL_DIR` pointing to ONNX Runtime install directory.
* `OpenCV_DIR` pointing to OpenCV folder with `OpenCVConfig.cmake`.

1. Clone repository and update submodules:
```
git clone https://github.com/itlab-vision/dl-benchmark
cd dl-benchmark
git submodule update --init --recursive
```
2. Create `build` directory:
```
mkdir build && cd build
```
3. In the created directory run `cmake` command:
```
cmake -DCMAKE_BUILD_TYPE=Release <dl-benchmark>/src/onnxruntime_benchmark
```

4. Run `cmake --build`
```
cmake --build .
```
Application binaries will be placed into `<path_to_build_directory>/<BUILD_TYPE>/bin` directory, where `BUILD_TYPE` whether `Debug` or `Release`.

## Usage
Running the tool  with `-h` option shows the help message:
```
onnxruntime_benchmark
Options:
        [-h]                                         show the help message and exit
        [-help]                                      print help on all arguments
         -m <MODEL FILE>                             path to an .onnx file with a trained model
        [-i <INPUT>]                                 path to an input to process. The input must be an image and/or binaries, a folder of images and/or binaries.
                                                     Ex, "input1:file1 input2:file2 input3:file3" or just path to the file or folder if model has one input
        [-b <NUMBER>]                                batch size value. If not provided, batch size value is determined from the model
        [-shape <[N,C,H,W]>]                         shape for network input.
                                                     Ex., "input1[1,128],input2[1,128],input3[1,128]" or just "[1,3,224,224]"
        [-layout <[NCHW]>]                           layout for network input.
                                                     Ex., "input1[NCHW],input2[NC]" or just "[NCHW]"
        [-mean <R G B>]                              Mean values per channel for input image.
                                                     Applicable only for models with image input.
                                                     Ex.: [123.675,116.28,103.53] or with specifying inputs src[255,255,255]
        [-scale <R G B>]                             Scale values per channel for input image.
                                                     Applicable only for models with image inputs.
                                                     Ex.: [58.395,57.12,57.375] or with specifying inputs src[255,255,255]
        [-nthreads <NUMBER>]                         number of threads to utilize.
        [-nireq <NUMBER>]                            number of inference requests. If not provided, default value is set.
        [-niter <NUMBER>]                            number of iterations. If not provided, default time limit is set.
        [-t <NUMBER>]                                time limit for inference in seconds
        [-save_report]                               save report in JSON format.
        [-report_path <PATH>]                        destination path for report.
```

### Basic usage
To run tool with default options you should provide only path to model file in `.onnx` format
```
./benchmark_app -m model.onnx
```
By default, the application make inference on randomly-generated tensors for 60 seconds. For inference only `CPU` device is available for now.

### Inputs
To pass inputs for model use `-i` option.
```
./onnxruntime_benchmark -m model.onnx -i <path_to_input_data>
```
By default, number of inputs to use determined based on model's batch size. If yout want to make inference on some set, you can specify the number of files to take with `-nireq` option.

If the model has several inputs, files or folders must be specified for each:
```
./onnxruntime_benchmark -m model.onnx -i input1:file1 input2:file2 input3:file3
```

### Shape and layout options
To make inference with dynamic model, you must specify shape for every model's input:
```
./onnxruntime_benchmark -m model.onnx -i input1:file1 input2:file2 input3:file3 -shape input1[NUM1,NUM2],input1[NUM3,NUM4],input1[NUM5,NUM6]
```
Or, if all inputs have the same shape, you could pass just `-shape [NUM1,NUM2]`. The same rules are applied to `-layout` option.

### Report
To save a report with tool configuration and performance results specify `-save_report` flag:
```
./benchmark_app -m model.onnx -save_report -report_path report/report.json
```
if `-report_path` wasn't provided then it will be saved in the current directory under `ort_benchmark.json` name. If specified folder doesn't exist, it will be created.

## Output examples
Below is a sample output of the tool from the terminal:
```
./onnxruntime_benchmark -m /home/ivikhrev/dev/models/public/resnet-50-pytorch/resnet-v1-50.onnx -i ../../../test_data -nireq 3
[Step 1/8] Parsing and validating input arguments
[ INFO ] Parsing input arguments
[ INFO ] Checking input files
[ WARNING ] Too many files to process. The number of files is limited to 20
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000001.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000002.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000003.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000004.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000005.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000006.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000007.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000008.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000009.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000010.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000011.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000012.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000013.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000014.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000015.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000016.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000017.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000018.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000019.JPEG
[ INFO ]        ../../../test_data/ILSVRC2012_val_00000020.JPEG
[Step 2/8] Loading ONNX Runtime
[ INFO ] ONNX Runtime version: 1.12.1
[Step 3/8] Reading model files
[ INFO ] Reading model /home/ivikhrev/dev/models/public/resnet-50-pytorch/resnet-v1-50.onnx
[ INFO ] Read model took 221.03 ms
[ INFO ] Model inputs/outputs info:
[ INFO ] Model inputs:
[ INFO ]        data: FP32 [1,3,224,224]
[ INFO ] Model outputs:
[ INFO ]        prob: FP32 [1,1000]
[ INFO ] Device: CPU
[ INFO ]        Threads number: DEFAULT
[Step 4/8] Configuring input of the model
[ WARNING ] Layout will be detected automatically, as it wasn't provided explicitly.
[ INFO ] Set batch to 1
[Step 5/8] Setting execution parameters
[ INFO ] Default time limit is set: 60 seconds
[Step 6/8] Creating input tensors
[ INFO ] Input config 0
[ INFO ]        data (NCHW FP32 [1,3,224,224])
[ INFO ]                ../../../test_data/ILSVRC2012_val_00000001.JPEG
[ INFO ] Input config 1
[ INFO ]        data (NCHW FP32 [1,3,224,224])
[ INFO ]                ../../../test_data/ILSVRC2012_val_00000002.JPEG
[ INFO ] Input config 2
[ INFO ]        data (NCHW FP32 [1,3,224,224])
[ INFO ]                ../../../test_data/ILSVRC2012_val_00000003.JPEG
[Step 7/8] Measuring model performance (3 inference requests, limits: 60000 ms)
[ INFO ] Warming up inference took 45.46 ms
[Step 8/8] Saving statistics report
[ INFO ] Count: 1862 iterations
[ INFO ] Duration: 60002.58 ms
[ INFO ] Latency:
[ INFO ]        Median   34.41 ms
[ INFO ]        Average: 32.22 ms
[ INFO ]        Min:     19.70 ms
[ INFO ]        Max:     84.00 ms
[ INFO ] Throughput: 29.06 FPS
```


