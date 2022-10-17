#!/bin/bash

omz_downloader --output_dir working_dir_smoke --cache_dir cache_dir_smoke      --name=alexnet,resnet-50-pytorch,resnet-50-tf
omz_converter  --output_dir working_dir_smoke --download_dir working_dir_smoke --name=alexnet,resnet-50-pytorch,resnet-50-tf

python3 ../../src/benchmark/inference_benchmark.py -r results.csv --executor_type host_machine -c ./smoke_config.xml

success_tests=$(grep -o 'Success' results.csv | wc -l)
if [ $success_tests -ne 7 ]; then
    echo "There are should be 7 tests and all the tests should be passed"
    exit 1
fi
exit 0
