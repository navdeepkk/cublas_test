#Bert outptut
nvcc -lcurand -lcublas matmultiming.cu && ./a.out 1536 1024 1024 10000
nvcc -lcurand -lcublas matmultiming.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1536 1024 1024 1
nvcc -lcurand -lcublas matmultiming_fp16.cu && ./a.out 1536 1024 1024 10000

echo

nvcc -lcurand -lcublas matmultiming.cu && ./a.out 1536 1024 4096 10000
nvcc -lcurand -lcublas matmultiming.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1536 1024 4096 1
nvcc -lcurand -lcublas matmultiming_fp16.cu && ./a.out 1536 1024 4096 10000

echo

nvcc -lcurand -lcublas matmultiming.cu && ./a.out 1536 4096 1024 10000
nvcc -lcurand -lcublas matmultiming.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1536 1024 1024 1
nvcc -lcurand -lcublas matmultiming_fp16.cu && ./a.out 1536 4096 1024 10000

echo
