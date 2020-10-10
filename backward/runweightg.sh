#Bert outptu
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 1024 1536 4096 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1024 1536 4096 1
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 1024 1536 4096 10000

echo

#bertIntermediat
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 4096 1536 1024 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 4096 1536 1024 1
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 4096 1536 1024 10000
echo

#bert selfoutput
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 1024 1536 1024 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1024 1536 1024 1
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 1024 1536 1024 10000
echo

#bert value
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 1024 1536 1024 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1024 1536 1024 1 
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 1024 1536 1024 10000
echo

#Bert key
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 1024 1536 1024 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1024 1536 1024 1
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 1024 1536 1024 10000
echo

#bert query
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && ./a.out 1024 1536 1024 10000
nvcc -lcurand -lcublas matmultimingbackprop_weightg.cu && sudo /usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./a.out 1024 1536 1024 1
nvcc -lcurand -lcublas matmultimingbackprop_fp16_weightg.cu && ./a.out 1024 1536 1024 10000
echo
