nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -lcurand -lcublas -expt-relaxed-constexpr matmultiming_copy.cu

for m in {10240..20480..1024}
 do
  for n in {10240..20480..1024}
    do
      for k in {10240..20480..1024}
        do
          if [ $m -eq $n ] && [ $n -eq $k ];
          then
	  ./a.out $m $n $((4 * k)) 10
	  fi
	done
    done
  done

