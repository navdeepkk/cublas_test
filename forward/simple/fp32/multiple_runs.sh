nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -lcurand -lcublas -expt-relaxed-constexpr matmultiming_use_this.cu

for m in {1024..10240..1024}
 do
  for n in {1024..10240..1024}
    do
      for k in {1024..10240..1024}
        do
          if [ $m -eq $n ] && [ $n -eq $k ];
          then
	  ./a.out $m $n $k 1000
	  fi
	done
    done
  done

