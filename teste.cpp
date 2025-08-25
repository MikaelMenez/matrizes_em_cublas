#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#define TAMANHO_MATRIZ 10


int main(){
    sycl::queue q(sycl::gpu_selector_v);

        printf("%s",typeid(q).name());

    return 0;
}