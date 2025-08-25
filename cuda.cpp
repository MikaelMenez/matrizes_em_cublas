#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#define TAMANHO_MATRIZ 10

// Função para copiar matriz da GPU para CPU para impressão
void copiar_para_cpu(sycl::queue& q, float* gpu_ptr, float* cpu_ptr, int size) {
    q.memcpy(cpu_ptr, gpu_ptr, size * sizeof(float)).wait();
}

// Função para copiar matriz da CPU para GPU
void copiar_para_gpu(sycl::queue& q, float* cpu_ptr, float* gpu_ptr, int size) {
    q.memcpy(gpu_ptr, cpu_ptr, size * sizeof(float)).wait();
}

// Kernel SYCL para gerar matriz simétrica diretamente na GPU
void gera_matriz_simetrica_gpu(sycl::queue& q, float* arr, int tam) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(tam), [=](sycl::id<1> i) {
            
            for (int j = i; j < tam; j++) {
                float aleatorio = rand() / (float)RAND_MAX * 10.0f;
                arr[i * tam + j] = aleatorio;
                if (i != j) {
                    arr[j * tam + i] = aleatorio;
                }
            }
        });
    }).wait();
}

void imprimir_matrizes(int tam, float *arr) {
    for (int i = 0; i < tam; i++) {
        puts("");
        for (int j = 0; j < tam; j++) {
            printf("|%0.2f|", arr[i * TAMANHO_MATRIZ + j]);
        }
    }
}

// Kernel para calcular traço diretamente na GPU
float calcula_traco_gpu(sycl::queue& q, float* arr, int tam) {
    float* trace_result = sycl::malloc_shared<float>(1, q);
    *trace_result = 0.0f;
    
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(tam), [=](sycl::id<1> i) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                            sycl::memory_scope::device> atomic_trace(*trace_result);
            atomic_trace.fetch_add(arr[i * tam + i]);
        });
    }).wait();
    
    float result = *trace_result;
    sycl::free(trace_result, q);
    return result;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);

    struct timeval temporizadorinicial, temporizadorfinal;
    gettimeofday(&temporizadorinicial, NULL);

    // Alocar memória para as matrizes
    float* matriz_a = sycl::malloc_device<float>(TAMANHO_MATRIZ * TAMANHO_MATRIZ, q);
    float* matriz_b = sycl::malloc_device<float>(TAMANHO_MATRIZ * TAMANHO_MATRIZ, q);
    float* matriz_resultante = sycl::malloc_device<float>(TAMANHO_MATRIZ * TAMANHO_MATRIZ, q);

    // Buffers temporários na CPU para impressão
    float* matriz_a_cpu = new float[TAMANHO_MATRIZ * TAMANHO_MATRIZ];
    float* matriz_b_cpu = new float[TAMANHO_MATRIZ * TAMANHO_MATRIZ];
    float* resultado_cpu = new float[TAMANHO_MATRIZ * TAMANHO_MATRIZ];

    // Gerador de matriz simétrica aleatória
    gera_matriz_simetrica_gpu(q, matriz_a, TAMANHO_MATRIZ);
    gera_matriz_simetrica_gpu(q, matriz_b, TAMANHO_MATRIZ);

    // Copiar para CPU apenas para impressão
    copiar_para_cpu(q, matriz_a, matriz_a_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);
    copiar_para_cpu(q, matriz_b, matriz_b_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);

    puts("\ninício da matriz A:");
    imprimir_matrizes(TAMANHO_MATRIZ, matriz_a_cpu);
    puts("\nfim da matriz A");
    puts("início da matriz B:");
    imprimir_matrizes(TAMANHO_MATRIZ, matriz_b_cpu);
    puts("\nfim da matriz B");

    // multiplicação de matrizes na gpu
    gettimeofday(&temporizadorinicial, NULL);
    oneapi::mkl::blas::column_major::symm(q,oneapi::mkl::side::left,oneapi::mkl::uplo::upper,TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0f,matriz_a,TAMANHO_MATRIZ,matriz_b,TAMANHO_MATRIZ,1.0f,matriz_resultante,TAMANHO_MATRIZ).wait();
    gettimeofday(&temporizadorfinal, NULL);

    // Copiar resultado para impressão
    copiar_para_cpu(q, matriz_resultante, resultado_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);

    puts("\ninício da matriz AxB:");
    imprimir_matrizes(TAMANHO_MATRIZ, resultado_cpu);
    puts("\nfim da matriz AxB");
    
    printf("\ntempo de execução da multiplicação das matrizes: %ld.%06ld segundos\n",temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    // Soma de matrizes na gpu
    gettimeofday(&temporizadorinicial, NULL);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(TAMANHO_MATRIZ * TAMANHO_MATRIZ), [=](sycl::id<1> i) {
            matriz_resultante[i] = matriz_a[i] + matriz_b[i];
        });
    }).wait();
    gettimeofday(&temporizadorfinal, NULL);

    copiar_para_cpu(q, matriz_resultante, resultado_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);

    puts("\ninício da matriz A+B:");
    imprimir_matrizes(TAMANHO_MATRIZ, resultado_cpu);
    puts("\nfim da matriz A+B");
    printf("\ntempo de execução da soma das matrizes: %ld.%06ld segundos\n",
           temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,
           labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    // calculo de traço na gpu
    gettimeofday(&temporizadorinicial, NULL);
    float traco_a = calcula_traco_gpu(q, matriz_a, TAMANHO_MATRIZ);
    gettimeofday(&temporizadorfinal, NULL);
    
    printf("\ntraço de A %f", traco_a);
    printf("\ntempo de execução do cálculo do traço de A: %ld.%06ld segundos\n",temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    gettimeofday(&temporizadorinicial, NULL);
    float traco_b = calcula_traco_gpu(q, matriz_b, TAMANHO_MATRIZ);
    gettimeofday(&temporizadorfinal, NULL);
    
    printf("\ntraço de B %f", traco_b);
    printf("\ntempo de execução do cálculo do traço de B: %ld.%06ld segundos\n",temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    // matrizes transpostas
    gettimeofday(&temporizadorinicial, NULL);
    oneapi::mkl::blas::column_major::omatcopy(q,oneapi::mkl::transpose::trans,TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0f,matriz_a,TAMANHO_MATRIZ,matriz_resultante,TAMANHO_MATRIZ).wait();
    gettimeofday(&temporizadorfinal, NULL);
    copiar_para_cpu(q, matriz_resultante, resultado_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);
    puts("\ninício da transposta de A:");
    imprimir_matrizes(TAMANHO_MATRIZ, resultado_cpu);
    puts("\nfim da transposta de A:");
    printf("\ntempo de execução da transposição de A: %ld.%06ld segundos\n",temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    gettimeofday(&temporizadorinicial, NULL);
    oneapi::mkl::blas::column_major::omatcopy(q,oneapi::mkl::transpose::trans,TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0f,matriz_b,TAMANHO_MATRIZ,matriz_resultante,TAMANHO_MATRIZ).wait();
    gettimeofday(&temporizadorfinal, NULL);

    copiar_para_cpu(q, matriz_resultante, resultado_cpu, TAMANHO_MATRIZ * TAMANHO_MATRIZ);
    puts("\ninício da transposta de B:");
    imprimir_matrizes(TAMANHO_MATRIZ, resultado_cpu);
    puts("\nfim da transposta de B:");
    printf("\ntempo de execução da transposição de B: %ld.%06ld segundos\n",temporizadorfinal.tv_sec - temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec - temporizadorinicial.tv_usec));

    printf("\ntamanho das matrizes: %dx%d\n", TAMANHO_MATRIZ, TAMANHO_MATRIZ);

    // Liberar memória
    sycl::free(matriz_a, q);
    sycl::free(matriz_b, q);
    sycl::free(matriz_resultante, q);
    delete[] matriz_a_cpu;
    delete[] matriz_b_cpu;
    delete[] resultado_cpu;

    return 0;
}