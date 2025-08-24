#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#define TAMANHO_MATRIZ 10


void matriz_simetrica(int tam,float *arr){
    
    for (int i = 0; i < tam; i++){
        for (int j = i; j < tam; j++){
        float aleatorio=(float)rand()/(float)RAND_MAX*10.0;
        arr[i*tam+j]=aleatorio;
        arr[j*tam+i]=aleatorio;
    }
    }
    
}
void matriz_identidade(int tam,float *arr){
    
    for (int i = 0; i < tam; i++){
        for (int j = i; j < tam; j++){
        arr[i*tam+j]=i==j?1.0:0.0;
    }
    }
    
}
void imprimir_matrizes(int tam,float *arr){
    for (int i = 0; i < tam; i++){
        puts("");
        for (int j = 0; j < tam; j++){
        printf("|%0.2f|",arr[i*TAMANHO_MATRIZ+j]);
    }
    }
}
float* alocar_matriz(int tam){
   float *arr=(float*)mkl_malloc(TAMANHO_MATRIZ*TAMANHO_MATRIZ*sizeof(float*),64);
   if (arr==NULL)
    {
        puts("erro ao alocar a matriz");
        free(arr);
        return NULL;
    }
    
   
   return arr;
}
float calcula_traço(int tam, float *arr){
      float trace = 0.0;
    cblas_saxpy(tam, 1.0, arr, tam+1, &trace, 0); 
    return trace;
}
int main(){
    struct timeval temporizadorinicial,temporizadorfinal;
    gettimeofday(&temporizadorinicial,NULL);
    float *matriz_a=alocar_matriz(TAMANHO_MATRIZ),
    *matriz_b=alocar_matriz(TAMANHO_MATRIZ),
    *matriz_c=alocar_matriz(TAMANHO_MATRIZ);
    if (matriz_a==NULL || matriz_b==NULL || matriz_c==NULL){
        puts("erro ao alocar memória"); 
        return 1;
    }
    matriz_simetrica(TAMANHO_MATRIZ,matriz_a);
    matriz_simetrica(TAMANHO_MATRIZ,matriz_b);
    matriz_identidade(TAMANHO_MATRIZ,matriz_c);
    
   
    
   
    puts("\ninício da matriz A:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_a);
    puts("\nfim da matriz A");
    puts("início da matriz B:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_b);
    puts("\nfim da matriz B");
    gettimeofday(&temporizadorinicial,NULL);
    //cblas_ssym é melhor do que cblas_sgemm já que A é simétrica, ele só utiliza uma parte de A para os cálculos
    cblas_ssymm(CblasRowMajor,CblasLeft,CblasUpper,TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0,matriz_a,TAMANHO_MATRIZ,matriz_b,TAMANHO_MATRIZ,1.0,matriz_c,TAMANHO_MATRIZ);
    puts("\ninício da matriz AxB:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_c);
    puts("\nfim da matriz AxB");
    gettimeofday(&temporizadorfinal,NULL);
    printf("\ntempo de execução da multiplicação das matrizes,junto com o print: %ld.%ld segundos",temporizadorfinal.tv_sec-temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec-temporizadorinicial.tv_usec));
    gettimeofday(&temporizadorinicial,NULL);
    cblas_scopy(TAMANHO_MATRIZ*TAMANHO_MATRIZ,matriz_b,1,matriz_c,1);
    cblas_saxpy(TAMANHO_MATRIZ*TAMANHO_MATRIZ,1.0,matriz_a,1,matriz_c,1);
    puts("\ninício da matriz A+B:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_c);
    puts("\nfim da matriz A+B");
    gettimeofday(&temporizadorfinal,NULL);
    printf("\ntempo de execução da soma das matrizes, junto com o print: %ld.%ld segundos",temporizadorfinal.tv_sec-temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec-temporizadorinicial.tv_usec));
    gettimeofday(&temporizadorinicial,NULL);
    printf("\ntraço de A %f",calcula_traço(TAMANHO_MATRIZ,matriz_a));
    gettimeofday(&temporizadorfinal,NULL);
    printf("\ntempo de execução do cálculo do traço de A, junto com o print: %ld.%ld segundos",temporizadorfinal.tv_sec-temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec-temporizadorinicial.tv_usec));
    gettimeofday(&temporizadorinicial,NULL);
    printf("\ntraço de B %f",calcula_traço(TAMANHO_MATRIZ,matriz_b));
    gettimeofday(&temporizadorfinal,NULL);
    printf("\ntempo de execução do cálculo do traço de B, junto com o print: %ld.%ld segundos",temporizadorfinal.tv_sec-temporizadorinicial.tv_sec,labs(temporizadorfinal.tv_usec-temporizadorinicial.tv_usec));
    mkl_somatcopy('R','T',TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0,matriz_a,TAMANHO_MATRIZ,matriz_c,TAMANHO_MATRIZ);
    puts("\ninício da transposta de A:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_c);
    puts("\nfim da transposta de A:");
    mkl_somatcopy('R','T',TAMANHO_MATRIZ,TAMANHO_MATRIZ,1.0,matriz_b,TAMANHO_MATRIZ,matriz_c,TAMANHO_MATRIZ);
    puts("\ninício da transposta de B:");
    imprimir_matrizes(TAMANHO_MATRIZ,matriz_c);
    puts("\nfim da transposta de B:");
    printf("\ntamanho das matrizes: %dx%d",TAMANHO_MATRIZ,TAMANHO_MATRIZ);
    mkl_free(matriz_a);
    mkl_free(matriz_b);
    mkl_free(matriz_c);

    return 0;
}