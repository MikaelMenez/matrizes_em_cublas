#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TAMANHO_MATRIZ 10
void matriz_simetrica(int tam,float **arr){
    srand(time(NULL));
    for (int i = 0; i < tam; i++){
        for (int j = i; j < tam; j++){
        float aleatorio=(float)rand()/(float)RAND_MAX*10.0;
        arr[i][j]=aleatorio;
        arr[j][i]=aleatorio;
    }
    }
    
}
int main(){
    float **matriz_a,**matriz_b;
    
    
   matriz_a=(float**)malloc(TAMANHO_MATRIZ*sizeof(float*));
   if (matriz_a==NULL)
    {
        puts("erro ao alocar a matriz");
        free(matriz_a);
        return 1;
    }
   for (int i = 0; i < TAMANHO_MATRIZ; i++)
   {
    matriz_a[i]=(float*)malloc(TAMANHO_MATRIZ*sizeof(float));
    if (matriz_a[i]==NULL)
    {
        printf("erro ao alocar a linha %d",i);
        for(int l=0;l<i;l++){
            free(matriz_a[i]);
        }
        free(matriz_a);
        return 1;
    }
    
   }
    
   matriz_simetrica(TAMANHO_MATRIZ,matriz_a);

   
    for (int i = 0; i < TAMANHO_MATRIZ; i++){
        puts("");
        for (int j = 0; j < TAMANHO_MATRIZ; j++){
        printf("|%0.2f|",matriz_a[i][j]);
    }
    }
    
    return 0;
}