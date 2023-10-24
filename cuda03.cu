#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *ReadFile(const char *filename)
{
   char *buffer = NULL;
   int string_size, read_size;
   FILE *handler = fopen(filename, "r");

   if (handler)
   {
       fseek(handler, 0, SEEK_END);
       string_size = ftell(handler);
       rewind(handler);
       buffer = (char*)malloc(sizeof(char) * (string_size + 1) );
       read_size = fread(buffer, sizeof(char), string_size, handler);
       buffer[string_size] = '\0';

       if (string_size != read_size)
       {
           free(buffer);
           buffer = NULL;
       }

       fclose(handler);
    }

    return buffer;
}

__global__ 
void parallel_Histogram(char *data, long length, int *histo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int section_size = (length-1) / (blockDim.x * gridDim.x) + 1;
    int start = i*section_size;

    for (int k = 0; k < section_size; k++) 
    {
        if (start+k < length) 
        {
            int pos = data[start+k] - 'a';
            if (pos >= 0 && pos < 26) atomicAdd(&(histo[pos/4]), 1);
        }
    }
}


void sequential_Histogram(char *data, long length, int *histo) 
{
    for (int i = 0; i < length; i++) 
    {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) 
        {
            if((pos/4) >= 0 && (pos/4) <= 5)
                histo[pos/4]++;
        }
    }
}

int main(int argc, char *argv[])
{
    if(argc < 3) return -1;
    int t = atoi(argv[1]);

    int *histo = (int *)malloc(6 * sizeof(int));
    char *data = ReadFile(argv[2]);
    int length = strlen(data);

    char *d_data;
    int *d_histo;

    printf("Arquivo   : %s \n", argv[2]);
    printf("Tamanho   : %d caracteres \n\n", length);

    if(t == 0) 
    {
        //SEQUENCIAL
        printf("Histograma - Sequencial >> | a-d | e-h | i-l | m-p | q-t | u-z | \n");
        sequential_Histogram(data, length, histo);
    }
    else 
    {
        //CUDA
        printf("Histograma - CUDA >> | a-d | e-h | i-l | m-p | q-t | u-z | \n");
        cudaMalloc((void**)&d_data, sizeof(char) * length);
        cudaMalloc((void**)&d_histo, sizeof(int) * 6);

        cudaMemcpy(d_data, data, sizeof(char) * length, cudaMemcpyHostToDevice);
        cudaMemcpy(d_histo, histo, sizeof(int) * 6, cudaMemcpyHostToDevice);

        parallel_Histogram<<<(length+256)/256, 256>>>(d_data, length, d_histo);
        cudaMemcpy(histo, d_histo, sizeof(int) * 6, cudaMemcpyDeviceToHost);
    }

    printf("Quantidade: ");
    for (int i=0; i<6; i++) printf("| %d ", histo[i]);
    printf("| \n");

    printf("Percentual: ");
    for (int i=0; i<6; i++) printf("| %.1f ", (((float)histo[i]/(float)length)*100));  
    printf("| \n");

    return 0;
}