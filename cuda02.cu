#include <stdlib.h>
#include <stdio.h>

void mult_matrix(float *m_a, float*m_b, float *m_c, int n)
{
    for (int i=0; i<n; i++)
        for ( int j=0; j<n; j++) 
        {
            m_c[(i*n)+j] = 0.0;
            for ( int k=0; k<n; k++)
                m_c[(i*n)+j] = m_c[(i*n)+j] + m_a[(i*n)+k] * m_b[(k*n)+j];
        }
}

__global__ 
void mult_matrix_kernel(float *m_a, float *m_b, float *m_c, int n) {
    
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;

    if ((i < n) && (j < n)) {
        float c = 0;
        for (int k = 0; k < n; ++k) {
            c += m_a[(i*n)+k] * m_b[(k*n)+j];
        }
    
        m_c[(i*n)+j] = c;
    }
}
    
void print_matrix(float *m, int n, char l){

    if (n > 10) return;

    printf("Matrix: %c \n", l);

    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            printf("%f \n", m[(i*n)+j]);

    printf("\n\n");
}


int main(int argc, char *argv[]) {

    int n, t;
    float *m_a, *m_b, *m_c; //matrizes "linearizadas"
    float *d_m_a, *d_m_b, *d_m_c;

    if(argc < 3) return -1;
    n = atoi(argv[1]);
    t = atoi(argv[2]);

    m_a = (float*)malloc(n * n * sizeof(float));
    m_b = (float*)malloc(n * n * sizeof(float));
    m_c = (float*)malloc(n * n * sizeof(float));

    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
        {
            m_a[(i*n)+j] = i+1;
            m_b[(i*n)+j] = (i+1) + (j+1);
            m_c[(i*n)+j] = 0.0;
        }

    print_matrix(m_a, n, 'A');
    print_matrix(m_b, n, 'B');

    if(t == 0) 
    {
        mult_matrix(m_a, m_b, m_c, n);
    }
    else 
    {
        cudaMalloc((void**)&d_m_a, sizeof(float) * n * n);
        cudaMalloc((void**)&d_m_b, sizeof(float) * n * n);
        cudaMalloc((void**)&d_m_c, sizeof(float) * n * n);
        
        cudaMemcpy(d_m_a, m_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_m_b, m_b, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(((float)n) / blockDim.x), ceil(((float)n) / blockDim.y));

        mult_matrix_kernel<<<gridDim, blockDim>>>(d_m_a, d_m_b, d_m_c, n);
        cudaMemcpy(m_c, d_m_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    }
    
    print_matrix(m_c, n, 'C');

    return 0;
}