#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "embedding.h"
#include <openblas/cblas.h>

/* ---------- Custom Matrix Kernel ---------- */
void matmul_kernel(
    float *one_hot,
    float *embedding,
    float *output,
    int V,
    int D
) {
    for (int j = 0; j < D; j++) {
        output[j] = 0.0f;
        for (int i = 0; i < V; i++) {
            output[j] += one_hot[i] *
                         embedding[i * D + j];
        }
    }
}


/* ---------- Kernel Evaluation ---------- */
void evaluate_embeddings(float *std_emb,
                         uint8_t *light_emb,
                         int id) {

    clock_t start, end;

    float one_hot[VOCAB_SIZE] = {0};
    one_hot[id] = 1.0f;

    float kernel_output[STD_DIM];

    /* ---- Standard Kernel ---- */
    start = clock();

    for(int k = 0; k < 100000; k++) {
        matmul_kernel(
            one_hot,
            std_emb,
            kernel_output,
            VOCAB_SIZE,
            STD_DIM
        );
    }

    end = clock();

    printf("\nStandard Embedding (Kernel):\n");
    for (int i = 0; i < STD_DIM; i++) {
        printf("%.2f ", kernel_output[i]);
    }

    printf("\nKernel time: %.6f ms\n",
           ((double)(end - start) /
            CLOCKS_PER_SEC) * 1000);


    /* ---- Lightweight INT4 ---- */
    start = clock();

    for(int k = 0; k < 500000; k++) {

        for (int j = 0; j < LIGHT_DIM; j += 2) {

            uint8_t packed =
                light_emb[id * (LIGHT_DIM/2) + (j/2)];

            uint8_t val1 = packed >> 4;
            uint8_t val2 = packed & 0x0F;

        }

    }

    end = clock();

    printf("\nLightweight Embedding (INT4):\n");

    for (int j = 0; j < LIGHT_DIM; j += 2) {

        uint8_t packed =
            light_emb[id * (LIGHT_DIM/2) + (j/2)];

        uint8_t val1 = packed >> 4;
        uint8_t val2 = packed & 0x0F;

        printf("%u %u ", val1, val2);
    }

    printf("\nLightweight time: %.6f ms\n",
           ((double)(end - start) /
            CLOCKS_PER_SEC) * 1000);


    /* ---- Memory ---- */
    printf("\nMemory Usage:\n");

    printf("Standard: %lu bytes\n",
           VOCAB_SIZE * STD_DIM *
           sizeof(float));

    printf("Lightweight (INT4): %lu bytes\n",
           VOCAB_SIZE *
           (LIGHT_DIM/2) *
           sizeof(uint8_t));
}


/* ---------- BLAS Evaluation ---------- */
void evaluate_embeddings_blas(float *std_emb,
                              uint8_t *light_emb,
                              int id) {

    clock_t start, end;

    float one_hot[VOCAB_SIZE] = {0};
    one_hot[id] = 1.0f;

    float blas_output[STD_DIM];

    start = clock();

    for(int k = 0; k < 50000; k++) {

        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            VOCAB_SIZE,
            STD_DIM,
            1.0f,
            std_emb,
            STD_DIM,
            one_hot,
            1,
            0.0f,
            blas_output,
            1
        );

    }

    end = clock();

    printf("\nStandard Embedding (BLAS):\n");

    for (int i = 0; i < STD_DIM; i++) {
        printf("%.2f ", blas_output[i]);
    }

    printf("\nBLAS time: %.6f ms\n",
           ((double)(end - start) /
            CLOCKS_PER_SEC) * 1000);
}