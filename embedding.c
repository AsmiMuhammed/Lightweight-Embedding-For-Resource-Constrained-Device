#include <stdlib.h>
#include <stdint.h>
#include "embedding.h"

/* ---------- Standard Float Embedding ---------- */
float* create_standard_embedding() {

    float *emb = (float*)malloc(
        VOCAB_SIZE * STD_DIM * sizeof(float)
    );

    if (!emb) return NULL;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < STD_DIM; j++) {
            emb[i * STD_DIM + j] =
                (float)((i + j) % 10) / 10.0f;
        }
    }

    return emb;
}


/* ---------- Lightweight INT4 Packed Embedding ---------- */
uint8_t* create_light_embedding() {

    /* Each word has LIGHT_DIM values
       Each byte stores 2 values (4 bits each) */

    int packed_size =
        VOCAB_SIZE * (LIGHT_DIM / 2);

    uint8_t *emb = (uint8_t*)malloc(packed_size);

    if (!emb) return NULL;

    for (int i = 0; i < VOCAB_SIZE; i++) {

        for (int j = 0; j < LIGHT_DIM; j += 2) {

            uint8_t val1 = (i + j + 1) % 16;  // 0–15
            uint8_t val2 = (i + j + 2) % 16;  // 0–15

            uint8_t packed =
                (val1 << 4) | (val2 & 0x0F);

            emb[i * (LIGHT_DIM/2) + (j/2)] = packed;
        }
    }

    return emb;
}