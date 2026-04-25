#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdint.h>

/* Configuration */
#define VOCAB_SIZE 6
#define STD_DIM 16
#define LIGHT_DIM 4   // 4 values per word (INT4)

/* Vocabulary */
int get_word_id(char *word);

/* Embedding creation */
float* create_standard_embedding();
uint8_t* create_light_embedding();

/* Evaluation */
void evaluate_embeddings(float *std_emb, uint8_t *light_emb, int id);
void evaluate_embeddings_blas(float *std_emb, uint8_t *light_emb, int id);

#endif