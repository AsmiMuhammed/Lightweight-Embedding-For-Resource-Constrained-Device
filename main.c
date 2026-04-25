#include <stdio.h>
#include <stdlib.h>
#include "embedding.h"

int main() {

    char word[20];

    float *std_emb =
        create_standard_embedding();

    uint8_t *light_emb =
        create_light_embedding();

    if (!std_emb || !light_emb) {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("Enter word: ");
    scanf("%s", word);

    int id = get_word_id(word);

    if (id == -1) {
        printf("Word not found\n");
    } else {
        evaluate_embeddings(std_emb,
                            light_emb,
                            id);

        evaluate_embeddings_blas(std_emb,
                                 light_emb,
                                 id);
    }

    free(std_emb);
    free(light_emb);

    return 0;
}