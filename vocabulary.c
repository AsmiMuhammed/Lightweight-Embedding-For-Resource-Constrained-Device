#include <string.h>
#include "embedding.h"

char *vocab[VOCAB_SIZE] = {
    "sensor",
    "alert",
    "device",
    "edge",
    "power",
    "network"
};

int get_word_id(char *word) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(vocab[i], word) == 0)
            return i;
    }
    return -1;
}