/*
 * kws_nnom_main.c — Spike accuracy harness with Confusion Matrix
 *
 * INPUT FILE:  test_data.bin
 * Format: int32 n_clips, then batches of
 * [uint8 labels[128]] [int8 audio[128 * 8000]]
 *
 * OUTPUT: stdout
 * RESULT:<true>,<pred>   (one per clip)
 * ACCURACY / TOTAL / CORRECT summary lines + PER CLASS breakdown + CONFUSION MATRIX
 *
 * Build with -DWEIGHTS_H='"your_weights.h"' to select the model.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include WEIGHTS_H

#ifndef NNOM_STATIC_BUF_SIZE
#define NNOM_STATIC_BUF_SIZE (512 * 1024)
#endif
#ifdef NNOM_USING_STATIC_MEMORY
static uint8_t nnom_static_buf[NNOM_STATIC_BUF_SIZE];
#endif

#define NUM_CLASSES       11
#define SAMPLES_PER_CLIP  8000
#define LABEL_BATCH       128

static const char * const class_names[NUM_CLASSES] = {
    "down","go","left","no","off","on","right","stop","up","yes","unknown"
};

int main(void)
{
    FILE   *fp;
    int8_t *buf;
    size_t  sz, seek;
    int     correct = 0, total = 0, n_real = 0;
    
    // Arrays to track per-class metrics
    int class_correct[NUM_CLASSES] = {0};
    int class_total[NUM_CLASSES]   = {0};
    
    // 2D Array for Confusion Matrix: [True_Label][Predicted_Label]
    int conf_matrix[NUM_CLASSES][NUM_CLASSES] = {0};

    fp = fopen("test_data.bin", "rb");
    if (!fp) {
        printf("ERROR: cannot open test_data.bin\n");
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    sz = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = (int8_t *)malloc(sz);
    if (!buf) { printf("ERROR: malloc failed\n"); fclose(fp); return 1; }
    fread(buf, 1, sz, fp);
    fclose(fp);

    { int32_t n; memcpy(&n, buf, 4); n_real = (int)n; }
    seek = 4;

    printf("Samples: %d\n", n_real);
    printf("Classes:");
    for (int c = 0; c < NUM_CLASSES; c++) printf(" %s", class_names[c]);
    printf("\n");

#ifdef NNOM_USING_STATIC_MEMORY
    nnom_set_static_buf(nnom_static_buf, sizeof(nnom_static_buf));
#endif

    nnom_model_t *model = nnom_model_create();
    if (!model) { printf("ERROR: nnom_model_create() failed\n"); free(buf); return 1; }

    model_stat(model);

    printf("RESULTS_START\n");

    while (seek < sz) {
        uint8_t labels[LABEL_BATCH];
        if (seek + LABEL_BATCH > sz) break;
        memcpy(labels, buf + seek, LABEL_BATCH);
        seek += LABEL_BATCH;

        for (int i = 0; i < LABEL_BATCH; i++) {
            if (total >= n_real || seek + SAMPLES_PER_CLIP > sz) goto done;

            memcpy(nnom_input_data, buf + seek, SAMPLES_PER_CLIP);
            seek += SAMPLES_PER_CLIP;

            model_run(model);

            int pred = 0;
            int8_t best = nnom_output_data[0];
            for (int j = 1; j < NUM_CLASSES; j++) {
                if (nnom_output_data[j] > best) { best = nnom_output_data[j]; pred = j; }
            }

            int true_label = (int)labels[i];
            
            // Track per-class, overall accuracy, and confusion matrix
            if (true_label >= 0 && true_label < NUM_CLASSES) {
                class_total[true_label]++;
                conf_matrix[true_label][pred]++; // Update confusion matrix
                if (pred == true_label) { 
                    class_correct[true_label]++;
                    correct++;
                }
            }
            total++;
            //printf("RESULT:%d,%d\n", true_label, pred);
        }
    }

done:
    printf("RESULTS_END\n");
    
    // Print Per-Class Summary
    printf("\n--- PER-CLASS METRICS ---\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (class_total[c] > 0) {
            double class_acc = (double)class_correct[c] / class_total[c] * 100.0;
            printf("%8s: %4d / %4d correct (%6.2f%%)\n", class_names[c], class_correct[c], class_total[c], class_acc);
        } else {
            printf("%8s:    0 /    0 correct (   N/A)\n", class_names[c]);
        }
    }
    printf("-------------------------\n");

    // Print Confusion Matrix
    printf("\n--- CONFUSION MATRIX (Rows: True, Cols: Predicted) ---\n");
    printf("         |");
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Print the first 3 characters of each class name as column headers
        printf(" %3.3s", class_names[c]); 
    }
    printf("\n---------+---------------------------------------------\n");
    for (int r = 0; r < NUM_CLASSES; r++) {
        printf("%8s |", class_names[r]);
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (conf_matrix[r][c] == 0) {
                printf("   ."); // Print dot for zero to make it easier to read
            } else {
                printf(" %3d", conf_matrix[r][c]);
            }
        }
        printf("\n");
    }
    printf("-------------------------------------------------------\n\n");

    // Print Overall Summary
    printf("ACCURACY:%.6f\n", total > 0 ? (double)correct / total : 0.0);
    printf("TOTAL:%d\n",   total);
    printf("CORRECT:%d\n", correct);

    model_delete(model);
    free(buf);
    return 0;
}