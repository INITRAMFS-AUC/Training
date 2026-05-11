CC    = /opt/riscv/gcc15/bin/riscv32-unknown-elf-gcc
SPIKE = /opt/riscv/bin/spike
PK    = /opt/riscv/riscv32-unknown-elf/bin/pk

ARCH  = rv32imac_zicsr_zifencei
ABI   = ilp32

NNOM_DIR  = /workspace/Desktop/kws-spike-validate/extern/nnom
NNOM_INC  = -I$(NNOM_DIR)/inc -I$(NNOM_DIR)/port
NNOM_SRCS = $(wildcard $(NNOM_DIR)/src/core/*.c) \
            $(wildcard $(NNOM_DIR)/src/layers/*.c) \
            $(wildcard $(NNOM_DIR)/src/backends/*.c)

WEIGHTS_DIR = /workspace/Desktop/Main/FinalModel

# Model header to test — override at the command line:
#   make run MODEL=weights_qat.h
MODEL      ?= weights_no_qat.h
MODEL_STEM  = $(basename $(notdir $(MODEL)))

# Resolve to absolute path for Make dependency tracking
ifeq ($(filter /%,$(MODEL)),)
MODEL_ABS = $(WEIGHTS_DIR)/$(MODEL)
else
MODEL_ABS = $(MODEL)
endif

CFLAGS = -march=$(ARCH) -mabi=$(ABI) \
         -O2 -std=c99 \
         -DNNOM_USING_STATIC_MEMORY \
         -DNNOM_BLOCK_NUM=16 \
         -DWEIGHTS_H='"$(MODEL)"' \
         $(NNOM_INC) \
         -I$(WEIGHTS_DIR)

LDFLAGS = -lm

SPIKE_FLAGS = --isa=$(ARCH) -m256

TEST_DATA    = $(WEIGHTS_DIR)/test_data/peak_pi_test_data_int8.bin
PI_TEST_DATA = $(WEIGHTS_DIR)/pi_data/test_data/pi_test_data_int8_peak_norm.bin
BUILD        = build

BIN = $(BUILD)/kws_$(MODEL_STEM)

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build run run_pi run_google_to_pi run_pi_raw run_pi_raw_mix clean

$(BIN): kws_nnom_main.c $(MODEL_ABS) $(NNOM_SRCS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) kws_nnom_main.c $(NNOM_SRCS) -o $@ $(LDFLAGS)

build: $(BIN)

run: $(BIN)
	@echo "=== Running $(MODEL) on Spike ==="
	@ln -sf $(TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_$(MODEL_STEM) \
	    | tee kws_$(MODEL_STEM).log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_$(MODEL_STEM).log || true

run_pi: $(BIN)
	@echo "=== Running $(MODEL) on Pi test data ==="
	@ln -sf $(PI_TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_$(MODEL_STEM) \
	    | tee kws_$(MODEL_STEM)_pi.log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_$(MODEL_STEM)_pi.log || true

GOOGLE_TO_PI_TEST_DATA = $(WEIGHTS_DIR)/Google_to_pi/test_data/test_data_pi_amp.bin
PI_RAW_TEST_DATA       = $(WEIGHTS_DIR)/pi_data/test_data/pi_test_data_int8_batched.bin

run_google_to_pi: $(BIN)
	@echo "=== Running $(MODEL) on Pi-amplitude Google test data ==="
	@ln -sf $(GOOGLE_TO_PI_TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_$(MODEL_STEM) \
	    | tee kws_$(MODEL_STEM)_google_to_pi.log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_$(MODEL_STEM)_google_to_pi.log || true

run_pi_raw: $(BIN)
	@echo "=== Running $(MODEL) on raw Pi test data (no peak normalization) ==="
	@ln -sf $(PI_RAW_TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_$(MODEL_STEM) \
	    | tee kws_$(MODEL_STEM)_pi_raw.log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_$(MODEL_STEM)_pi_raw.log || true

clean:
	rm -rf $(BUILD)

# ── left/right 3-class model ─────────────────────────────────────────────────
LR_MODEL     ?= kws_lr/weights_lr.h
LR_STEM       = $(basename $(notdir $(LR_MODEL)))
LR_TEST_DATA  = $(WEIGHTS_DIR)/kws_lr/test_data_lr.bin
BIN_LR        = $(BUILD)/kws_lr_$(LR_STEM)

LR_CFLAGS = -march=$(ARCH) -mabi=$(ABI) \
            -O2 -std=c99 \
            -DNNOM_USING_STATIC_MEMORY \
            -DNNOM_BLOCK_NUM=16 \
            -DWEIGHTS_H='"$(LR_MODEL)"' \
            $(NNOM_INC) \
            -I$(WEIGHTS_DIR)

$(BIN_LR): kws_lr/kws_lr_main.c $(WEIGHTS_DIR)/$(LR_MODEL) $(NNOM_SRCS)
	@mkdir -p $(BUILD)
	$(CC) $(LR_CFLAGS) kws_lr/kws_lr_main.c $(NNOM_SRCS) -o $@ $(LDFLAGS)

LR_PI_TEST_DATA = $(WEIGHTS_DIR)/kws_lr/pi_test_data_lr.bin

.PHONY: run_lr run_lr_pi
run_lr: $(BIN_LR)
	@echo "=== Running $(LR_MODEL) on GSCD ==="
	@ln -sf $(LR_TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_lr_$(LR_STEM) \
	    | tee kws_lr_$(LR_STEM).log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_lr_$(LR_STEM).log || true

run_lr_pi: $(BIN_LR)
	@echo "=== Running $(LR_MODEL) on Pi data ==="
	@ln -sf $(LR_PI_TEST_DATA) $(BUILD)/test_data.bin
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) kws_lr_$(LR_STEM) \
	    | tee kws_lr_$(LR_STEM)_pi.log
	@echo ""
	@grep -E "ACCURACY:|TOTAL:|CORRECT:" $(BUILD)/kws_lr_$(LR_STEM)_pi.log || true
