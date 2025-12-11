# 选择当前的操作系统
ifeq ($(OS),Windows_NT)
    RM         := del /Q 2>nul
    EXE_SUFFIX := .exe
    TARGET     := app$(EXE_SUFFIX)
    RUN_CMD    := $(TARGET)
    DATE_CMD   := powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"
else
    RM         := rm -f
    EXE_SUFFIX :=
    TARGET     := app
    RUN_CMD    := ./$(TARGET)
    DATE_CMD   := date '+%Y-%m-%d %H:%M:%S'
endif

# Makefile
CXX      := g++
# 执行用
# CXXFLAGS := -std=c++17 -O2 -w
# 调试用
CXXFLAGS := -std=c++17 -g -Wall -Wextra   # 调试必备
TARGET   := app
BUILD_LOG := build.log
RUN_LOG   := run.log

SOURCES  := $(wildcard *.cpp)

.PHONY: all run clean log

# 默认目标：编译 + 运行
all: clean $(TARGET)
	@echo "=== Build OK, running $(TARGET) ==="
	@echo "========================================" >> $(RUN_LOG)
	@echo "[$$($(DATE_CMD))] START RUN" >> $(RUN_LOG)
	@echo "----------------------------------------" >> $(RUN_LOG)
	@$(RUN_CMD) >> $(RUN_LOG) 2>&1 || ( \
		echo "Program crashed or exited with error! See $(RUN_LOG)" && \
		$(call TAIL,30,$(RUN_LOG)); \
		exit 1 \
	)
	@echo "=== Run finished (see $(RUN_LOG)) ==="

# 编译（错误追加到 build.log）
$(TARGET): $(SOURCES)
	@echo "Building $(TARGET) ..."
	@$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ 2>> $(BUILD_LOG) && echo "Build successful" || ( \
		echo "Build FAILED! See $(BUILD_LOG)" && \
		$(call TAIL,20,$(BUILD_LOG)); \
		exit 1 \
	)

clean:
	@$(RM) $(TARGET) *.o
	@echo "Clean done"

# 只看运行日志
log:
	@$(call TAIL,50,$(RUN_LOG))

# 只看编译错误日志
blog:
	@$(call TAIL,50,$(BUILD_LOG))