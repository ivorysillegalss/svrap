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
	@echo "[$(shell date '+%Y-%m-%d %H:%M:%S')] START RUN" >> $(RUN_LOG)
	@echo "----------------------------------------" >> $(RUN_LOG)
	@./$(TARGET) >> $(RUN_LOG) 2>&1 || (echo "Program crashed or exited with error! See $(RUN_LOG)" && tail -30 $(RUN_LOG); exit 1)
	@echo "=== Run finished (see $(RUN_LOG)) ==="

# 编译（错误追加到 build.log）
$(TARGET): $(SOURCES)
	@echo "Building $(TARGET) ..."
	@$(CXX) $(CXXFLAGS) *.cpp -o $@ 2>> $(BUILD_LOG) && echo "Build successful" || (echo "Build FAILED! See $(BUILD_LOG)"; tail -20 $(BUILD_LOG); exit 1)

clean:
	@rm -f $(TARGET) *.o
	@echo "Clean done"

# 只看运行日志
log:
	@tail -50 $(RUN_LOG)

# 只看编译错误日志
blog:
	@tail -50 $(BUILD_LOG)