﻿# run_experiments.ps1
# SVRAP 实验的批量执行脚本
# 执行步骤: 针对每个数据集 -> 运行 Python 策略网络生成 attention_probs.csv -> 运行带参的 C++ 禁忌搜索单独读取该 CSV 进行求解

$ErrorActionPreference = "Continue"

# ================= 配置区 =================
$DatasetDir = "formatted_dataset"
$LogFile = "experiment_results.log"
$Alpha = 7.0
$PythonScript = "svrap_solver.py"
$CppExe = ".\svrap.exe"

# 确保尽可能使用当前环境的 python
$PythonExe = "C:\Users\chenz\miniconda3\envs\altr-py310\python.exe"
Write-Host "使用的 Python 解释器: $PythonExe"
# =============================================

# 检查必要的文件和目录
if (-not (Test-Path $DatasetDir)) {
    Write-Error "错误: 找不到数据集目录 '$DatasetDir'。"
    exit 1
}
if (-not (Test-Path $PythonScript)) {
    Write-Error "错误: 找不到 Python 脚本 '$PythonScript'。"
    exit 1
}
if (-not (Test-Path $CppExe)) {
    Write-Error "错误: 找不到 C++ 可执行文件 '$CppExe'。请先编译 (make)。"
    exit 1
}

# 初始化日志文件
$StartTime = Get-Date
"=== SVRAP 批量实验开始于 $StartTime ===" | Out-File -FilePath $LogFile -Encoding utf8

# 获取所有 .txt 数据集文件
$Datasets = Get-ChildItem -Path $DatasetDir -Filter "*.txt"

Write-Host "找到 $( $Datasets.Count ) 个数据集。开始处理..."
Write-Host "日志将写入到: $LogFile"

foreach ($File in $Datasets) {
    $DatasetPath = $File.FullName
    $DatasetName = $File.Name
    
    Write-Host "--------------------------------------------------"
    Write-Host "正在处理: $DatasetName"
    
    # 写入分隔符到日志
    "`n`n==================================================" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "DATASET: $DatasetName" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "TIME: $(Get-Date)" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "==================================================" | Out-File -FilePath $LogFile -Append -Encoding utf8

    # 1. 运行 Python (训练/推理 + 覆盖生成当前数据集的初始解 CSV)
    Write-Host "  [1/2] 正在运行 Python 策略网络..." -NoNewline
    
    "COMMAND: $PythonExe $PythonScript --dataset `"$DatasetPath`" --train" | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    # 运行命令，捕获输出，检查状态，然后写入日志
    $pyOutput = & $PythonExe $PythonScript --dataset "$DatasetPath" --train 2>&1
    $pyStatus = $LASTEXITCODE
    
    $pyOutput | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    if ($pyStatus -eq 0) {
        Write-Host " 完成" -ForegroundColor Green
    } else {
        Write-Host " 失败 (退出代码: $pyStatus)" -ForegroundColor Red
        "ERROR: Python script failed with exit code $pyStatus" | Out-File -FilePath $LogFile -Append -Encoding utf8
        continue # 如果 Python 失败，跳过 C++ 步骤
    }

    # 2. 运行 C++ (作为单次任务调用，带有当前数据集参数，读取刚生成的初始解进行搜索)
    Write-Host "  [2/2] 正在运行 C++ 禁忌搜索..." -NoNewline
    
    "COMMAND: $CppExe $Alpha `"$DatasetPath`"" | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    $LASTEXITCODE = 0
    $cppOutput = & $CppExe $Alpha "$DatasetPath" 2>&1
    $cppStatus = $LASTEXITCODE
    
    $cppOutput | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    if ($cppStatus -eq 0) {
        Write-Host " 完成" -ForegroundColor Green
    } else {
        Write-Host " 失败 (退出代码: $cppStatus)" -ForegroundColor Red
        "ERROR: C++ executable failed with exit code $cppStatus" | Out-File -FilePath $LogFile -Append -Encoding utf8
    }
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "--------------------------------------------------"
Write-Host "所有实验完成。"
Write-Host "总耗时: $Duration"
Write-Host "详细日志: $LogFile"
