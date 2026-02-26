#!/usr/bin/env python
"""
SVRAP 完整测试执行器
根据TEST_PLAN.md自动执行所有测试模块
"""

import subprocess
import os
import sys
import time
from datetime import datetime

# 确保在scripts目录下运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 测试模块定义
TEST_MODULES = {
    "1_stability": {
        "name": "稳定性测试",
        "script": "run_stability_test.py",
        "description": "验证算法在kro*100数据集上的鲁棒性",
        "estimated_time": "10分钟"
    },
    "2_hyperparam": {
        "name": "超参数敏感性测试", 
        "script": "run_hyperparam_test.py",
        "description": "测试K/Tabu/Div/PR参数影响",
        "estimated_time": "15分钟"
    },
    "3_entropy": {
        "name": "熵权重敏感性测试",
        "script": "run_entropy_test.py", 
        "description": "测试熵机制参数λ的影响",
        "estimated_time": "20分钟"
    },
    "4_constraint": {
        "name": "α参数敏感性测试",
        "script": "run_constraint_test.py",
        "description": "测试不同成本异构度(α=3,5,7,9)",
        "estimated_time": "60分钟"
    },
    "5_final": {
        "name": "最终完整实验",
        "script": "run_final_experiments.py",
        "description": "在所有数据集上运行完整算法",
        "estimated_time": "45分钟"
    }
}

def check_prerequisites():
    """检查前置条件"""
    print("=" * 60)
    print("检查前置条件...")
    
    # 检查exe
    exe_path = "../svrap.exe"
    if not os.path.exists(exe_path):
        print(f"❌ 错误: 找不到 {exe_path}")
        print("   请先编译: cd .. && make")
        return False
    print(f"✅ 找到可执行文件: {exe_path}")
    
    # 检查数据集目录
    dataset_dir = "../formatted_dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 找不到数据集目录 {dataset_dir}")
        return False
    dataset_count = len([f for f in os.listdir(dataset_dir) if f.endswith('.txt')])
    print(f"✅ 找到数据集目录: {dataset_count} 个数据集")
    
    # 检查Python依赖
    try:
        import pandas
        import torch
        print(f"✅ Python依赖已安装 (pandas, torch)")
    except ImportError as e:
        print(f"⚠️ 警告: 缺少Python依赖 - {e}")
        print("   建议安装: pip install pandas torch")
    
    # 检查结果目录
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"✅ 创建结果目录: {results_dir}")
    else:
        print(f"✅ 结果目录已存在: {results_dir}")
    
    print("=" * 60)
    return True

def run_module(module_id, module_info):
    """运行单个测试模块"""
    print(f"\n{'='*60}")
    print(f"🚀 运行模块: {module_info['name']}")
    print(f"   脚本: {module_info['script']}")
    print(f"   描述: {module_info['description']}")
    print(f"   预计耗时: {module_info['estimated_time']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", module_info['script']],
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ 模块 {module_info['name']} 完成 (耗时: {elapsed:.1f}秒)")
            return True
        else:
            print(f"\n❌ 模块 {module_info['name']} 失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 模块 {module_info['name']} 异常: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("      SVRAP 混合式算法 - 完整测试执行器")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    # 显示菜单
    print("\n可用测试模块:")
    print("-" * 60)
    for mid, minfo in TEST_MODULES.items():
        print(f"  [{mid}] {minfo['name']} ({minfo['estimated_time']})")
    print("  [all] 运行所有模块")
    print("  [q] 退出")
    print("-" * 60)
    
    # 获取用户选择
    choice = input("\n请选择要运行的模块 (例如: 1_stability 或 all): ").strip().lower()
    
    if choice == 'q':
        print("退出")
        return
    
    modules_to_run = []
    
    if choice == 'all':
        modules_to_run = list(TEST_MODULES.keys())
    elif choice in TEST_MODULES:
        modules_to_run = [choice]
    else:
        print(f"❌ 无效选择: {choice}")
        return
    
    # 执行选定的模块
    total_start = time.time()
    results = {}
    
    for mid in modules_to_run:
        results[mid] = run_module(mid, TEST_MODULES[mid])
    
    total_elapsed = time.time() - total_start
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试执行汇总")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r)
    fail_count = len(results) - success_count
    
    for mid, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {TEST_MODULES[mid]['name']}: {status}")
    
    print("-" * 60)
    print(f"总计: {success_count} 成功, {fail_count} 失败")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 提示查看结果
    print("\n📊 结果文件保存在 ../results/ 目录:")
    results_dir = "../results"
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith('.csv'):
                print(f"   - {f}")

if __name__ == "__main__":
    main()
