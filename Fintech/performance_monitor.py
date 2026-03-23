"""
性能监控模块 (Performance Monitor)

此模块负责对各个 AI 引擎模块的性能（如查询耗时、请求次数、成功/失败状态等）
进行实时追踪、汇总统计及日志持久化存储，以便于系统运维与监控。
主要提供的功能包括：
1. 性能数据的内存统计计算
2. JSON 格式的本地日志文件读写
3. 性能监控装饰器 (Decorator)，以非侵入方式接入核心业务逻辑
"""
import time
import json
import os
from datetime import datetime
from functools import wraps
from collections import defaultdict
import threading

# 定义日志和会话状态文件的存储路径
LOG_FILE = "./logs/performance_log.json"
STATS_FILE = "./logs/session_stats.json"

# 线程锁，确保在多线程环境下并发写入日志文件时的线程安全
_log_lock = threading.Lock()

def ensure_log_directory():
    """
    辅助函数：确保日志所在的目录存在。
    如果目录不存在，则自动创建该目录。
    """
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

class PerformanceTracker:
    """
    性能追踪器类 (Performance Tracker)
    记录系统的每一次查询操作与相关的性能耗时，并进行会话级的汇总统计。
    """
    def __init__(self):
        # 记录本次会话的整体统计信息
        self.session_stats = {
            "session_start": datetime.now().isoformat(),         # 会话开始时间
            "total_queries": 0,                                  # 总查询次数
            "queries_by_module": defaultdict(int),               # 分模块的查询次数
            "total_processing_time": 0.0,                        # 总处理耗时（秒）
            "avg_processing_time": 0.0,                          # 平均每次查询耗时（秒）
            "processing_times_by_module": defaultdict(list),     # 各模块具体查询耗时列表
            "errors": []                                         # 记录查询过程中的错误详情
        }
        ensure_log_directory()
    
    def log_query(self, module_name: str, processing_time: float, 
                  input_length: int, status: str = "success", error_msg: str = None):
        """
        记录一次具体的查询行为。
        
        参数:
        - module_name: 调用的模块名称
        - processing_time: 该次请求处理消耗的时间（单位：秒）
        - input_length: 请求的输入字符长度（用于可能的复杂度分析）
        - status: 处理的结果状态（"success" 或 "error"）
        - error_msg: 具体的错误信息（如果有的话）
        
        返回: 记录生成的字典对象
        """
        with _log_lock:
            # 构建单条日志实体
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": module_name,
                "processing_time_seconds": round(processing_time, 3),
                "input_length": input_length,
                "status": status,
                "error_message": error_msg
            }
            
            # 更新总体统计信息
            self.session_stats["total_queries"] += 1
            self.session_stats["queries_by_module"][module_name] += 1
            self.session_stats["total_processing_time"] += processing_time
            self.session_stats["avg_processing_time"] = (
                self.session_stats["total_processing_time"] / 
                self.session_stats["total_queries"]
            )
            # 添加当前模块耗时至列表以便计算模块单独平均时间
            self.session_stats["processing_times_by_module"][module_name].append(processing_time)
            
            # 若状态为错误，则将错误详情存入错误队列中
            if status == "error":
                self.session_stats["errors"].append({
                    "timestamp": log_entry["timestamp"],
                    "module": module_name,
                    "error": error_msg
                })
            
            # 持久化当前日志实体到本地文件
            self._append_to_log(log_entry)
            
            # 在控制台输出监控指标提示
            print(f"📊 性能指标已记录: {module_name} - {processing_time:.2f}s [{status}]")
            
            return log_entry
    
    def _append_to_log(self, log_entry: dict):
        """
        内部方法：将单条性能日志追加写入至 JSON 日志文件中。
        同时只保留最近的 1000 条数据以避免日志文件过大。
        """
        try:
            ensure_log_directory()
            logs = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = [] # 若文件解析失败，则重置为新列表
            # 将新纪录追加到列表末尾
            logs.append(log_entry)
            
            # 写入本地文件，并在内存/磁盘中仅保留最新的 1000 条记录
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 异常捕获与提示
            print(f"⚠️ 日志写入失败: {e}")
    
    def get_session_summary(self) -> dict:
        """
        获取当前会话的汇总性能统计数据。
        通常用于在前端系统监控面板（如 Streamlit Sidebar）展示。
        """
        summary = {
            "session_start": self.session_stats["session_start"],
            "total_queries": self.session_stats["total_queries"],
            "queries_by_module": dict(self.session_stats["queries_by_module"]),
            "avg_processing_time": round(self.session_stats["avg_processing_time"], 3),
            "total_errors": len(self.session_stats["errors"]),
            "module_stats": {}
        }
        
        # 获取分模块的最小、最大及平均耗时
        for module, times in self.session_stats["processing_times_by_module"].items():
            if times:
                summary["module_stats"][module] = {
                    "count": len(times),
                    "avg_time": round(sum(times) / len(times), 3),
                    "min_time": round(min(times), 3),
                    "max_time": round(max(times), 3)
                }
        
        return summary
    
    def save_session_stats(self):
        """持会话统计信息到本地文件 `session_stats.json` 中。"""
        try:
            ensure_log_directory()
            summary = self.get_session_summary()
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 会话统计保存失败: {e}")

def track_performance(module_name: str):
    """
    性能追踪装饰器 (Decorator)。
    使用该装饰器可以非侵入地监控任何目标函数的运行时间并记录性能指标。
    
    参数:
    - module_name: 需要监控的模块名称常量或标识符。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()    # 记录执行开始时刻
            status = "success"          # 初始状态设为成功
            error_msg = None
            
            try:
                # 执行目标函数
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # 若执行中抛出异常，捕获错误信息及将状态设置为失败
                status = "error"
                error_msg = str(e)
                raise
            finally:
                # 无论成功与否，都要计算处理时间与输入大小
                processing_time = time.time() - start_time
                input_length = len(str(args[1])) if len(args) > 1 else 0
                
                # 若方法参数里传入了 tracker 实例，则自动进行日志记录
                if 'tracker' in kwargs and kwargs['tracker']:
                    kwargs['tracker'].log_query(
                        module_name, processing_time, input_length, status, error_msg
                    )
        return wrapper
    return decorator

# 单例模式 (Singleton): 
# 提供全局唯一的 Tracker 实例，供各个模块及 Streamlit 前端直接调用。
global_tracker = PerformanceTracker()

def get_tracker() -> PerformanceTracker:
    """获取全局唯一性能监控器 (Tracker) 实例。"""
    return global_tracker
