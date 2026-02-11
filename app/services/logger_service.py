import threading
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DETECTION = "DETECTION"
    DECISION = "DECISION"


class Logger:
    def __init__(self):
        self.log_mutex = threading.Lock()
        self.log_buffer = {
            LogLevel.ERROR: [],
            LogLevel.WARNING: [],
            LogLevel.INFO: [],
            LogLevel.DETECTION: [],
            LogLevel.DECISION: []
        }
        self.max_buffer_size = 1000

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def level_to_string(level):
        return level.value

    def write_log(self, level, message, context=""):
        with self.log_mutex:
            timestamp = self.timestamp()
            level_str = self.level_to_string(level)
            
            entry = {
                'timestamp': timestamp,
                'level': level_str,
                'message': message,
                'context': context
            }
            
            self.log_buffer[level].append(entry)
            if len(self.log_buffer[level]) > self.max_buffer_size:
                self.log_buffer[level] = self.log_buffer[level][-self.max_buffer_size:]

    def log(self, level, message, context=""):
        if level in [LogLevel.ERROR, LogLevel.WARNING, LogLevel.INFO, LogLevel.DETECTION, LogLevel.DECISION]:
            self.write_log(level, message, context)
        else:
            self.write_log(
                LogLevel.INFO,
                f"{message} [Logged with UNKNOWN level: defaulted to INFO]",
                context
            )

    def get_all_logs(self):
        with self.log_mutex:
            return {level.value: logs[:] for level, logs in self.log_buffer.items()}

    def get_logs_by_level(self, level):
        with self.log_mutex:
            return self.log_buffer[level][:]

    def clear_logs(self):
        with self.log_mutex:
            self.log_buffer = {
                LogLevel.ERROR: [],
                LogLevel.WARNING: [],
                LogLevel.INFO: [],
                LogLevel.DETECTION: [],
                LogLevel.DECISION: []
            }
