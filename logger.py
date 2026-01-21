import threading
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DETECTION = "DETECTION"
    DECISION = "DECISION"


class Logger:
    """
    In-memory logger for real-time log viewing.
    Thread-safe logging with support for context information.
    """
    
    def __init__(self):
        """Initialize the logger with in-memory buffer"""
        self.log_mutex = threading.Lock()
        
        # In-memory log storage for real-time viewing
        self.log_buffer = {
            LogLevel.ERROR: [],
            LogLevel.WARNING: [],
            LogLevel.INFO: [],
            LogLevel.DETECTION: [],
            LogLevel.DECISION: []
        }
        self.max_buffer_size = 1000  # Keep last 1000 logs per level
    
    @staticmethod
    def timestamp() -> str:
        """Return current timestamp in format YYYY-MM-DD HH:MM:SS"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def level_to_string(level: LogLevel) -> str:
        """Convert LogLevel enum to string"""
        return level.value
    
    def write_log(self, level: LogLevel, message: str, context: str = ""):
        """Write a log entry to memory buffer"""
        with self.log_mutex:
            timestamp = self.timestamp()
            level_str = self.level_to_string(level)
            
            entry = {
                'timestamp': timestamp,
                'level': level_str,
                'message': message,
                'context': context
            }
            
            # Store in memory buffer
            self.log_buffer[level].append(entry)
            # Keep only the last max_buffer_size entries
            if len(self.log_buffer[level]) > self.max_buffer_size:
                self.log_buffer[level] = self.log_buffer[level][-self.max_buffer_size:]
    
    def log(self, level: LogLevel, message: str, context: str = ""):
        """Main logging method - routes to appropriate log level"""
        if level == LogLevel.ERROR:
            self.write_log(level, message, context)
        elif level == LogLevel.WARNING:
            self.write_log(level, message, context)
        elif level == LogLevel.INFO:
            self.write_log(level, message, context)
        elif level == LogLevel.DETECTION:
            self.write_log(level, message, context)
        elif level == LogLevel.DECISION:
            self.write_log(level, message, context)
        else:
            # Unknown level defaults to INFO
            self.write_log(
                LogLevel.INFO,
                f"{message} [Logged with UNKNOWN level: defaulted to INFO]",
                context
            )
    
    def get_all_logs(self) -> dict:
        """Return all buffered logs organized by level"""
        with self.log_mutex:
            return {level.value: logs[:] for level, logs in self.log_buffer.items()}
    
    def get_logs_by_level(self, level: LogLevel) -> list:
        """Get logs for a specific level"""
        with self.log_mutex:
            return self.log_buffer[level][:]
