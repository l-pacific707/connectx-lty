import logging
import os
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import multiprocessing # Added for Queue

# Sentinel to stop the listener
LOG_QUEUE_SENTINEL = None

def configure_main_logger_handlers(logger_instance, log_file_name, log_level=logging.DEBUG, max_bytes=100*1024*1024, backup_count=10):
    """Configures file and console handlers for a given logger instance.
       This will be used by the listener process.
    """
    cwd = os.getcwd()
    log_dir = os.path.join(cwd, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name)

    # File Handler
    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s'))

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # Keep console less verbose
    ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))

    logger_instance.addHandler(fh)
    logger_instance.addHandler(ch)
    logger_instance.setLevel(log_level)


def listener_process_configure(log_queue: multiprocessing.Queue , log_configs):
    """
    Configures and runs the logging listener.
    Reads log records from the queue and dispatches them to the appropriate handlers.

    Args:
        log_queue (multiprocessing.Queue): The queue to listen on.
        log_configs (dict): A dictionary where keys are logger names and
                            values are their log file names.
                            Example: {"MCTS": "MCTS.log", "train": "Play_and_Train.log"}
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s')
    root_logger = logging.getLogger() # Get the root logger
    
    handlers = []
    # Configure handlers for specific loggers based on log_configs
    # These specific loggers will write to their respective files
    for logger_name, log_file in log_configs.items():
        logger_instance = logging.getLogger(logger_name)
        # Important: Prevent log propagation to avoid duplicate messages if root also has handlers
        logger_instance.propagate = False 
        
        cwd = os.getcwd()
        log_dir = os.path.join(cwd, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        
        fh = RotatingFileHandler(log_path, maxBytes=100*1024*1024, backupCount=10)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s'))
        logger_instance.addHandler(fh)
        logger_instance.setLevel(logging.DEBUG) # Set level for specific logger
        handlers.append(fh) # Keep track of handlers for the listener

    # # If you also want a general console handler for all logs coming through the queue:
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s'))
    # console_handler.setLevel(logging.INFO) # Or DEBUG if you want everything on console via listener
    
    # The listener will use the handlers attached to the loggers it knows about,
    # or if no specific logger matches, it can fall back to root logger's handlers.
    # For simplicity, we'll let specific loggers handle their files.
    # The `QueueListener` will dispatch records to loggers based on record.name.

    # It's often simpler to attach one or more "master" handlers to the listener
    # that process ALL records from the queue, rather than trying to replicate
    # per-logger file routing within the listener itself if it becomes complex.
    # However, the default QueueListener behavior dispatches to the named logger.

    # Setup a listener that uses the handlers configured on the respective loggers.
    # This relies on the listener correctly dispatching to loggers that have file handlers.
    # Ensure loggers are configured *before* listener starts.
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True) # Pass all file handlers
    # if console_handler: # Optionally add a console handler to the listener too
    #     # This console handler on the listener will log records from *all* workers
    #     # that go through the queue, regardless of their original logger name,
    #     # if not handled by a more specific logger's handler.
    #     # To avoid this, ensure specific loggers are well-defined.
    #     # For now, let's only use file handlers defined by log_configs.
    #     pass


    listener.start()
    print(f"Log listener started in process {os.getpid()}") # For debugging listener start

    # Wait for the sentinel
    while True:
        try:
            record = log_queue.get()
            if record is LOG_QUEUE_SENTINEL:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record) # Manually handle if listener isn't doing it as expected
        except Exception:
            import sys, traceback
            print(f"Log listener error: {traceback.format_exc()}", file=sys.stderr) # For debugging

    listener.stop()
    print(f"Log listener stopped in process {os.getpid()}") # For debugging listener stop

def worker_configurer(log_queue, worker_log_level=logging.DEBUG):
    """
    Configures logging for a worker process to send logs to the queue.
    This should be called in the worker's initialization.
    """
    # Configure all loggers obtained via logging.getLogger() to use the QueueHandler
    # You might want to be more selective if different modules need different levels.
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to avoid conflicts or duplicate logging
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            
    # Add the QueueHandler
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(worker_log_level) # Set the level for logs sent to the queue

    # Specific loggers can also be configured if needed, but root handles all by default.
    # For example, to ensure "MCTS" logger also uses the queue:
    mcts_logger = logging.getLogger("MCTS")
    if mcts_logger.hasHandlers(): # Clear existing specific handlers if any
        for handler in mcts_logger.handlers[:]:
            mcts_logger.removeHandler(handler)
    if not any(isinstance(h, QueueHandler) for h in mcts_logger.handlers): # Add if not already added via root
        mcts_logger.addHandler(queue_handler) # It will use the root's queue_handler if propagate=True
    mcts_logger.setLevel(worker_log_level)
    mcts_logger.propagate = True # Ensure it propagates to root where QueueHandler is

    # Do the same for other key loggers if they are configured with specific handlers elsewhere
    train_logger = logging.getLogger("train.py") # Or "AlphaZeroTraining" if that's the name used
    if train_logger.hasHandlers():
        for handler in train_logger.handlers[:]:
            train_logger.removeHandler(handler)
    if not any(isinstance(h, QueueHandler) for h in train_logger.handlers):
        train_logger.addHandler(queue_handler)
    train_logger.setLevel(worker_log_level)
    train_logger.propagate = True
    
    # Configure ConnectXNN logger
    connectxnn_logger = logging.getLogger("ConnectXNN")
    if connectxnn_logger.hasHandlers():
        for handler in connectxnn_logger.handlers[:]:
            connectxnn_logger.removeHandler(handler)
    if not any(isinstance(h, QueueHandler) for h in connectxnn_logger.handlers):
        connectxnn_logger.addHandler(queue_handler)
    connectxnn_logger.setLevel(worker_log_level)
    connectxnn_logger.propagate = True



# The original get_logger might not be directly used by workers anymore
# if worker_configurer sets up the root logger or specific known loggers.
# However, if modules call get_logger dynamically, they should get the
# queue-configured logger.
# For simplicity, the worker_configurer now sets up the root logger to use the queue.
# Any logger created via logging.getLogger(name) will inherit this if propagate=True.

def get_logger(name="DefaultLogger", log_file="Default.log", use_queue=None):
    """
    Gets a logger instance. If use_queue is provided, it's assumed to be called
    from a worker that should already be configured by worker_configurer.
    If use_queue is None, it sets up standard file/console handlers (now mainly for listener or single-process use).
    """
    logger_instance = logging.getLogger(name)

    if use_queue: # Called from a worker, assume worker_configurer has set up QueueHandler on root
        logger_instance.setLevel(logging.DEBUG) # Or desired level for this specific logger
        # Ensure it propagates to the root logger which has the QueueHandler
        logger_instance.propagate = True 
        # Remove any non-queue handlers that might have been added by mistake
        for handler in logger_instance.handlers[:]:
            if not isinstance(handler, QueueHandler):
                logger_instance.removeHandler(handler)
        return logger_instance

    # This part is for non-worker setup (e.g., initial main process logging before pool, or the listener itself)
    # However, with the listener pattern, the listener process will handle its own file setup.
    # The main process might only need a console logger before the listener starts.
    if not logger_instance.hasHandlers():
        cwd = os.getcwd()
        log_dir = os.path.join(cwd, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        # File Handler (for non-queued logging)
        fh = RotatingFileHandler(log_path, maxBytes=100*1024*1024, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))

        logger_instance.addHandler(fh)
        logger_instance.addHandler(ch)
        logger_instance.setLevel(logging.DEBUG)

    return logger_instance