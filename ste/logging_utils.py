import inspect
import logging
import os
from datetime import datetime, timezone


LOG_DIRECTORY = os.path.join(os.path.dirname(__file__), "../logs")


def get_main_module_name():
    import __main__
    
    try:
        main_file_path = __main__.__file__
    except AttributeError:
        return None
    
    main_file_name = os.path.basename(main_file_path)
    return os.path.splitext(main_file_name)[0]
    


def get_log_file_path():
    timestamp = datetime.now().astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")

    main_module_name = get_main_module_name()

    if main_module_name is None:
        log_file_name = f"{timestamp}.log"
    else:
        log_file_name = f"{main_module_name}_{timestamp}.log"
    
    return os.path.join(LOG_DIRECTORY, log_file_name)


def configure_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter.default_time_format = "%Y-%m-%dT%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    file_handler = logging.FileHandler(get_log_file_path())
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_module_logger():
    name = inspect.stack()[1].frame.f_locals["__name__"]
    if name == "__main__":
        import __main__
        name = __main__.__spec__.name
    return logging.getLogger(name)
