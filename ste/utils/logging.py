from contextlib import nullcontext
import dataclasses
import functools
import inspect
import logging
import os
import threading
import sys
from datetime import datetime, timezone
from types import GeneratorType
from typing import Callable


LOG_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../logs")


_TLS = threading.local()


@dataclasses.dataclass
class SetBreadcrumbsContext:
    new_breadcrumbs: tuple[str, ...]
    old_breadcrumbs: tuple[str, ...] = None

    def __enter__(self):
        self.old_breadcrumbs = _TLS.breadcrumbs
        _TLS.breadcrumbs = self.new_breadcrumbs
    
    def __exit__(self, *args):
        _TLS.breadcrumbs = self.old_breadcrumbs


@dataclasses.dataclass
class PushBreadcrumbContext:
    new_breadcrumb: str
    old_breadcrumbs: tuple[str, ...] = None

    def __enter__(self):
        self.old_breadcrumbs = _TLS.breadcrumbs
        _TLS.breadcrumbs = _TLS.breadcrumbs + (self.new_breadcrumb,)
    
    def __exit__(self, *args):
        _TLS.breadcrumbs = self.old_breadcrumbs


breadcrumb = PushBreadcrumbContext
breadcrumbs = SetBreadcrumbsContext


def get_breadcrumbs():
    return _TLS.breadcrumbs


@dataclasses.dataclass(frozen=True)
class LogEntryAndExitContext:
    message_level: int
    entry_text: str | None
    exit_text: str | None
    logger: logging.Logger

    def __enter__(self):
        if self.entry_text is not None:
            self.logger.log(self.message_level, self.entry_text)
    
    def __exit__(self, *args):
        if self.exit_text is not None:
            self.logger.log(self.message_level, self.exit_text)


@dataclasses.dataclass(frozen=True)
class _LoggedGenerator:
    generator: GeneratorType
    breadcrumbs: tuple[str, ...]

    def __iter__(self):
        return self

    def __next__(self):
        with SetBreadcrumbsContext(self.breadcrumbs):
            return self.generator.__next__()
    
    def send(self, value):
        with SetBreadcrumbsContext(self.breadcrumbs):
            return self.generator.send(value)
        
    def throw(self, *args, **kwargs):
        with SetBreadcrumbsContext(self.breadcrumbs):
            return self.generator.throw(*args, **kwargs)
        
    def close(self):
        return self.generator.close()


@dataclasses.dataclass
class _LoggedFunction[**P, R]:
    function: Callable[P, R]
    breadcrumb_text: str | None
    entry_text: str | None
    exit_text: str | None
    message_level: int | None
    logger: logging.Logger
    name_for_inference: dataclasses.InitVar[str | None]
    infer_breadcrumb_text: dataclasses.InitVar[bool]
    infer_entry_text: dataclasses.InitVar[bool]
    infer_exit_text: dataclasses.InitVar[bool]

    def __post_init__(self, name_for_inference, infer_breadcrumb_text, infer_entry_text, infer_exit_text):
        if name_for_inference is None:
            name_for_inference = self.function.__name__
        
        if infer_breadcrumb_text:
            self.breadcrumb_text = name_for_inference
        if infer_entry_text:
            self.entry_text = f"Entering {name_for_inference}"
        if infer_exit_text:
            self.exit_text = f"Exiting {name_for_inference}"

        functools.wraps(self.function)(self)
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if inspect.isgeneratorfunction(self.function):
            return self._call_generator(*args, **kwargs)
        else:
            return self._call_normally(*args, **kwargs)
    
    def _call_normally(self, *args, **kwargs):
        entry_and_exit_ctx = nullcontext()
        if self.message_level is not None:
            entry_and_exit_ctx = LogEntryAndExitContext(self.message_level, self.entry_text, self.exit_text, self.logger)

        breadcrumb_ctx = nullcontext()
        if self.breadcrumb_text is not None:
            breadcrumb_ctx = PushBreadcrumbContext(self.breadcrumb_text)

        with entry_and_exit_ctx, breadcrumb_ctx:
                return self.function(*args, **kwargs)

    def _call_generator(self, *args, **kwargs):
        breadcrumbs = _TLS.breadcrumbs
        if self.breadcrumb_text is not None:
            breadcrumbs += (self.breadcrumb_text,)
        generator = self.function(*args, **kwargs)
        return _LoggedGenerator(generator, breadcrumbs)
    
    def __reduce__(self):
        return self.function.__name__


_DEFAULT = object()


def logged[**P, R](*args, **kwargs) -> Callable[[Callable[P, R]], _LoggedFunction[P, R]]:
    logger: MyLogger = get_module_logger(up_frames=1)
    return logger.logged(*args, **kwargs)


@logging.setLoggerClass
class MyLogger(logging.getLoggerClass()):
    def _log(self, level, msg, *args, **kwargs):
        now = datetime.now().astimezone(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        level_symbol = self._get_level_symbol(level)
        breadcrumbs = " > ".join(_TLS.breadcrumbs)
        complete_message = f"{timestamp} [{level_symbol}] {breadcrumbs}: {msg}"
        return super()._log(level, complete_message, *args, **kwargs)
    
    @staticmethod
    def _get_level_symbol(level: int):
        return logging.getLevelName(level)[0]
    
    def log_entry_and_exit(self, message_level=logging.INFO, entry_text=_DEFAULT, exit_text=_DEFAULT, name=None):
        if name is None and entry_text in (_DEFAULT, None) and exit_text in (_DEFAULT, None):
            raise TypeError("At least one of 'entry_message', 'exit_message', or 'name' must be provided")
        
        if name is not None:
            if entry_text is _DEFAULT:
                entry_text = f"Entering {name}"
            if exit_text is _DEFAULT:
                exit_text = f"Exiting {name}"
        
        return LogEntryAndExitContext(message_level, entry_text, exit_text, self)
    
    def logged[**P, R](self, message_level=logging.DEBUG, breadcrumb_text=_DEFAULT, entry_text=_DEFAULT, exit_text=_DEFAULT, name=None) -> Callable[[Callable[P, R]], _LoggedFunction[P, R]]:
        if isinstance(message_level, Callable):
            raise TypeError("argument 'message_level' must be int, not callable; did you forget some empty parentheses?")

        return functools.partial(
            _LoggedFunction,
            breadcrumb_text=breadcrumb_text,
            entry_text=entry_text,
            exit_text=exit_text,
            message_level=message_level,
            logger=self,
            name_for_inference=name,
            infer_breadcrumb_text=(breadcrumb_text is _DEFAULT),
            infer_entry_text=(entry_text is _DEFAULT),
            infer_exit_text=(entry_text is _DEFAULT)
        )



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


def configure_logging(console_level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(get_log_file_path())
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    for handler in [file_handler, console_handler]:
        root_logger.addHandler(handler)

    _TLS.breadcrumbs = ()

    sys.excepthook = log_uncaught_exception


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.getLogger().critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def get_module_logger(up_frames=0):
    frame = inspect.stack()[1 + up_frames].frame
    name = frame.f_locals["__name__"]
    if name == "__main__":
        import __main__
        name = __main__.__spec__.name
    return logging.getLogger(name)
