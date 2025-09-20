"""Sandboxed execution environment for untrusted code evaluation."""

import os
import sys
import time
import types
import signal
import builtins
import platform
import tempfile
import contextlib
import subprocess
import unittest
import multiprocessing
from multiprocessing import Manager, Value
from typing import Tuple, Dict, Any
import io


# Status codes
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
TIMEOUT_LIMIT = 240.0

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    """Context manager for redirecting stdin."""
    _stream = "stdin"


@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


@contextlib.contextmanager
def swallow_io():
    """Context manager to swallow all IO operations."""
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    """Context manager to create and clean up temporary directory."""
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    """Context manager to change directory."""
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    """
    Disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING: This function is NOT a security sandbox. Untrusted code should
    not be blindly executed outside of proper sandboxing.
    """
    import os
    import time

    os.environ['TZ'] = 'UTC'
    time.tzset()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    os.environ['MPLBACKEND'] = 'Agg'

    if max_as_limit and max_data_limit and max_stack_limit:
        import resource

        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024

        resource.setrlimit(
            resource.RLIMIT_AS, (max_as_limit, max_as_limit)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (max_data_limit, max_data_limit)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit)
            )

    import faulthandler
    faulthandler.disable()

    builtins.exit = None
    builtins.quit = None


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    status,  # Value
    stat,  # Dict
    details,  # Dict
):
    """Execute code in a restricted environment."""
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir
        import os
        import shutil
        import builtins

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        module_name = "__test__"
        new_module = types.ModuleType(module_name)

        # Set necessary attributes for the module
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        # Initialize stats
        stat["num_tests"] = 0
        stat["num_tests_failed"] = 0
        stat["num_tests_passed"] = 0
        stat["has_syntax_error"] = False
        stat["has_name_error"] = False

        try:
            # Set matplotlib backend before any imports
            matplotlib_backend_setup = """
import os
os.environ['MPLBACKEND'] = 'Agg'
try:
    import matplotlib
    matplotlib.use('Agg', force=True)
except ImportError:
    pass
"""
            full_code = matplotlib_backend_setup + "\n" + code + "\n" + test_code
            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace

            status.value = _SUCCESS
            stat["num_tests"] = test_result.testsRun
            stat["num_tests_failed"] = len(issues)
            stat["num_tests_passed"] = stat["num_tests"] - stat["num_tests_failed"]

        except SyntaxError as e:
            details["ALL"] = str(e)
            status.value = _FAILED
            stat["has_syntax_error"] = True
        except NameError as e:
            details["ALL"] = str(e)
            status.value = _FAILED
            stat["has_name_error"] = True
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(e)
        except BaseException as e:
            details["ALL"] = str(e)
            status.value = _FAILED

        # Restore functions needed for cleanup
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float = 1024,
    max_data_limit: float = 1024,
    max_stack_limit: float = 1024,
    min_time_limit: float = 10,
    gt_time_limit: float = 60
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Run untrusted code in a sandboxed environment.

    Args:
        code: The code to test
        test_code: The test code to run against the solution
        entry_point: The function name to test
        max_as_limit: Maximum address space in MB
        max_data_limit: Maximum data segment in MB
        max_stack_limit: Maximum stack size in MB
        min_time_limit: Minimum time limit in seconds
        gt_time_limit: Ground truth time limit in seconds

    Returns:
        Tuple of (status, statistics, details)
    """
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), min_time_limit) + 1

    # Shared memory objects for multiprocessing
    status = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()
    stat = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            status,
            stat,
            details,
        ),
    )

    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    # Convert to regular Python types
    status = _mapping[status.value]
    stat = dict(stat)
    details = dict(details)

    if not status:
        status = TIMEOUT
    if status == PASS and details:
        status = FAIL

    return status, stat, details