import inspect
import logging
import os
from typing import Callable, Iterable, Optional, TypeVar

import triton
from triton.runtime.jit import DependenciesFinder
from triton.runtime.jit import JITFunction as TritonJITFunction

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JITFunction(TritonJITFunction[T]):
    def __init__(
        self,
        fn,
        version=None,
        do_not_specialize=None,
        do_not_specialize_on_alignment=None,
        debug=None,
        noinline=None,
        repr=None,
        launch_metadata=None,
        walk_dependency=None,
    ):
        super().__init__(
            fn,
            version,
            do_not_specialize,
            do_not_specialize_on_alignment,
            debug,
            noinline,
            repr,
            launch_metadata,
        )

        assert triton.__version__ in ["3.4.0"]

        self.walk_dependency = walk_dependency

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from triton.compiler import ASTSource, CompiledKernel, compile, make_backend
        from triton.runtime.jit import create_function_from_signature, driver

        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel

        def compile_wrapper(*args, **kwargs):
            logger.info(f"Triton compile started {self.fn}")
            result = compile(*args, **kwargs)
            logger.info(f"Triton compile ended {self.fn}")
            return result

        self.compile = compile_wrapper
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, target, backend, binder

    @property
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        if self.hash is None:
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            if self.walk_dependency:
                dependencies_finder = DependenciesFinder(
                    name=self._fn_name,
                    globals=self.__globals__,
                    nonlocals=nonlocals,
                    src=self.src,
                )
                dependencies_finder.visit(self.parse())
                self.hash = dependencies_finder.ret + str(self.starting_line_number)
                self.used_global_vals = dict(
                    sorted(dependencies_finder.used_global_vals.items())
                )
            else:
                self.hash = (
                    str(self.src) + str(self._fn_name) + str(self.starting_line_number)
                )
                self.used_global_vals = {}
        return self.hash


def triton_jit(
    # jit parameters
    walk_dependency: bool = None,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
    # autotune parameters
    do_autotune: Optional[bool] = None,
    configs=None,
    key=None,
    restore_value=None,
):
    if walk_dependency is None:
        walk_dependency = (
            os.getenv("HIP_DEBUG_TRITON_COMPILE_IGNORE_DEPENDENCY", "0") == "0"
        )

    def decorator(fn: T) -> JITFunction[T]:
        logger.info(f"JIT Initilized {fn}")

        assert callable(fn)
        if os.getenv("TRITON_INTERPRET", "0") == "1":
            raise Exception("interpret should avoided")
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
                walk_dependency=walk_dependency,
            )

    if do_autotune is None:
        do_autotune = configs is not None

    if do_autotune:
        return lambda fn: triton.autotune(
            configs=configs,
            key=key,
            restore_value=restore_value,
        )(decorator(fn))
    return decorator
