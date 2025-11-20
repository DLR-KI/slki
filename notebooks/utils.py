# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#

# Some utilities for the notebooks
###############################################################################

from typing import Any, overload


###############################################################################
# temporary fix utils issue/bug is solved.
# https://github.com/pwwang/python-varname/issues/114
#############

try:
    from varname import VarnameRetrievingError, argname  # type: ignore[import]
    from varname.ignore import IgnoreList  # type: ignore[import]
    from varname.utils import bytecode_nameof, get_node_by_frame  # type: ignore[import]

    varname_installed = True
except ImportError:
    # varname is not installed
    varname_installed = False
    print("ERROR: could not load 'varname' package")


if varname_installed:

    @overload
    def nameof(
        var: Any,
        *,
        frame: int = 1,
        vars_only: bool = True,
    ) -> str:  # pragma: no cover
        ...

    @overload
    def nameof(
        var: Any,
        more_var: Any,
        /,  # introduced in python 3.8
        *more_vars: Any,
        frame: int = 1,
        vars_only: bool = True,
    ) -> tuple[str, ...]:  # pragma: no cover
        ...

    def nameof(  # noqa: D417
        var: Any,
        *more_vars: Any,
        frame: int = 1,
        vars_only: bool = True,
        **kwargs,  # <<<-------------- THIS IS THE TEMPORARY FIX (just adding **kwargs to the signature)
    ) -> str | tuple[str, ...]:
        """
        Get the names of the variables passed in.

        Examples:
            >>> a = 1
            >>> nameof(a) # 'a'

            >>> b = 2
            >>> nameof(a, b) # ('a', 'b')

            >>> x = lambda: None
            >>> x.y = 1
            >>> nameof(x.y, vars_only=False) # 'x.y'

        Note:
            This function works with the environments where source code is
            available, in other words, the callee's node can be retrieved by
            `executing`. In some cases, for example, running code from python
            shell/REPL or from `exec`/`eval`, we try to fetch the variable name
            from the bytecode. This requires only a single variable name is passed
            to this function and no keyword arguments, meaning that getting full
            names of attribute calls are not supported in such cases.

        Args:
            var: The variable to retrieve the name of
            *more_vars: Other variables to retrieve the names of
            frame: The this function is called from the wrapper of it. `frame=1`
                means no wrappers.
                Note that the calls from standard libraries are ignored.
                Also note that the wrapper has to have signature as this one.
            vars_only: Whether only allow variables/attributes as arguments or
                any expressions. If `False`, then the sources of the arguments
                will be returned.

        Returns:
            The names/sources of variables/expressions passed in.
                If a single argument is passed, return the name/source of it.
                If multiple variables are passed, return a tuple of their
                names/sources.
                If the argument is an attribute (e.g. `a.b`) and `vars_only` is
                `True`, only `"b"` will returned. Set `vars_only` to `False` to
                get `"a.b"`.

        Raises:
            VarnameRetrievingError: When the callee's node cannot be retrieved or
                trying to retrieve the full name of non attribute series calls.
        """
        # Frame is anyway used in get_node
        frameobj = IgnoreList.create(
            ignore_lambda=False,
            ignore_varname=False,
        ).get_frame(frame)

        node = get_node_by_frame(frameobj, raise_exc=True)
        if not node:
            # We can't retrieve the node by executing.
            # It can be due to running code from python/shell, exec/eval or
            # other environments where sourcecode cannot be reached
            # make sure we keep it simple (only single variable passed and no
            # full passed) to use bytecode_nameof
            #
            # We don't have to check keyword arguments here, as the instruction
            # will then be CALL_FUNCTION_KW.
            if not more_vars:
                return bytecode_nameof(frameobj.f_code, frameobj.f_lasti)

            # We are anyway raising exceptions, no worries about additional burden
            # of frame retrieval again
            source = frameobj.f_code.co_filename
            if source == "<stdin>":
                raise VarnameRetrievingError(
                    "Are you trying to call nameof in REPL/python shell? "
                    "In such a case, nameof can only be called with single "
                    "argument and no keyword arguments."
                )
            if source == "<string>":
                raise VarnameRetrievingError(
                    "Are you trying to call nameof from exec/eval? "
                    "In such a case, nameof can only be called with single "
                    "argument and no keyword arguments."
                )
            raise VarnameRetrievingError(
                "Source code unavailable, nameof can only retrieve the name of "
                "a single variable, and argument `full` should not be specified."
            )

        out = argname(
            "var",
            "*more_vars",
            func=nameof,
            frame=frame,
            vars_only=vars_only,
        )
        return out if more_vars else out[0]  # type: ignore

###############################################################################
