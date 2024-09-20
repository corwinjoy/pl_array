from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from minimal_plugin._internal import __version__ as __version__

if TYPE_CHECKING:
    from minimal_plugin.typing import IntoExpr

LIB = Path(__file__).parent


def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="pig_latinnify",
        is_elementwise=True,
    )
    
def noop(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="noop",
        is_elementwise=True,
    )

def array(expr: IntoExpr, dtype: str) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="array",
        is_elementwise=True,
        kwargs={"dtype": dtype},
    )
