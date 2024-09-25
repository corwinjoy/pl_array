from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from minimal_plugin._internal import __version__ as __version__

if TYPE_CHECKING:
    from minimal_plugin.typing import IntoExpr, PolarsDataType

LIB = Path(__file__).parent


def array(expr: IntoExpr, dtype: Optional[PolarsDataType]) -> pl.Expr:
    if dtype is None:
        dtype_expr = ""
    else:
        # Hacky way to pass a DataType to rust, we serialize an expression
        # for selecting columns based on dtypes.
        dtype_expr = pl.col(dtype).meta.serialize(format="json")
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="array",
        is_elementwise=True,
        kwargs={"dtype_expr": dtype_expr},
        input_wildcard_expansion = True # It seems we need this to get all columns at once?
    )
