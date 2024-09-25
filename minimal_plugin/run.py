from time import sleep

import polars as pl
import minimal_plugin as mp

pl.Config.set_verbose(True)

def build_array_f64():
    df = pl.DataFrame(
        [
            pl.Series("f1", [1, 2]),
            pl.Series("f2", [3, 4]),
        ],
        schema={
            "f1": pl.Float64,
            "f2": pl.Float64,
        },
    )
    print(df)

    # Now we call our plugin:
    # sleep(30)
    result = print(df.with_columns(arr=mp.array(pl.all(), dtype="f64")))
    print(result)

def build_array_i32():
    df = pl.DataFrame(
        [
            pl.Series("f1", [1, 2]),
            pl.Series("f2", [3, 4]),
        ],
        schema={
            "f1": pl.Int32,
            "f2": pl.Int32,
        },
    )
    print(df)

    # Now we call our plugin:
    # sleep(30)
    result = print(df.with_columns(arr=mp.array(pl.all(), dtype=pl.Float64)))
    print(result)

# build_array_f64()
build_array_i32()
