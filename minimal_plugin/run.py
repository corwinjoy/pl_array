import polars as pl
import minimal_plugin as mp
from minimal_plugin import pig_latinnify

def pig():
    df = pl.DataFrame(
        {
        "english": ["this", "is", "not", "pig", "latin"],
        }
    )
    result = df.with_columns(pig_latin=pig_latinnify("english"))
    print(result)
    
def noop():
    df = pl.DataFrame({
        'a': [1, 1, None],
        'b': [4.1, 5.2, 6.3],
        'c': ['hello', 'everybody!', '!']
    })
    print(df.with_columns(mp.noop(pl.all()).name.suffix('_noop')))

def print_array():
    array_df = pl.DataFrame(
        [
            pl.Series("Array_1", [[1, 3], [2, 5]]),
            pl.Series("Array_2", [[1, 7, 3], [8, 1, 0]]),
        ],
        schema={
            "Array_1": pl.Array(pl.Int64, 2),
            "Array_2": pl.Array(pl.Int64, 3),
        },
    )
    print(array_df)

    data = {"col1": [0, 2], "col2": [3, 7]}
    df2 = pl.DataFrame(data, schema={"col1": pl.Float32, "col2": pl.Int64})
    df2
    
# test_noop()

print_array()
