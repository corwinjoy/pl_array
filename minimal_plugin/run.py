import polars as pl
import minimal_plugin as mp
from minimal_plugin import pig_latinnify

def test_pig():
    df = pl.DataFrame(
        {
        "english": ["this", "is", "not", "pig", "latin"],
        }
    )
    result = df.with_columns(pig_latin=pig_latinnify("english"))
    print(result)
    
def test_noop():
    df = pl.DataFrame({
        'a': [1, 1, None],
        'b': [4.1, 5.2, 6.3],
        'c': ['hello', 'everybody!', '!']
    })
    print(df.with_columns(mp.noop(pl.all()).name.suffix('_noop')))
    
test_noop()
