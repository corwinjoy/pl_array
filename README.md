# pl_array
Prototype Module for array creation function in polars.

To firm up discussion this initial work is done as a plug-in.
1. See https://github.com/pola-rs/polars/issues/18090 for the proposal / issue.
2. See https://marcogorelli.github.io/polars-plugins-tutorial/arrays/ for a polars plug-in overview.

Usage:
1. Setup a virtual environment as per `minimal_plugin/requirements.txt`
2. Install rust.
3. Edit function definitions in the minimal_plugin directory:
    1. `minimal_plugin/__init__.py`
    2. `src/lib.rs`
4. Compile via the command `maturin develop`
5. Run the sample code via `python run.py`
