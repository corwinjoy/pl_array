#![allow(clippy::unused_unit)]
#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use polars::export::arrow::array::{FixedSizeListArray, PrimitiveArray};
use polars::prelude::*;
use polars_plan::dsl::Expr;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::arrow::bitmap::MutableBitmap;
use pyo3_polars::export::polars_core::utils::Container;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct ArrayKwargs {
    // I guess DataType is not one of the serializable types?
    // In the source code I see this done vie Wrap<DataType>
    dtype_expr: String,
}

fn deserialize_dtype(dtype_expr: &str) -> PolarsResult<Option<DataType>> {
    match dtype_expr.len() {
        0 => Ok(None),
        _ => match serde_json::from_str::<Expr>(dtype_expr) {
            Ok(Expr::DtypeColumn(dtypes)) if dtypes.len() == 1 => Ok(Some(dtypes[0].clone())),
            Ok(_) => Err(
                polars_err!(ComputeError: "Expected a DtypeColumn expression with a single dtype"),
            ),
            Err(_) => Err(polars_err!(ComputeError: "Could not deserialize dtype expression")),
        },
    }
}

fn array_output_type(input_fields: &[Field], kwargs: ArrayKwargs) -> PolarsResult<Field> {
    let expected_dtype = deserialize_dtype(&kwargs.dtype_expr)?
        .unwrap_or(input_fields[0].dtype.clone());

    for field in input_fields.iter() {
        if !field.dtype().is_numeric() {
            polars_bail!(ComputeError: "all input fields must be numeric")
        }
    }

    /*
    // For now, allow casting to either the first, or provided dtype
    for field in input_fields.iter().skip(1) {
        if field.dtype != expected_dtype {
            polars_bail!(ComputeError: "all input fields must have the same type")
        }
    }
    */

    Ok(Field::new(
        PlSmallStr::from_static("array"),
        DataType::Array(Box::new(expected_dtype), input_fields.len()),
    ))
}

#[polars_expr(output_type_func_with_kwargs=array_output_type)]
fn array(inputs: &[Series], kwargs: ArrayKwargs) -> PolarsResult<Series> {
    array_internal(inputs, kwargs)
}

// Create a new array from a slice of series
fn array_internal(inputs: &[Series], kwargs: ArrayKwargs) -> PolarsResult<Series> {
    let ref dtype = deserialize_dtype(&kwargs.dtype_expr)?
        .unwrap_or(inputs[0].dtype().clone());

    // Convert dtype to native numeric type and invoke array_numeric
    with_match_physical_numeric_polars_type!(dtype, |$T| {
        array_numeric::<$T>(inputs, dtype)
    })
}

// Combine numeric series into an array
fn array_numeric<'a, T: PolarsNumericType>(inputs: &[Series], dtype: &DataType)
    -> PolarsResult<Series> {
    let rows = inputs[0].len();
    let cols = inputs.len();
    let capacity = cols * rows;

    let mut values: Vec<T::Native> = vec![T::Native::default(); capacity];

    // Support for casting
    // Cast fields to the target dtype as needed
    let mut casts = HashMap::new();
    for j in 0..cols {
        if inputs[j].dtype() != dtype {
            let cast_input = inputs[j].cast(dtype)?;
            casts.insert(j, cast_input);
        }
    }

    let mut cols_ca = Vec::new();
    for j in 0..cols {
        if inputs[j].dtype() != dtype {
            cols_ca.push(casts.get(&j).expect("expect conversion").unpack::<T>()?);
        } else {
            cols_ca.push(inputs[j].unpack::<T>()?);
        }
    }

    for i in 0..rows {
        for j in 0..cols {
            values[i * cols + j] = unsafe { cols_ca[j].value_unchecked(i) };
        }
    }

    let validity = if cols_ca.iter().any(|col| col.has_nulls()) {
        let mut validity = MutableBitmap::from_len_zeroed(capacity);
        for (j, col) in cols_ca.iter().enumerate() {
            let mut row_offset = 0;
            for chunk in col.chunks() {
                if let Some(chunk_validity) = chunk.validity() {
                    for set_bit in chunk_validity.true_idx_iter() {
                        validity.set(cols * (row_offset + set_bit) + j, true);
                    }
                } else {
                    for chunk_row in 0..chunk.len() {
                        validity.set(cols * (row_offset + chunk_row) + j, true);
                    }
                }
                row_offset += chunk.len();
            }
        }
        Some(validity.into())
    } else {
        None
    };

    let values_array = PrimitiveArray::from_vec(values).with_validity(validity);
    let dtype = DataType::Array(Box::new(dtype.clone()), cols);
    let arrow_dtype = dtype.to_arrow(CompatLevel::newest());
    let array = FixedSizeListArray::try_new(arrow_dtype.clone(), Box::new(values_array), None)?;
    Ok(unsafe {Series::_try_from_arrow_unchecked("Array".into(), vec![Box::new(array)], &arrow_dtype)?})
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_f64() {
        println!("\ntest_array_f64");
        let f1 = Series::new("f1".into(), &[1.0, 2.0]);
        let f2 = Series::new("f2".into(), &[3.0, 4.0]);

        let cols = vec![
            f1,
            f2
        ];

        let array_df = DataFrame::new(cols.clone()).unwrap();
        println!("input df\n{}\n", &array_df);

        let mut fields: Vec<Field> = Vec::new();
        for col in &cols{
            let f: Field = (col.field().to_mut()).clone();
            fields.push(f);
        }
        let kwargs = ArrayKwargs{dtype_expr: "{\"DtypeColumn\":[\"Float64\"]}".to_string()};
        let expected_result = array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = array_internal(&cols, kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }

    fn i32_series() -> (Vec<Series>, Vec<Field>){
        let f1 = Series::new("f1".into(), &[1, 2]);
        let f2 = Series::new("f2".into(), &[3, 4]);

        let cols = vec![
            f1,
            f2
        ];

        let array_df = DataFrame::new(cols.clone()).unwrap();
        println!("input df\n{}\n", &array_df);

        let mut fields: Vec<Field> = Vec::new();
        for col in &cols{
            let f: Field = (col.field().to_mut()).clone();
            fields.push(f);
        }
        (cols, fields)
    }

    #[test]
    fn test_array_i32() {
        println!("\ntest_array_i32");
        let (cols, fields) = i32_series();
        let kwargs = ArrayKwargs{dtype_expr: "{\"DtypeColumn\":[\"Int32\"]}".to_string()};
        let expected_result = array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = array_internal(&cols, kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }

    #[test]
    fn test_array_i32_converted() {
        println!("\ntest_array_i32_converted");
        let (cols, fields) = i32_series();
        let kwargs = ArrayKwargs{dtype_expr: "{\"DtypeColumn\":[\"Float64\"]}".to_string()};
        let expected_result = array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = array_internal(&cols, kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }
}


/*
/// Concat lists entries.
pub fn concat_list<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(s: E) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();

    polars_ensure!(!s.is_empty(), ComputeError: "`concat_list` needs one or more expressions");

    Ok(Expr::Function {
        input: s,
        function: FunctionExpr::ListExpr(ListFunction::Concat),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default() | FunctionFlags::INPUT_WILDCARD_EXPANSION,
            ..Default::default()
        },
    })
}

/// Concatenate multiple [`Array`] of the same type into a single [`Array`].
pub fn concatenate(arrays: &[&dyn Array]) -> PolarsResult<Box<dyn Array>> {
    if arrays.is_empty() {
        polars_bail!(InvalidOperation: "concat requires input of at least one array")
    }

    if arrays
        .iter()
        .any(|array| array.data_type() != arrays[0].data_type())
    {
        polars_bail!(InvalidOperation: "It is not possible to concatenate arrays of different data types.")
    }

    let lengths = arrays.iter().map(|array| array.len()).collect::<Vec<_>>();
    let capacity = lengths.iter().sum();

    let mut mutable = make_growable(arrays, false, capacity);

    for (i, len) in lengths.iter().enumerate() {
        // SAFETY: len is correct
        unsafe { mutable.extend(i, 0, *len) }
    }

    Ok(mutable.as_box())
}

/// Concatenate multiple [`Array`][Array] of the same type into a single [`Array`][Array].
/// This does not check the arrays types.
///
/// [Array]: arrow::array::Array
pub fn concatenate_owned_unchecked(arrays: &[ArrayRef]) -> PolarsResult<ArrayRef> {
    if arrays.is_empty() {
        polars_bail!(InvalidOperation: "concat requires input of at least one array")
    }
    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }
    let mut arrays_ref = Vec::with_capacity(arrays.len());
    let mut lengths = Vec::with_capacity(arrays.len());
    let mut capacity = 0;
    for array in arrays {
        arrays_ref.push(&**array);
        lengths.push(array.len());
        capacity += array.len();
    }

    let mut mutable = make_growable(&arrays_ref, false, capacity);

    for (i, len) in lengths.iter().enumerate() {
        // SAFETY:
        // len is within bounds
        unsafe { mutable.extend(i, 0, *len) }
    }

    Ok(mutable.as_box())
}

// /home/cjoy/src/polars/crates/polars-plan/src/dsl/function_expr/list.rs: 385
pub(super) fn concat(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let mut first = std::mem::take(&mut s[0]);
    let other = &s[1..];

    let mut first_ca = match first.list().ok() {
        Some(ca) => ca,
        None => {
            first = first.reshape_list(&[-1, 1]).unwrap();
            first.list().unwrap()
        },
    }
    .clone();

    if first_ca.len() == 1 && !other.is_empty() {
        let max_len = other.iter().map(|s| s.len()).max().unwrap();
        if max_len > 1 {
            first_ca = first_ca.new_from_index(0, max_len)
        }
    }

    first_ca.lst_concat(other).map(|ca| Some(ca.into_series()))
}

// namespace.rs
 fn lst_concat(&self, other: &[Series]) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let other_len = other.len();
        let length = ca.len();
        let mut other = other.to_vec();
        let mut inner_super_type = ca.inner_dtype().clone();

        for s in &other {
            match s.dtype() {
                DataType::List(inner_type) => {
                    inner_super_type = try_get_supertype(&inner_super_type, inner_type)?;
                    #[cfg(feature = "dtype-categorical")]
                    if matches!(
                        &inner_super_type,
                        DataType::Categorical(_, _) | DataType::Enum(_, _)
                    ) {
                        inner_super_type = merge_dtypes(&inner_super_type, inner_type)?;
                    }
                },
                dt => {
                    inner_super_type = try_get_supertype(&inner_super_type, dt)?;
                    #[cfg(feature = "dtype-categorical")]
                    if matches!(
                        &inner_super_type,
                        DataType::Categorical(_, _) | DataType::Enum(_, _)
                    ) {
                        inner_super_type = merge_dtypes(&inner_super_type, dt)?;
                    }
                },
            }
        }

        // cast lhs
        let dtype = &DataType::List(Box::new(inner_super_type.clone()));
        let ca = ca.cast(dtype)?;
        let ca = ca.list().unwrap();

        // broadcasting path in case all unit length
        // this path will not expand the series, so saves memory
        let out = if other.iter().all(|s| s.len() == 1) && ca.len() != 1 {
            cast_rhs(&mut other, &inner_super_type, dtype, length, false)?;
            let to_append = other
                .iter()
                .flat_map(|s| {
                    let lst = s.list().unwrap();
                    lst.get_as_series(0)
                })
                .collect::<Vec<_>>();
            // there was a None, so all values will be None
            if to_append.len() != other_len {
                return Ok(ListChunked::full_null_with_dtype(
                    ca.name().clone(),
                    length,
                    &inner_super_type,
                ));
            }

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();

            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name().clone(),
            )?;
            ca.into_iter().for_each(|opt_s| {
                let opt_s = opt_s.map(|mut s| {
                    for append in &to_append {
                        s.append(append).unwrap();
                    }
                    match inner_super_type {
                        // structs don't have chunks, so we must first rechunk the underlying series
                        #[cfg(feature = "dtype-struct")]
                        DataType::Struct(_) => s = s.rechunk(),
                        // nothing
                        _ => {},
                    }
                    s
                });
                builder.append_opt_series(opt_s.as_ref()).unwrap();
            });
            builder.finish()
        } else {
            // normal path which may contain same length list or unit length lists
            cast_rhs(&mut other, &inner_super_type, dtype, length, true)?;

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();
            let mut iters = Vec::with_capacity(other_len + 1);

            for s in other.iter_mut() {
                iters.push(s.list()?.amortized_iter())
            }
            let mut first_iter: Box<dyn PolarsIterator<Item = Option<Series>>> = ca.into_iter();
            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name().clone(),
            )?;

            for _ in 0..ca.len() {
                let mut acc = match first_iter.next().unwrap() {
                    Some(s) => s,
                    None => {
                        builder.append_null();
                        // make sure that the iterators advance before we continue
                        for it in &mut iters {
                            it.next().unwrap();
                        }
                        continue;
                    },
                };

                let mut has_nulls = false;
                for it in &mut iters {
                    match it.next().unwrap() {
                        Some(s) => {
                            if !has_nulls {
                                acc.append(s.as_ref())?;
                            }
                        },
                        None => {
                            has_nulls = true;
                        },
                    }
                }
                if has_nulls {
                    builder.append_null();
                    continue;
                }

                match inner_super_type {
                    // structs don't have chunks, so we must first rechunk the underlying series
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => acc = acc.rechunk(),
                    // nothing
                    _ => {},
                }
                builder.append_series(&acc).unwrap();
            }
            builder.finish()
        };
        Ok(out)
    }
}


*/

