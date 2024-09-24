#![allow(clippy::unused_unit)]
#![allow(unexpected_cfgs)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::Container;
use serde::Deserialize;

#[derive(Deserialize)]
struct ArrayKwargs {
    // I guess DataType is not one of the serializable types?
    // In the source code I see this done vie Wrap<DataType>
    // e.g. polars/crates/polars-python/src/series /construction.rs
    //      fn new_from_any_values_and_dtype
    dtype: String,
}

pub fn array_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if input_fields.is_empty() {
        // TODO: Allow specifying dtype?
        polars_bail!(ComputeError: "need at least one input field to determine dtype")
    }
    let expected_dtype: DataType = input_fields[0].dtype.clone();

    if !expected_dtype.is_numeric() {
        polars_bail!(ComputeError: "all input fields must be numeric")
    }

    for field in input_fields.iter().skip(1) {
        if field.dtype != expected_dtype {
            // TODO: Support casting?
            polars_bail!(ComputeError: "all input fields must have the same type")
        }
    }
    Ok(Field::new(
        PlSmallStr::from_static("array"),  // Not sure how to set field name, maybe take the first input field name, or concatenate names?
        DataType::Array(Box::new(expected_dtype), input_fields.len()),
    ))
}

// It looks like output_type_fn_kwargs can support keyword arguments??
// But I can't find any docs and am not sure how to use it.
// #[polars_expr(output_type_fn_kwargs=array_output_type)]
#[polars_expr(output_type_func=array_output_type)]
fn array(inputs: &[Series], kwargs: ArrayKwargs) -> PolarsResult<Series> {
    array_internal(inputs, kwargs)
}

// Create a new array from a slice of series
fn array_internal(inputs: &[Series], _kwargs: ArrayKwargs) -> PolarsResult<Series> {
    // TODO. Figure out how to pass an optional dtype
    /*
    let dtype: DataType = &kwargs.dtype;
    let s = &inputs[0];
    polars_ensure!(
        s.dtype() == dtype,
        ComputeError: "Expected {}, got: {}", dtype, s.dtype()
    );
    */

    let dtype: &DataType = inputs[0].dtype();

    /*
    I feel like there should be some kind of function to map DataType to native types
    But, I can't seem to find it so I have a manual map for now.
    I do also see dispatch code like this in the source code, so maybe there is no such function?
     */

    match dtype {
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => array_numeric::<UInt8Type>(inputs, dtype),
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => array_numeric::<UInt16Type>(inputs, dtype),
        DataType::UInt32 => array_numeric::<UInt32Type>(inputs, dtype),
        DataType::UInt64 => array_numeric::<UInt64Type>(inputs, dtype),
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => array_numeric::<Int8Type>(inputs, dtype),
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => array_numeric::<Int16Type>(inputs, dtype),
        DataType::Int32 => array_numeric::<Int32Type>(inputs, dtype),
        DataType::Int64 => array_numeric::<Int64Type>(inputs, dtype),
        DataType::Float32 => array_numeric::<Float32Type>(inputs, dtype),
        DataType::Float64 => array_numeric::<Float64Type>(inputs, dtype),
        dt => polars_bail!(ComputeError: "array not implemented for dtype {:?}", dt)
    }
}

// Combine numeric series into an array
fn array_numeric<'a, T: PolarsNumericType>(inputs: &[Series], dtype: &DataType)
    -> PolarsResult<Series> {
    let rows = inputs[0].len();
    let cols = inputs.len();
    let capacity = cols * rows;

    let mut rows_ca: ListPrimitiveChunkedBuilder<T> =
        ListPrimitiveChunkedBuilder::new("Array".into(), capacity, capacity,
                                         dtype.clone());
    let mut cols_ca: Vec<_> = inputs.as_ref().iter().map(|e|
        e.unpack::<T>().unwrap().iter()).collect();

    let mut row: Vec<T::Physical<'a>> = Vec::with_capacity(cols);
    for _i in 0..rows {
        row.clear();
        for j in 0..cols {
            if let Some(Some(val)) = cols_ca[j].next() {
                row.push(val);
            } else {
                row.push(T::Native::default());
            }
        }
        rows_ca.append_slice(&row);
    }

    let s = rows_ca.finish().into_series();
    let sa = s.cast(&DataType::Array(Box::new(dtype.clone()), cols))?;
    Ok(sa.into_series())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_f64() {
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
        let kwargs = ArrayKwargs{dtype: "f64".to_string()};
        let expected_result = array_output_type(&fields).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = array_internal(&cols, kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }

    #[test]
    fn test_array_i32() {
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
        let kwargs = ArrayKwargs{dtype: "f64".to_string()};
        let expected_result = array_output_type(&fields).unwrap();
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

