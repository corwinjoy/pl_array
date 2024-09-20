#![allow(clippy::unused_unit)]
use polars::prelude::*;
use polars::datatypes::DataType;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use polars::export::arrow::legacy::utils::CustomIterTools;
use serde::Deserialize;
// use polars_core::utils::Wrap;

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    Ok(s.clone())
}

pub fn array_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if input_fields.is_empty() {
        // TODO: Allow specifying dtype?
        polars_bail!(ComputeError: "need at least one input field to determine dtype")
    }
    // let expected_dtype: DataType = input_fields[1].dtype.clone();
    let expected_dtype: DataType = DataType::Float64;
    for field in input_fields.iter().skip(1) {
        if field.dtype != expected_dtype {
            // TODO: Support casting?
            polars_bail!(ComputeError: "all input fields must have the same type")
        }
    }
    Ok(Field::new(
        PlSmallStr::from_static("array"),  // Not sure how to set field name, maybe take the first input field name, or concatenate names?
        DataType::Array(Box::new(expected_dtype), input_fields.len()-1),
    ))
}

pub fn array2_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("arr"),
        DataType::Array(Box::new(DataType::Float64), 2),
    ))
}

#[derive(Deserialize)]
struct ArrayKwargs {
    // I guess DataType is not one of the serializable types?
    // In the source code I see this done vie Wrap<DataType>
    // e.g. polars/crates/polars-python/src/series /construction.rs
    // fn new_from_any_values_and_dtype
    dtype: String,
}

#[polars_expr(output_type_func=array2_output)]
fn array(inputs: &[Series], kwargs: ArrayKwargs) -> PolarsResult<Series> {
    array_internal(inputs, kwargs)
}

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

    let out_arr: &ArrayChunked = inputs[0].array()?; // TODO, create new array of 0s here
    let mut one_plus = inputs.as_ref().iter(); // Columns 1 and up
    one_plus.next();
    let cols_ca: Vec<_> = one_plus.map(|e| e.f64()).collect();
    let mut row_idx = 0;

    let out: ArrayChunked = unsafe {
        out_arr.try_apply_amortized_same_type(|row| {
            let s = row.as_ref();
            let ca = s.f64()?;
            let mut cols_ca_iter = cols_ca.iter();
            let out_inner: Float64Chunked = ca
                .iter()
                .map(|opt_val| {
                    opt_val.map(|val| {
                        val + cols_ca_iter.next().unwrap().as_ref().unwrap().value_unchecked(row_idx)
                    })
                }).collect_trusted();
            row_idx += 1;
            Ok(out_inner.into_series())
        })}?;

    Ok(out.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array() {
        let mut col1: ListPrimitiveChunkedBuilder<Float64Type> =
            ListPrimitiveChunkedBuilder::new("Array_1".into(), 8, 8,
                                             DataType::Float64);
        col1.append_slice(&[0.0, 0.0]);
        col1.append_slice(&[0.0, 0.0]);

        let f1 = Series::new("f1".into(), &[1.0, 2.0]);
        let f2 = Series::new("f2".into(), &[3.0, 4.0]);

        let cols = vec![
            col1.finish().into_series(),
            f1,
            f2
        ];

        let array_df = DataFrame::new(cols.clone()).unwrap();
        println!("input df\n{}", &array_df);

        let mut fields: Vec<Field> = Vec::new();
        for col in &cols{
            let f: Field = (col.field().to_mut()).clone();
            fields.push(f);
        }
        let expected_result = array_output_type(&fields).unwrap();
        println!("expected result\n{:?}", &expected_result);

        let kwargs = ArrayKwargs{dtype: "f64".to_string()};
        let new_arr = array_internal(&cols, kwargs);
        println!("actual result\n{:?}", &new_arr);

        // TODO check output type
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

