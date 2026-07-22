"""
Module for preprocessing the cleveland dataset
"""

using CSV
using DataFrames
using Missings
using Statistics
using CategoricalArrays

"""
Loads the dataset (cleveland for now) from given path and returns
the dataframe
# Arguments
- `path::String`: path to the Cleveland dataset
"""
function load_data(path::String)::DataFrame
    df = CSV.read(path, DataFrame; header=false, missingstring="?")
    return df
end

"""
Since data in Cleveland set does not include the header with column names
they are hardcoded and set here
"""
function set_columns_names!(df::DataFrame)
    # Functions with ! mutate the argument (Haskell don't like)
    rename!(df, [:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope, :ca, :thal, :num])
end

"""
There are two columns with missing data: ca and thal.
Ca is numeric so it's inputet with median of the whole column
Thal is categorical (and only missing for 2 rows) so they are just dropped
"""
function handle_missing_data!(df::DataFrame)
    # Since ca is numeric value I input it with median of the column
    df.:ca = coalesce.(df.:ca, median(skipmissing(df.:ca)))
    # Since thal is categorical and has only 2 values missing I just drop those rows
    df = dropmissing(df, :thal)
end


"""
Converts the categorical columns to CategoricalArray
"""
function convert_categorical!(df::DataFrame)
    df.:sex = categorical(df.:sex)
    df.:cp = categorical(df.:cp)
    df.:fbs = categorical(df.:fbs)
    df.:restecg = categorical(df.:restecg)
    df.:exang = categorical(df.:exang)
    df.:slope = categorical(df.:slope)
    df.:thal = categorical(df.:thal)
end

"""
Loads the dataset from file to the dataframe, sets column names, handles missing data
and categorical features and retun the clean dataframe, ready for EDA or model training.
# Arguments
- `path::String`: path to the Cleveland dataset
"""
function prepare_data(path::String)::DataFrame
    df = load_data(path)
    set_columns_names!(df)
    handle_missing_data!(df)
    convert_categorical!(df)
    return df
end
