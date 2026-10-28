import numpy as np
import bisect

BINS_OUT_OF_UPPER_RANGE_VALUE = 5
BINS_OUT_OF_LOWER_RANGE_VALUE = -5
EXPONENT_AUX = 1

# input: 1d np array
# returns: bins relative values (mass proba func), bins edges, min and max values of input array
def get_bins_and_relative_proba(x_values_origin, intermediate_range=(0.0,1.0)):
    x_values = x_values_origin.copy()
    min_value = np.min(x_values)
    max_value = np.max(x_values)

    x_values = min_max_scale_fixed_range(x_values, data_min = min_value, data_max = max_value, new_min=intermediate_range[0], new_max=intermediate_range[1])
    hist, edges = np.histogram(x_values, bins=500, density=True)

    return hist, edges, min_value, max_value

# inout: x values non scaled, min and max of the original collection of values, hist and edges
# returns bin relative value and edges
def transform_values_using_bins(x_values,min_value, max_value, hist, edges,intermediate_range=(0.0,1.0), exponents_values = [2,3], training_mode=True, exp_minv = None, exp_maxv=None):
    original_values = x_values.copy()

    # scale to [0,1]
    x_values = min_max_scale_fixed_range(x_values, data_min = min_value, data_max = max_value, new_min=intermediate_range[0], new_max=intermediate_range[1]) # can return values out of range [0,1]

    # get bins relative values, if out of edges return specific constants
    obtained_bin_values = [get_bin_hist(x_val, hist, edges) for x_val in x_values]


    # values to be transformed: use bin relatives values or updated values if out of range in bin range
    # for a given scaled value x in [0,1],  transform as follow: 2**3**t, where t:
    #
    # if x inside bin edges, 2-t = relative value of bin of x
    # if x falls outside bin edges on left side (x <= 0), then t = 2-(x) resulted in a number >=2
    # if x falls outside bin edges on right side (x >= 1), then t = 2+(x-1) resulted in a number >=2
    # obs: 2 is EXPONENT AUX and can be changed; now it is one
    values_to_be_transformed = np.array([EXPONENT_AUX - obtained_bin_values[i] if (obtained_bin_values[i] != BINS_OUT_OF_UPPER_RANGE_VALUE and obtained_bin_values[i] != BINS_OUT_OF_LOWER_RANGE_VALUE)
                                 else EXPONENT_AUX - x_values[i] if obtained_bin_values[i] == BINS_OUT_OF_LOWER_RANGE_VALUE
                                 else EXPONENT_AUX + (x_values[i] - 1)
                                 for i in range(len(original_values))])

    resulted_exp = apply_exp(values_to_be_transformed, exponents_values)

    exp_min_value = np.min(resulted_exp)
    exp_max_value = np.max(resulted_exp)
    final_values = None
    if training_mode is True:
        final_values = min_max_scale_fixed_range(resulted_exp, data_min = exp_min_value, data_max = exp_max_value,
                                             new_min=intermediate_range[0], new_max=intermediate_range[1]) # can return values out of range [0,1]
    else:
        final_values = min_max_scale_fixed_range(resulted_exp, data_min = exp_minv, data_max = exp_maxv,
                                                     new_min=intermediate_range[0], new_max=intermediate_range[1]) # can return values out of range [0,1]

    if training_mode is True:
        return final_values, exp_min_value, exp_max_value
    else:
        return final_values


# find bin of value, if not in the edges return some constants
def get_bin_hist(value, hist, edges):
    if value == edges[-1]:
        return hist[len(hist) - 1]

    # Find insertion position in edges
    i = bisect.bisect_right(edges, value) - 1

    # Value is outside the histogram range
    if i < 0:
        return BINS_OUT_OF_LOWER_RANGE_VALUE
    if i >= len(hist):
        return BINS_OUT_OF_UPPER_RANGE_VALUE

    return hist[i]


# x values from a feature - 1d array, not scaled
def training_at_feature_level(x_values,intermediate_range=(0,1), exponents_values_v = [2,3]):
    hist, edges, min_value, max_value= get_bins_and_relative_proba(x_values, intermediate_range)
    final_values, exp_min_value, exp_max_value= transform_values_using_bins(x_values,min_value, max_value, hist, edges,intermediate_range, exponents_values = exponents_values_v, training_mode=True)

    return final_values, hist, edges, min_value, max_value, exp_min_value, exp_max_value

# 2d np array, not scaled
def training_at_dataset_level(X_values,intermediate_range=(0,1), exponents_values_v = [2,3]):
    training_data = []
    training_output_col_i=[]

    n_cols = X_values.shape[1]

    for i in range(n_cols):
        x_values_col = X_values[:, i]
        x_values_col = x_values_col.T
        final_values, hist, edges, min_value, max_value, exp_min_value, exp_max_value = training_at_feature_level(x_values_col,intermediate_range, exponents_values_v)
        training_output_col_i.append( (hist, edges, min_value, max_value, exp_min_value, exp_max_value) )
        training_data.append(final_values)

    training_data = np.array(training_data)
    training_data = training_data.T
    return training_data, training_output_col_i

# 2d np array, not scaled
def testing_at_dataset_level(X_testing, training_output_col_i, intermediate_range=(0,1), exponents_values_v = [2,3]):
    testing_data = []

    n_cols = X_testing.shape[1]

    for i in range(n_cols):
        x_values_col = X_testing[:, i]
        x_values_col = x_values_col.T
        final_values = transform_values_using_bins(x_values_col,training_output_col_i[i][2], training_output_col_i[i][3], training_output_col_i[i][0], training_output_col_i[i][1],
                                                   intermediate_range, exponents_values = exponents_values_v,
                                               training_mode=False, exp_minv= training_output_col_i[i][4], exp_maxv= training_output_col_i[i][5])
        testing_data.append(final_values)

    testing_data = np.array(testing_data)
    testing_data = testing_data.T

    f32 = np.finfo(np.float32)
    testing_data = np.clip(testing_data, f32.min, f32.max)
    testing_data[np.isnan(testing_data)] = f32.max

    return testing_data


# input: data is a 1d np array
# return the resized data using the new_min and new_max and considering the data_min and data_max;
# - e.g., if data_max = 100 and new_max = 1, for an input data = [105], its value will be higher than 1 (new_max)
def min_max_scale_fixed_range(data, data_min, data_max, new_min=0, new_max=1):
  data_copy = data.copy()

  # if all elements are equal, set all of them to 0
  if np.all(data_copy == data_copy[0]) and data_min==data_copy[0]:
    data_copy[:] = new_min
    return data_copy

  return ((data_copy - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min

def apply_exp(data, exponents):
    f32 = np.finfo(np.float32)
    # Set NumPy to raise warnings for overflows
    np.seterr(over='warn')

    # Loop through the exponents in reverse order
    for exponent in exponents[::-1]:
        # Try to apply the exponentiation operation
        try:
            data = exponent ** data
        except Exception:
            # If overflow occurs, set those values to np.finfo(data.dtype).max
            data = np.where(data > np.finfo(data.dtype).max, np.finfo(data.dtype).max, data)


    data = np.clip(data, f32.min, f32.max)
    return data
