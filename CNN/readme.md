Simple 4 layered 2D CNN designed to predict the location of the UAV based on the specified inputs.

For this case the inputs are tensors with dimension n_bs x n_paths x n_features (features are equivalent to channels).
The output is a 3 dimensional vector (x,y,z).

Download the data from google drive before running the code.

There is no noise considered in this implementation.
