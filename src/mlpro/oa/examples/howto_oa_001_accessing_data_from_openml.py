from mlpro.wrappers.openml import WrStreamProviderOpenML


# Create a Wrapper for OpenML stream provider
open_ml = WrStreamProviderOpenML()


# Get a list of streams available at the stream provider
stream_list = open_ml.get_stream_list()
for stream in stream_list:
    print(stream)


# Get a specific stream from the stream provider
stream = open_ml.get_stream(61)


#get the feature space of the stream
print("Number of features in the stream:",stream.get_feature_space().get_num_dim())

#resetting the stream
stream.reset()


# Iterating through the stream instances
for i in range(10):
    curr_instance = stream.get_next().get_values()
    stream.log(stream.C_LOG_TYPE_I,'Current Instance:' , curr_instance)

# resetting the stream
stream.reset()

# Iterating through the stream instances
for i in range(5):
    curr_instance = stream.get_next().get_values()
    stream.log(stream.C_LOG_TYPE_I, 'Current Instance:' , curr_instance)
