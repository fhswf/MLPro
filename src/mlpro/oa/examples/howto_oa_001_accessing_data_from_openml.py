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
print(stream.get_feature_space()._dim_ids)


stream.reset()

for i in range(10):
    print(stream.get_next())
