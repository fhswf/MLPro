from mlpro.wrappers.openml import WrStreamProviderOpenML

# Create a Wrapper for OpenML stream provider
open_ml = WrStreamProviderOpenML()

# Get a list of streams available at the stream provider
open_ml.get_stream_list()

# Get a specific stream from the stream provider
open_ml.get_stream(46)