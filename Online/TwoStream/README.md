Online two-stream attention classifier.

Takes a stream of data and computes appearance and flow features. The model then
annotates each incoming frame with an attention level (using a fusion of both
outputs). The LSTM models the current and previous frames.
