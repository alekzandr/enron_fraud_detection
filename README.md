The tester.py and poi_id.py scripts have been modified
from the original versions to allow LightGBM and
Tensorflow APIs.

Currently the tester.py script returns perfect
evalatuation metrics. I'm currently under the
impression my LightGBM model has overfit the
data.

To respect the reviewer's time, I've commented out
the following lines in poi_id.py so that
you only train the required LightGBM model.

Line 288 - Train Neural Network
Line 369 - Train KMeans Model
Line 376 - Beginning of Block Quotes
Line 427 - End of Block Quotes

Remove the above comments to allow
full training of the two other models
and evaluation of the ensemble.

Expected resultes are documented at
the end of the 'Intro to Machine Learning' document.
