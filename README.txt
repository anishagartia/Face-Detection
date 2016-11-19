1. vlfeat not included in the submission. The user MUST download vlfeat in order to successfully run this project.

2. For graduate credit, the following is done
implementing HOG
Using Hard negative mining
using additional positive train data
Implementing additional classifier: DeepFaces Neural network

3. Change the following parameters to true to use them, and false to not use the technique. This must be changed in proj5 line 58.

use_vl_hog = true; 
use_hard_negative_mining = false;
use_lfw_dataset = false;
use_neural_net = false;

4. For the neural network, test_model.py must be executed separately. Keras is not included in submission. The code WILL BREAK if keras and python 3.5 are not installed in the system. There are no input parameters required to execute test_model.py. After executing test_model.py, run the validation cell of proj5 again. 


