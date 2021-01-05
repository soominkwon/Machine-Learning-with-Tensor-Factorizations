# Machine-Learning-with-Tensor-Factorizations

Machine learning algorithms (logistic regression, support vector machines, linear regression) with tensor factorization structure on coefficients/predictors
These algorithms are useful for when the data inherently has "low-rank structured data".

For more information, refer to the paper submitted to **Rutgers Undergraduate Aresty Journal**: [Paper](https://0dd37264-afc1-4a24-9ba5-e79720bc9ea4.filesusr.com/ugd/f056cc_32f3263618ca48b198a532343ec17ddd.pdf)


## Contact and Contributions
This package is in alpha, so bug reports regarding implementation and etc. are highly welcomed. More information about tutorials can also be requested and is highly encouraged!

Contact: soominkwon0402@gmail.com


## Dependencies
The algorithms CP-SVM, CP-LogisticRegression, and CP-LinearRegression have the following dependencies:

* Python 3.x
* Numpy 1.17.2
* Scipy 1.3.1
* TensorLy


## Programs
The following is a list of which algorithms correspond to which Python script:

* cp_linear_svm.py - CP-LinearSVM
* cp_logistic_regression.py - CP-LogisticRegression
* cp_linear_regression.py - CP-LinearRegression
* traditional_ml.py - Traditional vectorized ML algorithms (Linear SVM, Logistic Regression, Linear Regression)
* generate_data.py - generates synthetic data
 
 
## Tutorial
This tutorial can be found in example.py:

```
import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_data
from cp_linear_svm import CP_LinearSVM
from cp_linear_regression import CP_LinearRegression
from cp_logistic_regression import CP_LogisticRegression
from traditional_ml import traditionalML

# initialize variables
sample_size = 1500
variance = 1.5
CP_rank = 2

# generate data
true_pred, X, Y_hard, Y_soft = generate_data(sample_size, variance)


# training CP SVM model
CP_SVM = CP_LinearSVM(CP_rank=CP_rank) # instantiating class
svm_dict = CP_SVM.fit(X, Y_hard) # fitting data (X, y)
cp_svm_reconstructed = CP_SVM.reconstructTensor(solved_dict=svm_dict, tensor_n_dim=2) # reconstruct tensor


# training traditional SVM model
vec_SVM = traditionalML(algorithm='Linear SVM') # instantiating class
vec_X = np.reshape(X, (sample_size, -1)) # vectorizing data
vec_svm_params = vec_SVM.fit(vec_X, Y_hard)
vec_svm_reconstructed = np.reshape(vec_svm_params, (cp_svm_reconstructed.shape))


# comparing CP SVM to traditional SVM model
cp_mse_val, cp_cos_dis, cp_recon_err = CP_SVM.evaluateMetrics(cp_svm_reconstructed, true_pred) # evaluate metrics
vec_mse_val, vec_cos_dis, vec_recon_err = CP_SVM.evaluateMetrics(vec_svm_reconstructed, true_pred)

print('CP MSE:', cp_mse_val)
print('Traditional MSE:', vec_mse_val, '\n')

print('CP Cosine Distance:', cp_cos_dis)
print('Traditional Cosine Distance:', vec_cos_dis, '\n')

print('CP Reconstruction Error:', cp_recon_err)
print('Traditional Reconstruction Error:', vec_recon_err, '\n')


# visualizing the solved parameters
plt.imshow(true_pred, cmap='gray')
plt.title('True Predictor')
plt.show()

plt.imshow(cp_svm_reconstructed, cmap='gray')
plt.title('CP Solved Predictor')
plt.show()

plt.imshow(vec_svm_reconstructed, cmap='gray')
plt.title('Traditional Solved Predictor')
plt.show()
```
