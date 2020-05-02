**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:[https://colab.research.google.com/github/abksyed/EVA4/blob/master/S6/EVA4S6.ipynb](https://colab.research.google.com/github/abksyed/EVA4/blob/master/S6/EVA4S6.ipynb)

To run the model, need to upload all the necessary packagesto the colab directory. The packages can be found in the S6 folder of EVA4 repo.

**OBJECTIVE:**

Run your model for 25 epochs for each:

  1. without L1/L2 with BN
  2. without L1/L2 with GBN
  3. with L1 with BN
  4. with L1 with GBN
  5. with L2 with BN
  6. with L2 with GBN
  7. with L1 and L2 with BN
  8. with L1 and L2 with GBN

**Finding the Optimal Lambda Values for L1 and L2 Regularization:**

_For L1:_

Values checked for : 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05

![L1Lambda.png](https://github.com/abksyed/EVA4/blob/master/S6/Images/L1Lambda.PNG)

The value of 1e-5, gave the best validation accuracy.

_For L2:_

Values checked for : 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05

![L2Lambda.png](https://github.com/abksyed/EVA4/blob/master/S6/Images/L2Lambda.PNG)

Here too the Lambda value of 1e-5 gave the best validation accuracy.

Hence, in the model we used the lambda value of 1e-5 (0.000001) for both L1 and L2.

**Validation Accuracy curves for all 8 task:**
![ValidationAccuracy.png](https://github.com/abksyed/EVA4/blob/master/S6/Images/ValidationAccuracy.png)

**Validation Loss curves for all 8 task:**
![ValidationLoss.png](https://github.com/abksyed/EVA4/blob/master/S6/Images/ValidationLoss.png)
