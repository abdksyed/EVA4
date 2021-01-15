### First Iteration (**_Basic Model_**)

[Colab Link](https://github.com/abksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter01.ipynb)

**Target**

- Less than 10000 parameters
- Less than 15 epochs

#### Results

- Parameters Used= **7472**
- Train Accuracy = **100%**
- Test Accuracy = **99.1%**

**Analysis**

- Clear Over-fitting of model, as Train accuracy has reached 100% and can&#39;t be increase further, so no more scope for increase in Test Accuracy
- Number of Parameters used are nearly 7500, so more 2500 parameters can be used if required.
- Need to reduce to Overfitting

### Second Iteration (**_Model + BatchNorm_**)

[Colab Link](https://github.com/abksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter02.ipynb)

**Target achieved in Last Iteration**

- Parameters Used:  **7474**
- Test Accuracy:  **99.1%**
- Training Accuracy:  **100%**
- Epochs Used:  **15**

**Target for this Iteration**

- Improve the Model
- Add Batch Normalization to increase model accuracy.

**Results**

- Parameters Used= **7408**
- Train Accuracy = **100%**
- Test Accuracy = **99.27%**

**Analysis**

- Over-fitting still exists, as Train accuracy has reached 100% and can&#39;t be increase further, so no more scope for increase in Test Accuracy
- But the test accuracy has increased from 99.1 in previous iteration(without BatchNorm) to 99.27(with BatchNorm)
- Number of Parameters used are 7624, so more 2400 parameters can be used if required.
- Need to reduce to Overfitting

### Third Iteration (**_Model + BatchNorm + DropOut_**)

[Colab Link](https://github.com/abksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter03.ipynb)

**Target achieved in Last Iteration**

- Parameters Used:  **7624**
- Test Accuracy:  **99.27%**
- Training Accuracy:  **100%**
- Epochs Used:  **15**

**Target for this Iteration**

- Reduce Over-fitting
- Add Dropout to reduce overfitting.

**Results**

- Parameters Used= 7408
- Train Accuracy = 94.54%
- Test Accuracy = 99.22%

**Analysis**

- Over-fitting has been resolved, as we can inspect Train Accuracy is always lesser than Test accuracy.
- But the test accuracy has decreased from 99.27 in previous iteration(with BatchNorm) to 99.22(with BatchNor+DropOut)
- Number of Parameters used are 7408, so more 2500 parameters can be used if required.
- Need to improve accuracy.

### Fourth Iteration (**_Model + BatchNorm + RandomRotation_**)

[Colab Link](https://github.com/abksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter04.ipynb)

**Target achieved in Last Iteration**

- Parameters Used:  **7408**
- Test Accuracy:  **99.22%**
- Training Accuracy:  **95.54%**
- Epochs Used:  **15**

**Target for this Iteration**

- Improve Accuracy.
- Add RandomRotation of (-12, 12) degrees
- Add extra Layer after GAP

**Results**

- Parameters Used= 8464
- Train Accuracy = 99.18%
- Test Accuracy = 99.41%

**Analysis**

- Achieved 99.41% Test Accuracy at epoch:12
- BatchNorm and DropOut doesn&#39;t work well hand in hand. ([https://arxiv.org/abs/1801.05134](https://arxiv.org/abs/1801.05134))
- Test accuracy reaches 99.4 but isn&#39;t consistent.

### Fifth Iteration (**_Model + BatchNorm + RandomRotation + StepLR_**)

[Colab Link](https://github.com/abksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter05.ipynb)

**Target achieved in Last Iteration**

- Parameters Used:  **8464**
- Test Accuracy:  **99.41%**
- Training Accuracy:  **95.18%**
- Epochs Used:  **15**

**Target for this Iteration**

- Stabilize the accuracy.
- Achieve 99.4% accuracy for 3-5 epochs consistently.
- Use ScheduledLR

**Results**

- Parameters Used= 8464
- Train Accuracy = 99.37%
- Test Accuracy = 99.60%

**Analysis**

- Achieved 99.55% Test Accuracy at epoch:8
- StepLR has done wonders, keeping accuracy consistent.
- Test accuracy 99.55 (epoch 8)
- Test accuracy 99.60 (epoch 9)
- Test accuracy 99.57 (epoch 10)
- Test accuracy 99.55 (epoch 11)
- Test accuracy 99.55 (epoch 12)
- Test accuracy 99.57 (epoch 13)
- Test accuracy 99.60 (epoch 14)
- Test accuracy 99.58 (epoch 15)

### Sixth Iteration (**_Model + BatchNorm + DropOut + RandomRotation + StepLR_**)

[Colab Link](https://github.com/abdksyed/EVA4/blob/master/05_CodingFrenzy/EVA4_S5_Assignment_iter06.ipynb)

**Target achieved in Last Iteration**

- Parameters Used:  **8464**
- Test Accuracy:  **99.60%**
- Training Accuracy:  **99.37%**
- Epochs Used:  **15**

**Target for this Iteration**

- Use DropOut(3%), to increase challenge for Training
- Create gap between training and testing accuracy.

**Results**

- Parameters Used= 8464
- Train Accuracy = 98.83%
- Test Accuracy = 99.55%

**Analysis**

- As expected, training accuracy is decreased, but test accuracy remains almost same.
- The model can be further pushed or challenged even more by increasing DropOut percentage. But again, it&#39;s not better to use BatchNorm with DropOut.
