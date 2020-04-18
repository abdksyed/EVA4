Task:
---------------------------------------------------------------
1) To acheive 99.4% validation accuracy in MNIST Dataset.
2) To use less than 20k Parameters
3) To converge in less than 20 Epochs
4) Without the use of fully connected layer

Results:
----------------------------------------------------------------

Accuracy Acheived : 99.55%

Parameters Used: 19,122

epochs Used: 20


================================================================

        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
         Dropout2d-3           [-1, 16, 28, 28]               0
            Conv2d-4           [-1, 32, 28, 28]           4,640
       BatchNorm2d-5           [-1, 32, 28, 28]              64
         Dropout2d-6           [-1, 32, 28, 28]               0
         MaxPool2d-7           [-1, 32, 14, 14]               0
            Conv2d-8            [-1, 8, 14, 14]             264
            Conv2d-9           [-1, 16, 14, 14]           1,168
      BatchNorm2d-10           [-1, 16, 14, 14]              32
        Dropout2d-11           [-1, 16, 14, 14]               0
           Conv2d-12           [-1, 32, 14, 14]           4,640
      BatchNorm2d-13           [-1, 32, 14, 14]              64
        Dropout2d-14           [-1, 32, 14, 14]               0
        MaxPool2d-15             [-1, 32, 7, 7]               0
           Conv2d-16             [-1, 16, 7, 7]             528
           Conv2d-17             [-1, 32, 5, 5]           4,640
        Dropout2d-18             [-1, 32, 5, 5]               0
           Conv2d-19             [-1, 10, 3, 3]           2,890
        AvgPool2d-20             [-1, 10, 1, 1]               0

Total params: 19,122
Trainable params: 19,122
Non-trainable params: 0

================================================================


Input size (MB): 0.00
Forward/backward pass size (MB): 1.17
Params size (MB): 0.07
Estimated Total Size (MB): 1.24

================================================================

Log
------

epoch:1
loss=0.08413107693195343 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 61.17it/s]
Test set: Average loss: 0.0613, Accuracy: 9775/10000 (97.75%)

epoch:2
loss=0.07872176170349121 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 62.60it/s]
Test set: Average loss: 0.0426, Accuracy: 9852/10000 (98.52%)

epoch:3
loss=0.09042373299598694 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 62.80it/s]
Test set: Average loss: 0.0303, Accuracy: 9905/10000 (99.05%)

epoch:4
loss=0.08693313598632812 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 62.20it/s]
Test set: Average loss: 0.0242, Accuracy: 9922/10000 (99.22%)

epoch:5
loss=0.007796168327331543 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 62.57it/s]
Test set: Average loss: 0.0279, Accuracy: 9914/10000 (99.14%)

epoch:6
loss=0.031596213579177856 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 62.01it/s]
Test set: Average loss: 0.0238, Accuracy: 9918/10000 (99.18%)

epoch:7
loss=0.010419100522994995 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 63.54it/s]
Test set: Average loss: 0.0222, Accuracy: 9926/10000 (99.26%)

epoch:8
loss=0.01521909236907959 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.17it/s]
Test set: Average loss: 0.0253, Accuracy: 9924/10000 (99.24%)

epoch:9
loss=0.027932628989219666 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 63.73it/s]
Test set: Average loss: 0.0249, Accuracy: 9924/10000 (99.24%)

epoch:10
loss=0.06742291152477264 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 63.33it/s]
Test set: Average loss: 0.0173, Accuracy: 9941/10000 (99.41%)

epoch:11
loss=0.0070523470640182495 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.79it/s]
Test set: Average loss: 0.0186, Accuracy: 9934/10000 (99.34%)

epoch:12
loss=0.010588020086288452 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.34it/s]
Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99.39%)

epoch:13
loss=0.006601244211196899 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.27it/s]
Test set: Average loss: 0.0184, Accuracy: 9947/10000 (99.47%)

epoch:14
loss=0.005680173635482788 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.74it/s]
Test set: Average loss: 0.0164, Accuracy: 9951/10000 (99.51%)

epoch:15
loss=0.024711519479751587 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.37it/s]
Test set: Average loss: 0.0206, Accuracy: 9945/10000 (99.45%)

epoch:16
loss=0.005393654108047485 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.35it/s]
Test set: Average loss: 0.0159, Accuracy: 9950/10000 (99.50%)

epoch:17
loss=0.0007447898387908936 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.09it/s]
Test set: Average loss: 0.0169, Accuracy: 9955/10000 (99.55%)

epoch:18
loss=0.025433897972106934 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.57it/s]
Test set: Average loss: 0.0171, Accuracy: 9944/10000 (99.44%)

epoch:19
loss=0.010704070329666138 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.05it/s]
Test set: Average loss: 0.0172, Accuracy: 9948/10000 (99.48%)

epoch:20
loss=0.0003122687339782715 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 66.00it/s]
Test set: Average loss: 0.0168, Accuracy: 9952/10000 (99.52%)


Best Accuracy in 20 epochs = 99.55% (@epoch:17)
---
