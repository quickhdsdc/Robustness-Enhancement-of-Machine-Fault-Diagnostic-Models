# Robustness-Enhancement-of-Machine-Fault-Diagnostic-Models
## Introduction
This is a repository of our paper "Robustness Enhancement of Machine Fault Diagnostic Models for Railway Applications through Data Augmentation". 
A preprint version can be found at https://www.researchgate.net/publication/344119566_REVISED--Robustness_enhancement_of_machine_fault_diagnostic_models_for_railway_applications_through_data_augmentation.

The used Tensorflow version is 1.14.0. Spyder is used for programming and run the scripts.

Three augmentation methods have been implemented: C-DCGAN, SIM-GAN and our MBS-FWFSA.
Exp_mmd.py measures MMD under different condition variations. ExpA, ExpB and ExpC are the experiments to test the ResNet with different preprocessing methods under different condition variations.  ExpA_aug, ExpB_aug and ExpC_aug are the experiments to test the augmentation methods.

The experiment data is shared at https://depositonce.tu-berlin.de/handle/11303/13269. The data is in the format of python numpy. It will be transfered to some common formats like .csv or .mat in the future. The data includes the raw vibration data at axlebox level for the measurement at four different wagons in different operating speed. It also includes the processed data in the fft, envelope and scaled averaged cwt spectrum, and the extracted statistical features. Furthermore, the synthetic data generated by data augmentation techniques and the pure simulation data are included. More details can be found in our paper.
