# AREP-RSIs
An one-stop platform for conducting adversarial defenses and conveniently evaluating adversarial robustness of DNN-based visual recognition system.
Users can operate just on AREP-RSIs to perform a complete robustness evaluation with all necessary procedures, including training, adversarial attacks, tests for recognition accuracy, proactive defense and reactive defense. AREP-RSIs can be deployed on the edge devices like UAVs and connected with cameras for real-time recognition as well. Equipped with various network architectures, several training paradigms and classical defense methods, to the best of our knowledge, AREP-RSIs is the first platform for adversarial robustness improvements and evaluations in the remote sensing field. 
The graphic interface of this platform is designed with PyQt and built upon necessary libraries like Pytorch, Adversarial Robustness Toolbox (ART), OpenCV  and Scikit-learn.

# Database
The dataset can be placed in the folder of Data. For example, if the scene recognition dataset is UCM, the address for train set is "./Data/UCM/train" and "./Data/UCM/val" for validation set.

# Running Environments
PyQt 5.15.4, pytorch 1.8.0, torchvision 0.9.0, numpy 1.19.5, opencv-python 4.5.3, matplotlib 3.5.4, adversarial-robustness-toolbox 1.12.0, openpyxl 3.0.9, scipy 1.5.4, torchcam 0.3.2, tqdm 4.62.3 and so on (See in codes)

# Functions and Corresponding Python Files:
1. Training: ui_CNN_train.py;
2. Single image test: ui_adv_samples_classification.py;
3. Batch images test: ui_CNN_Batch_Dataset_test.py;
4. Craft adversarial examples: ui_adv_samples_create.py;
5. adversarial examples of MSTAR display: ui_asc_attack.py;
6. train detectors for reactive defense: ui_adv_detector_train.py;
7. use the trained detector: ui_adv_detector.py;
8. proactive defense to generate robust models: ui_CNN_jiagu_singlesource.py and ui_CNN_jiagu_multisource.py
9. ......
