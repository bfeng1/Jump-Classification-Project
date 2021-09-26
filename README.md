- 👋 Hi, I’m Bin Feng
- 👀 I’m interested in data science, artificial intellegence, machine learning models
- 🌱 I’m currently learning electrical and computer engineer in university of memphis
- 💞️ I’m looking to collaborate on 3D marker-less gait analysis system
- 📫 How to reach me: binfengmemphis@gmail.com

## Main purpose of this project:
Use gathed kinematical infomation duing vertical jump to determine if the jump style is good or bad.

## Dataset:
The training dataset contains total 46 jump cycles(25 normal jump/ 12 inner jump/ 9 outer jump). Each training jump cycle contains 135 timeseries points * 5 features. 
### csv_files\lable.csv: 
Labels for each jump(good or bad)
### csv_files\good*.csv: 
Normal jump cycles examples. It contains timeseries data with five different features for multiple jump cycles, such as left knee angle, left ankle angle, right knee angle, right ankle angle and left knee angle ratio(left knee angle / left ankle angle)
### csv_files\inner*.csv: 
Bad jump cycles with knee going inner side. It contains timeseries data with five different features for multiple jump cycles, such as left knee angle, left ankle angle, right knee angle, right ankle angle and left knee angle ratio(left knee angle / left ankle angle)
### csv_files\outer*.csv: 
Bad jump cycles with knee going outer side. It contains timeseries data with five different features for multiple jump cycles, such as left knee angle, left ankle angle, right knee angle, right ankle angle and left knee angle ratio(left knee angle / left ankle angle)
### list_info.txt
Manually selected time range for each jump cycle in a multiple-jump-cycles example

## Project method 
### Preprocessing
* Each jump cycle do not have the same time points because each jump cycle has different height. During preprocssing phrase, I resampled each jump cycle, so each jump cycle contains 135 data points. 
* Since each jump cycle data are timeseries data and it is not perfectly aligned with each other, it is important to align all jump cycles before training to aviod noisy. That is why I choose to use DTW (Dynamic Time Warping) to preprocess all input data. 
### Training
* Used KNN (K nearest neighbor) with DTW to classify the good and bad jump cycles using "left knee angle ratio" feature.

## Results
* the accuracy of the classifier: 76.0%
* false positive rate: 0.2926190476190476
* false negative rate: 0.1830952380952381

## Future work
* The training dataset only contains total 46 jump cycles(25 normal jump/ 12 inner jump/ 9 outer jump). So we need to collect more jump data for better accuracy.
* Add different classifier models to compare the accuracy

## Credit
K Nearest Neighbors & Dynamic Time Warping. The source code for the index construction and search is available at https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping.
<!---
bfeng1/bfeng1 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
