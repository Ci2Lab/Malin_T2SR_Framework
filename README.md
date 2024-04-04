# T2SR Framework

The T2SR Framework is a transformer-based deep learning framework for Super-Resolution (SR) in smart meter data. 

## Super-Resolution (SR)
SR is a technology aiming to generate high-resolution values from low-resolution measurements. SR was originally
introduced in the field of computer vision, where the goal was to transform low-
resolution images with rough characteristics into high-resolution images with
improved characteristics and enhanced features. A broad diversity of deep
learning techniques, as well as traditional techniques, have evolved for SR in
image processing. Recently, the potential of SR has also gained attention
in the field of power system applications, reflecting its flexibility across different
fields.

![image](https://github.com/Ci2Lab/Malin_T2SR_Framework/assets/42604421/61ef21a7-3b4c-4925-8911-46e92e26ef35)


## Architecture of the T2SR Framework

![image](https://github.com/Ci2Lab/Malin_T2SR_Framework/assets/42604421/30e7c9b6-1db7-435e-bc61-3e77c22ed620)



## Usage 
The repository houses a collection of Python files within the src directory, encompassing essential methods for data preprocessing, model implementation, and the training/evaluation pipeline.

File Structure
* **src**: This directory holds Python scripts for various tasks including data preprocessing, model implementation, and training/evaluation.
* **main.ipynb**: The main Notebook consolidates the entire workflow from data preprocessing to model evaluation, providing a comprehensive overview of the process. 

To effectively utilize the codebase:

1. **Data Requirements**: The code expects time series data with associated value and timestamp attributes for processing. Ensure the dataset conforms to this structure for seamless integration.

2. **Data Generation**: For demonstrative purposes, we include random example power consumption data. This aids users in understanding the expected format and characteristics of the input data.
