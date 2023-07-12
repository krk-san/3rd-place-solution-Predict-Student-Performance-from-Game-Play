# Predict-Student-Performance-from-Game-Play-3rd-place-solution

## About this repository

This repository deals with my part of the 3rd place solution of [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play).

- Processing code for the raw log data

- The training code for the model I created

## How to run

- Processing code for the raw log data

  1. Download raw log datasets from https://fielddaylab.wisc.edu/opengamedata/.
 
  2. Unzip the datasets and place them in an appropriate location.
 
  3. Rewrite I/O path in notebooks/generate_additional_datasets.ipynb appropriately and execute it.

- The training code for the model I created

  1. Prepare a config yaml file. An example is config/sample.yml.
 
  2. Execute the following commands on the terminal.
 
        ```python src/train.py --cfg {config filepath}```


## Related Documents

- Overall explanation of the 3rd place solution.

  https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420235
  
- My part explanation of the 3rd place solution.
  
  https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420274
