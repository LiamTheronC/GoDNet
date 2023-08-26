# Goal Driven Multi-Modal Motion Prediction

## Introduction

In this work, we introduce a novel framework known as the **Goal-Driven Motion Prediction Network** (**GoDNet**). GoDNet is engineered to excel in autonomous driving motion prediction tasks by learning from short-term historical observations and vectorized map information. To harness both spatial and temporal dimensions, the framework employs 1D-CNN on historical trajectories, as well as graph convolution on a sparse lane graph. To incorporate interaction-aware information, we introduce specialized cross-attention modules that facilitate nuanced interactions between the map and the actors, as well as among the actors themselves. Additionally, we have designed a goal-driven unit to enhance the model's capacity for long-term predictions across varying scales. Our proposed approach sets a new benchmark, achieving state-of-the-art performance in terms of minimum Final Displacement Error (**minFDE**) and minimum Average Displacement Error (**minADE**) on the **Waymo Open Dataset**.

---

## Architecture
![Architecture](pictures/Architecture.png)

---

## Table of Contents
* [Introduction]()
* [Architecture]()
* [Get started](https://github.com/LiamTheronC/waymo_motion_prediction#installation)
* [How to use](https://github.com/LiamTheronC/waymo_motion_prediction#usage)
* [License](https://github.com/LiamTheronC/waymo_motion_prediction/blob/main/README.md#license)

---

## Results
![result](pictures/result.png)

---
 
## License
  
The work is released under the MIT license.
  
---
