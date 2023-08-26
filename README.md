# Goal Driven Multi-Modal Motion Prediction

In this work, new framework termed Goal Driven Motion Prediction Network(GoDNet) is presented, which learns from short term history observations and vecterised map information to support multiple autonomous driving motion prediction tasks. GoDNet exploits both spatial and temporal information by leveraging 1D-CNN on history trajectories and graph convolution on sparsed lane graph. To aggregate interaction aware information, the authors design cross-attention modules between map and actor or between actors them selves. Furthermore, a goal-driven unit is developed to enhance the ability of long-term prediction on various scales. The proposed approach achieves the state-of-the-artin terms of minFDE and minADE metric on the Waymo Open Dataset.
---

## Table of Contents
* [Get started](https://github.com/LiamTheronC/waymo_motion_prediction#installation)
* [How to use](https://github.com/LiamTheronC/waymo_motion_prediction#usage)
* [License](https://github.com/LiamTheronC/waymo_motion_prediction/blob/main/README.md#license)

---

## Architecture
![Architecture](pictures/Architecture.png)

 
---
 
## License
  
The work is released under the MIT license.
  
---
