# Mobile Sensing and Robotics - 2020/21

## Instructor
* Cyrill Stachniss

## Resources
* [MSR Part 1 Videos](https://www.youtube.com/watch?v=5KZpWAe9hSk&list=PLgnQpQtFTOGQEn33QDVGJpiZLi-SlL7vA)
* [MSR Part 2 Videos](https://www.youtube.com/watch?v=mQvKhmWagB4&list=PLgnQpQtFTOGQh_J16IMwDlji18SWQ2PZ6)
* [MSR Part 2 Slides](https://drive.google.com/drive/folders/1oBOP26q2ta88ITBtsWmWVNZ4si9hkHaF?usp=sharing)

## SLAM - Simultaneous Localization and Mapping

**Localization** - Estimating robot's pose

**Mapping** - Task of modelling the environment

![SLAM Example](https://www.societyofrobots.com/images/sensors_IRSLAM.gif)

*Where am I --><-- What does the world look like*

*Full SLAM estimates the entire navigation path whereas, Online SLAM seeks to recover only the most recent pose*

## Two fundamental questions in Mobile Robotics 

1. **State estimation** - What is the state of the world? Typically, *Sensor data* required to build the map of the environment.
    * **Estimating Semantics** - Understanding what we see 
    * **Estimating Geometry** - Understanding what the world looks like

    Both can be fused.

2. **Action Selection** - Which action should state execute? Typically, *physical navigation*

These influence each other. It's a chicken-and-egg problem.

## Mathematical description of the problem
**Given**
1. The robot's controls ![u_{1:T}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+u_%7B1%3AT%7D%0A)
2. Sensor Observations ![z_{1:T}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z_%7B1%3AT%7D%0A)

**Wanted**
1. Map of the environment *m*
2. Path of the robot ![x_{0:T}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+x_%7B0%3AT%7D%0A)

**We estimate the probability distribution** ![\begin{align*} p(x_{0:T}, m \, | \, z_{1:T}, u_{1:T}) \end{align*}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ap%28x_%7B0%3AT%7D%2C+m+%5C%2C+%7C+%5C%2C+z_%7B1%3AT%7D%2C+u_%7B1%3AT%7D%29%0A%5Cend%7Balign%2A%7D%0A)

### Solved using probabilistic approaches because
- Uncertainty both in robot motion and observations
- Use of probability theory to explicitly represent the uncertainty 

## Why is SLAM a Hard problem to solve?
1. Robot path and map both are unknown
2. Map and pose estimates are correlated
3. Mapping between observations and the map is unknown
4. issue of divergence - picking wrong data associations can have catastrophic consequences

## Paradigms for solving SLAM problems
* Kalman Filter
* Particle Filter
* Graph based representations - *Focus on this*

## Notebooks (Exercises with Notes)
* [Introduction](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Introduction.ipynb)
* [Bayes Filter](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Bayes_Filter.ipynb)
* [Occupancy Grid Maps](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Occupancy_Grid_Maps.ipynb)
* [Iterative Closest Point]()
* [Graph-based SLAM]()
* [Visual Features and RANSAC]()
* [Camera Intrinsics and Extrinsics]()

