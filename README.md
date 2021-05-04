# Mobile Sensing and Robotics - 2020/21

## Instructors
* Cyrill Stachniss
* Nived Chebrolu

## Video Playlist
* [MSR Part 1](https://www.youtube.com/watch?v=5KZpWAe9hSk&list=PLgnQpQtFTOGQEn33QDVGJpiZLi-SlL7vA)
* [MSR Part 2](https://www.youtube.com/watch?v=mQvKhmWagB4&list=PLgnQpQtFTOGQh_J16IMwDlji18SWQ2PZ6)

**SLAM** - Simultaneous Localization and Mapping. 

Where am I -><- What does the world look like.

## Two fundamental questions in Robotics World

1. **State estimation** - What is the state of the world? Typically, *Sensor data* required to build the map of the environment.
    * **Estimating Semantics** - Understanding what we see 
    * **Estimating Geometry** - Understanding what the world looks like

    Both can be fused too.

2. **Action Selection** - Which action should state execute? Typically, *physical navigation*

Both impacts each other.

### Solved using probabilistic approaches because
- Uncertainty both in robot motion and observations
- Use of probability theory to explicitly represent the uncertainty 

## Notebooks
* [Introduction](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Introduction.ipynb)
* [Bayes Filter](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Bayes_Filter.ipynb)
* [Occupancy Grid Maps](https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Occupancy_Grid_Maps.ipynb)
