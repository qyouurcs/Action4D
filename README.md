# Action4D

by [Quanzeng You](http://cs.rochester.edu/u/qyou/), [Hao Jiang](http://hao-jiang.net)

This is the code and data repository for our work ``Action4D: Online Action REcognition in the Crowd and Clutter'' published in CVPR 2019. 

### Introduction
We propose to tackle action recognition using a holistic 4D “scan” of a cluttered scene to include every detail
about the people and environment. In this work, we tackle a new
problem, i.e., recognizing multiple people’s actions in the
cluttered 4D representation.
Our method is invariant to
camera view angles, resistant to clutter and able to handle crowd.

#### Example snapshots of different actions

<img src="figures/actions/bend.jpg" alt="Bending" width="100"><img src="figures/actions/drink.jpg" alt="Drinking" width="100"><img src="figures/actions/lift.jpg" alt="Lifting" width="100"><img src="figures/actions/push.jpg" alt="Pushing/Pulling" width="100"><img src="figures/actions/squat.jpg" alt="Squatting" width="100"><img src="figures/actions/yawn.jpg" alt="yawning" width="100"><img src="figures/actions/call.jpg" alt="Calling" width="100"><img src="figures/actions/eat.jpg" alt="Eating" width="100">
<img src="figures/actions/open_drawer.jpg" alt="Opening Drawer" width="100"><img src="figures/actions/read.jpg" alt="Read" width="100"><img src="figures/actions/wave.jpg" alt="Waving" width="100"><img src="figures/actions/clap.jpg" alt="Clapping" width="100"><img src="figures/actions/kick.jpg" alt="Kicking" width="100"><img src="figures/actions/point.jpg" alt="Pointing" width="100"><img src="figures/actions/sit.jpg" alt="Sitting" width="100"><img src="figures/actions/web.jpg" alt="Browsing cell phone" width="100">

![Video](https://player.vimeo.com/video/111525512)

[![Audi R8](http://img.youtube.com/vi/KOxbO0EI4MA/0.jpg)](https://www.youtube.com/watch?v=KOxbO0EI4MA "Audi R8")
### Requirements
```
PyTorch >= 0.4.1
```
### Action4D Dataset

### Citation

    @InProceedings{action4d,
      author = {You, Quanzeng and Jiang, Hao},
      title = {Action4D: Online Action Recognition in the Crowd and Clutter},
      booktitle = {CVPR},
      year = {2019}
    }
