# Third-party software, data, and research

This thesis repository integrates external software and published research.
The project-specific contribution is the weather-fault adaptation, experiment
orchestration, and evaluation integration described in the root README.

## KISS-ICP

The `kiss_icp_modifications/` directories contain adapted KISS-ICP source code.
Upstream: <https://github.com/PRBonn/kiss-icp>

MIT License

Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
Stachniss.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## EVO

Trajectory evaluation uses the external
[EVO](https://github.com/MichaelGrupp/evo) package, distributed under GPL-3.0.
EVO source code is not vendored here.

## KITTI

Experiments expect the separately downloaded
[KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
KITTI data is not redistributed by this repository and remains subject to its
own terms.

## Fog-model parameterization

The probabilistic fog mechanisms and fitted parameters are adapted from:

Sven Teufel, Georg Volk, Alexander von Bernuth, and Oliver Bringmann,
“Simulating Realistic Rain, Snow, and Fog Variations For Comprehensive
Performance Characterization of LiDAR Perception,” *2022 IEEE 95th Vehicular
Technology Conference (VTC2022-Spring)*, pp. 1–7.
<https://doi.org/10.1109/VTC2022-Spring54318.2022.9860868>
