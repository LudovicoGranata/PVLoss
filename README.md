# PVLoss
Physical Violation Loss - PyTorch Implementation 

![Figure_1](https://user-images.githubusercontent.com/18251575/171359784-f0c631fa-dcc6-4a6b-b8f1-5012b5f835d0.png)

## Introduction
This loss has been designed having in mind a 6D pose estimation setting: rigid objects should not penetrate each other so this loss measure how much this happens for a pair of them. Each object is represented by its 3D bounding box and not by the CAD model to have faster computation, even though it could be less precise. The algorithm is based on the Separating axis theorem that will be introduced in the following.

## Separating axis theorem

The separating axis theorem states that two convex objects do not overlap if there exists a line (axis) onto which the two objects' projections do not overlap. To test this in 3D space we have to test both face's normals and also the additional axes taken from the cross-products of pair of edges, one taken from each objects.

https://en.wikipedia.org/wiki/Hyperplane_separation_theorem#:~:text=The%20separating%20axis%20theorem%20(SAT,convex%20solids%20intersect%20or%20not


## Usage
This is just a python class, copy it in your project and use it. You can also directly run the pv_loss.py file to test it. 

