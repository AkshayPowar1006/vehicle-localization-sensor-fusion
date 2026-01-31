# Vehicle Localization Using Multi-GNSS Sensor Fusion

## Overview
Accurate localization is a critical requirement for autonomous navigation. Autonomous vehicles must continuously estimate their pose (position and orientation) in the environment to navigate safely and reliably. Global Navigation Satellite System (GNSS) sensors play a key role in this process by providing real-time positioning information such as latitude, longitude, and altitude.

Recent advances in GNSS technology, including Real-Time Kinematic (RTK) systems, enable centimeter-level localization accuracy. However, these systems are often expensive. At the same time, alternative low-cost solutions—such as smartphones with built-in GNSS and IMU fusion, or affordable GNSS modules like the u-blox NEO-6M combined with microcontrollers (e.g., Arduino)—offer promising accuracy at a fraction of the cost.

This project explores and evaluates these alternatives for vehicle localization.

## Project Objective
The primary objective of this research is to:
- Compare raw GNSS measurements collected simultaneously from smartphones and an Arduino-based multi-GNSS setup
- Benchmark their localization accuracy against a GNSS RTK reference system
- Evaluate the effectiveness of sensor fusion and state estimation techniques, particularly Kalman Filter–based approaches, for improving localization accuracy

## Methodology
An experimental setup was developed using multiple GNSS localization systems mounted on a vehicle. The collected data was processed using state estimation algorithms, including:
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)

These algorithms fuse measurements from multiple GNSS sensors and apply a constant-acceleration motion model to estimate the vehicle’s state. The EKF was additionally used to explore the feasibility of estimating orientation alongside position.

## Key Findings
- Multi-GNSS fusion significantly improves localization accuracy compared to using a single GNSS sensor
- Simulation results show that the Extended Kalman Filter can estimate additional state variables, such as vehicle orientation
- Real-world experiments confirm the accuracy improvement achieved through multiple GNSS sensors
- Orientation estimation observed in simulation does not consistently transfer to real-world data, likely due to sensor noise, multipath effects, and signal obstructions

## Conclusion
This work demonstrates that low-cost GNSS hardware, when combined with sensor fusion techniques, can achieve improved localization accuracy suitable for autonomous navigation research. The discrepancy between simulated and real-world results—particularly in vehicle orientation estimation—highlights the need for further investigation into sensor modeling, calibration, and fusion strategies.
