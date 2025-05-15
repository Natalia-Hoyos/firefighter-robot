# firefighter-robot (Project in progress)
This project is a self-assigned robotics challenge where the goal is to simulate a firefighting robot capable of detecting, ranking, and responding to simulated fires using computer vision and basic navigation.

🔥 Project Overview

The robot operates in a predefined environment where 2 to 4 red beacons represent fire sources. Each beacon varies in size, representing different fire intensities. The robot uses its camera to detect these beacons and determine their size in the image, ranking them from most to least intense.

Once the ranking is complete, the robot will navigate to each beacon location, starting from the most intense fire to the least, using odometry and a preloaded map of known beacon locations.

🎯 Project Goals

Detect red beacons using computer vision (OpenCV).

Estimate beacon size from the image to determine fire intensity.

Sort beacon targets based on estimated size.

Navigate to each beacon location using odometry data.

Simulate a fire extinguishing action (e.g., LED blink or buzzer).

⚙️ Technologies Used

Python

OpenCV (for image processing)

PenguinPi5

Basic odometry (dead-reckoning) for navigation

📍 Assumptions

All beacon (fire) locations are known and stored ahead of time.

The robot uses a single forward-facing camera.

The environment is static (beacons don't move).
