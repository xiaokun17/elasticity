We propose an implicit smoothed particle hydrodynamics fluid-elastic coupling approach that effectively reduces the instability issue for fluid-fluid, fluid-elastic and elastic-elastic coupling circumstances.

The code environment needs to support Taichi and Vulkan.

The demo shows the coupling of two elastic blocks with a Young's modulus of 500 KPa. A maximum density ratio of 60 can be achieved using our method. We use color to indicate the elastic force of the elastic block below. It can be seen that the color of particles is darker at the interface and extended down along the boundary. In contrast, the middle color is lighter, which shows that our method is more continuous and robust to force conduction.