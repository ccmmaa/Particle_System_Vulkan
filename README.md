# Particle_System_Vulkan

This project is two particles systems made using the API Vulkan

My development environment was Windows 7 and I was using Visual Studio 2015.

To run this program, you will need the correct drivers so that your graphics card will run with Vulkan, and the Vulkan SDK which can be downloaded from here: <https://vulkan.lunarg.com/sdk/home#windows>

If more help is needed please look at this webpage which describes setting up the environment: <https://vulkan-tutorial.com/Development_environment>

Once the program is running, a window will pop up asking what particle system you want to use. The two choices are fire and sparks. If the input of the user is incorrect, the program will default to fire. The particle system will continue running until you exit out of it.

Both depth buffering and texture mapping have been implemented in this program.

All of my code is in main.cpp, shader.vert, and shader.frag

My sources are:
<https://vulkan-tutorial.com/Introduction>
<http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/particles-instancing/>
