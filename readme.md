# Starchlight MFC model
In the context of the 2022 iGEM competition, and as part of the Starchlight project, I modeled the main phenomena underlying a special kind of bacterial fuel cell. The model was then tuned according to experimental data we gathered in the lab to predict the output of an industrial-scale device. The simulation also allowed us to further identify the main bottlenecks of a large-scale device.

![image](https://user-images.githubusercontent.com/66411147/233458729-d9c531db-a803-433a-8cfd-7b342f6ed188.png)


Modify **params.py** to change the simulation's parameters.  
Modify **molecules.py** to alter the particle interactions.
Run **StarchLight_Simulator.py** to run the simulation.
## The theory behind the model

Starch gives out one molecule of glucose per interaction with the amylase (1). Theoretically, 1 molecule of glucose can generate 2 molecules of lactate via glycolysis and the reaction carried out by the LDH (2). In S.oneidensis, the conversion of lactate to formate can produce 4 electrons, and the conversion of lactate into citric acid can produce 12 electrons. It is hypothesized that 40% of the lactate is used to create formate, and 30% goes into the tricarboxylic acid cycle (TCA). Thus, on average, one molecule of lactate generates 5 electrons in S.oneidensis (3).The intensity of the generated electric current can be deduced from the results of this simulation.

Starch-(n) + Amylase → Starch-(n-1) + Glucose
Glucose + E.coli → 2 Lactate + E.coli
Lactate + S.oneidensis → 5 electrons
The model can be finetuned by setting different values for each interaction, by modifying the number of each component and their relative speeds. The purpose of this model is to help us understand the bottlenecks of our design by tuning its parameters to fit experimental data. Moreover, once the model is close enough to experimental data, we could use it to predict the efficiency of our upscaled device.

![simulation](https://user-images.githubusercontent.com/66411147/180600178-a502323e-3b8a-4552-b7d1-e8744cc0e279.gif)

## How the model works
### Initialization
The simulation is initialized by randomly placing starting particles (i.e S.oneidensis, E.coli, Starch, and amylases) in the working area.
### Particle movement
At each time step and for each dimension of the 3D space (x, y, z), each particle randomly moves at a predetermined speed, meaning a certain distance at each time step.

![image](https://user-images.githubusercontent.com/66411147/233457357-65aed868-da60-4f81-b0f2-3d4e33de66a3.png)
### Area of interaction
Each particle has a predetermined area of interaction and interaction probability. If a particle enters the area of interaction of another particle which it can interact with, they will interact as described in the theory behind the model.

![image](https://user-images.githubusercontent.com/66411147/233457546-efddf5e5-d5e0-4dc9-9202-bbe818039bbc.png)
### Periodic boundary conditions
To prevent particles from stacking against the battery walls, we designed an “infinite box” adapted from the periodic boundary conditions used in gas simulations. It teleports particles as described by the figure below. Another advantage of this method is the improvement of the scalability of the model. In theory, the results of a box can be propagated to predict the behavior of a larger box.

![image](https://user-images.githubusercontent.com/66411147/233457699-7be89686-cd70-4ff4-8896-5a64fa38d391.png)
### Tunable parameters
In order to finetune our model, these are the parameters we could modify:

* Working area size
* Particles’ speeds
* Particles’ area of interaction
* Particles interaction probability
* Duration of the simulation
* Amount of each particle at initialization
