"""
---
created: 2022-11-30-10-26-00
---

# PostPro V0R0
tags: tags: #computational #constitutive #numeric

# Progress
- Lo script serve ad analizzare più files di risultati di simulazione FEM1elem


# log
- 30/11/22
    - creato con parte di codice da Test V2R0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ----------------------------------------------------------------
# List the simulations you want to compare
# ----------------------------------------------------------------

simulations = ["test_2022-12-02-16-27-40"]

# ----------------------------------------------------------------
# Set plot colors and figure size and properties
# ----------------------------------------------------------------

n = len(simulations)
# change the colormap by changing the name after cm
# colors = plt.cm.bynary(np.linspace(0.3, 1, n))
colors = plt.cm.rainbow(np.linspace(0.3, 1, n))

figdpi = 600
axlebelfont = 8 #[pt]
axtickfont = axlebelfont - 1 #[pt]
width = 12 #[cm]
heigth = 9 #[cm]

fig = plt.figure(figsize=(width/2.54, heigth/2.54), dpi=figdpi)

matplotlib.rc('xtick', labelsize=axtickfont) 
matplotlib.rc('ytick', labelsize=axtickfont) 

# ----------------------------------------------------------------
# Loop through the files 
# ----------------------------------------------------------------
for i, c in zip(range(n), colors):
    simulation = simulations[i] + ".dat"
    with open(simulation, "r") as f:
        lines = f.readlines()
    metadata = [line for line in lines if line.startswith("#")]
    for line in metadata:
        if line.find("Material")> 0:
            model = line[2:]
        if line.find("range")> 0:
            Trange = line[2:]
        if line.find("testID")> 0:
            testID = line[2:]
    data = np.loadtxt(simulation, comments='#', delimiter=",")
    E11 = data[:,0]
    S11 = data[:,1]
    
    # plot results
    plt.plot(-E11,-S11/1e6,'-o', 
             color=c,
             linewidth=1,
             markersize=2,
             label= testID + model.replace(" ","\n") + Trange )

plt.xlabel("E11 [mm/mm]", fontsize=axlebelfont )
plt.ylabel("S11 [MPa]", fontsize=axlebelfont )
plt.xlim([0,0.013])
plt.ylim([0,55])
plt.legend( 
           # title=model.replace(" ","\n"),
           title_fontsize=axtickfont,
           fontsize=axtickfont,
           # markerscale=.8,
           frameon=False,
                        # (x, y, width, height)
           # bbox_to_anchor=(1, .5, 0.2, 0.2),
           loc='upper left') 
plt.title("S11-E11", fontsize=axlebelfont)
plt.grid()

DEBUG = False

# Save the plot with unique identifier name
if DEBUG != True:
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("SE_" + dt_string + ".png", dpi=figdpi, bbox_inches='tight')
else:
    plt.show()    
            
            
            
            
            
            
            
            
            
            
            
            
            
            

















