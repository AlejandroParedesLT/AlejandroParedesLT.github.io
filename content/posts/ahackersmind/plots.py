import matplotlib.pyplot as plt

layers = ["Legal", "Financial", "Technical", "Social"]
hack_potential = [0.3, 0.7, 0.9, 0.5]

plt.figure(figsize=(6,4))
plt.barh(layers, hack_potential, color='skyblue')
plt.xlabel("Hackability / Exploit Potential")
plt.title("System Layering and Exploit Potential")
plt.xlim(0,1)
plt.savefig("system_layers.png")
plt.show()

import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from([
    ("Banking Rule", "Synthetic Credit"),
    ("Tax Code", "Profit Shifting"),
    ("Accounting", "Structured Loss Harvesting")
])

pos = nx.spring_layout(G)
plt.figure(figsize=(6,4))
nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=2000, arrowstyle='->', arrowsize=20)
plt.title("Financial Arbitrage Hack Mapping")
plt.savefig("financial_arbitrage.png")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

time = np.arange(0,10,1)
resources_normal = np.minimum(time*2, 10)
resources_exploit = np.minimum(time*3, 10)

plt.plot(time, resources_normal, label="Normal")
plt.plot(time, resources_exploit, label="Exploit", linestyle="--")
plt.xlabel("Time (hours)")
plt.ylabel("Resource Availability")
plt.title("Resource Respawn Exploit Over Time")
plt.legend()
plt.savefig("game_exploit.png")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
x_adv = x + np.random.normal(0, 0.1, 50)
y_adv = y + np.random.normal(0, 0.1, 50)

plt.scatter(x, y, label="Normal Inputs")
plt.scatter(x_adv, y_adv, label="Adversarial Inputs", marker='x')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Minimal Adversarial Pathway")
plt.legend()
plt.savefig("ml_adversarial.png")
plt.show()


from matplotlib.sankey import Sankey
import matplotlib.pyplot as plt

Sankey(flows=[1, -0.4, -0.3, -0.3],
       labels=['Resources', 'Lobbying', 'Patents', 'Contracts'],
       orientations=[0, 1, 1, -1]).finish()
plt.title("Policy-Level Hack Flow")
plt.savefig("policy_capture.png")
plt.show()

