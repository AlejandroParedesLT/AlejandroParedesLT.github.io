import matplotlib.pyplot as plt
import numpy as np

# Simulate misclassification probabilities
skin_tones = ['Light', 'Medium', 'Dark']
genders = ['Male', 'Female']
error_rates = np.array([[0.02, 0.10], [0.05, 0.20], [0.08, 0.30]])  # Dark female highest

fig, ax = plt.subplots()
im = ax.imshow(error_rates, cmap='Reds')

# Show labels
ax.set_xticks(np.arange(len(genders)))
ax.set_yticks(np.arange(len(skin_tones)))
ax.set_xticklabels(genders)
ax.set_yticklabels(skin_tones)
ax.set_title("Simulated Face Recognition Error Rates")
for i in range(len(skin_tones)):
    for j in range(len(genders)):
        ax.text(j, i, f"{error_rates[i, j]*100:.0f}%", ha="center", va="center", color="black")
plt.savefig("system_layers.png")
plt.show()


import seaborn as sns

# Fake dataset label distribution
labels = ['Male', 'Female']
skin = ['Light', 'Dark']
np.random.seed(42)
data = np.random.choice(labels, size=2000, p=[0.7,0.3])
sns.histplot(data, hue=np.random.choice(skin, size=2000), multiple='stack')
plt.title("Simulated Dataset Label Distribution")
plt.savefig("otherExample.png")
plt.show()