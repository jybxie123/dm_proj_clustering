import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [0.6870, 0.9140, 0.9100, 0.9350, 0.9590, 0.9750, 0.9770, 0.9820, 0.9870, 0.9880]
y2 = [0.6560, 0.9040, 0.8730, 0.8830, 0.9600, 0.9680, 0.9480, 0.9860, 0.9860, 0.9910]
y3 = [0.2770, 0.4350, 0.4500, 0.3980, 0.4680, 0.5290, 0.5600, 0.7710, 0.7770, 0.7980]
y4 = [0.7260, 0.8040, 0.9200, 0.9440, 0.9650, 0.9700, 0.9400, 0.9400, 0.9280, 0.9880]

y1 = [val * 100 for val in y1]
y2 = [val * 100 for val in y2]
y3 = [val * 100 for val in y3]
y4 = [val * 100 for val in y4]

plt.plot(x, y1, marker='o', label='K-Means') 
plt.plot(x, y2, marker='s', label='K-Means++')  
plt.plot(x, y4, marker='x', label='Minibatch K-Means') 
plt.plot(x, y3, marker='^', label='K-Median')  

plt.title('IVF with K-Means Series Clustering: RR @ 100')
plt.xlabel('nprobe')
plt.ylabel('Recall Rate (%)')
plt.grid(axis='x', linestyle='--')
plt.grid(axis='y', linestyle='--')
plt.legend()

plt.show()