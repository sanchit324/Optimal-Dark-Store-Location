import pulp
import matplotlib.pyplot as plt
import numpy as np

# Create the problem object
prob = pulp.LpProblem("Graph Optimization", pulp.LpMinimize)

# Define node coordinates and number of nodes
coordi = [
    [2.0, 2.0],
    [2.1, 2.1],
    [1.9, 2.2],
    [2.2, 1.9],
    [1.8, 2.0],
    [2.0, 1.8],
    [2.1, 1.9],
    [1.9, 2.1],
    [2.0, 1.9],
    [1.8, 2.1],
    [2.2, 2.2],
    [1.8, 1.8],
    [2.0, 2.2],
    [1.9, 1.9],
    [1.8, 2.3],
    [2.1, 1.8],
    [2.3, 2.0],
    [2.2, 1.7],
    [1.7, 2.2],
    [2.3, 1.9]
]

N = len(coordi)

# Compute distances between nodes
d = [[0]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        dist = np.sqrt((coordi[i][0] - coordi[j][0])**2 + (coordi[i][1] - coordi[j][1])**2)
        d[i][j] = dist

# Define the number of clusters
k = 3

# Initialize decision variables
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N) for j in range(k)), cat="Binary")
y = pulp.LpVariable.dicts("y", ((i, j) for i in range(N) for j in range(i + 1, N)), cat="Binary")

# Define the objective function
prob += sum(d[i][j] * y[i, j] for i in range(N) for j in range(i + 1, N))

# Define constraints
for i in range(N):
    prob += sum(x[i, j] for j in range(k)) == 1

for c in range(k):
    prob += sum(x[i, c] for i in range(N)) >= 1

for i in range(N - 1):
    for j in range(i + 1, N):
        for c in range(k):
            prob += y[i, j] >= x[i, c] + x[j, c] - 1

# Solve the problem
prob.solve()

# Extract variable values from the solution
x_values = {(i, j): x[i, j].varValue for i in range(N) for j in range(k)}
y_values = {(i, j): y[i, j].varValue for i in range(N) for j in range(i + 1, N)}

# Create a list of edges based on y variable values
edges = [(i, j) for (i, j) in y_values if y_values[i, j] == 1]

# Define node colors for each cluster
cluster_colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'purple']

# Plotting nodes and edges
plt.figure(figsize=(10, 8))

# Plot nodes
for i in range(N):
    cluster_assignment = [c for c in range(k) if x_values[(i, c)] == 1]
    color = cluster_colors[cluster_assignment[0]] if cluster_assignment else 'gray'
    plt.scatter(coordi[i][0], coordi[i][1], color=color, s=400, edgecolors='k', linewidth=1.5)
    plt.text(coordi[i][0], coordi[i][1], str(i), color='black', ha='center', va='center', fontsize=12, fontweight='bold')

# Plot edges
for (i, j) in edges:
    plt.plot([coordi[i][0], coordi[j][0]], [coordi[i][1], coordi[j][1]], 'gray', alpha=0.6)

# Plot centroids of each cluster
for c in range(k):
    cluster_nodes = [i for i in range(N) if x_values[(i, c)] == 1]
    if cluster_nodes:
        cluster_x = np.mean([coordi[i][0] for i in cluster_nodes])
        cluster_y = np.mean([coordi[i][1] for i in cluster_nodes])
        plt.scatter(cluster_x, cluster_y, color=cluster_colors[c], s=400, edgecolors='k', linewidth=1, marker='*')

# Customize plot appearance
plt.title('Optimal Dark Store Location', fontsize=16)
plt.xlabel('X-coordinate', fontsize=14)
plt.ylabel('Y-coordinate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Create legend for clusters
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i+1}', 
                               markerfacecolor=cluster_colors[i], markersize=10) for i in range(k)]
legend_elements.append(plt.Line2D([0], [0], marker='*', color='k', label='Dark Store Location', 
                                   markerfacecolor='white', markersize=10, markeredgewidth=2))
plt.legend(handles=legend_elements, loc='upper center', fontsize=12)

plt.show()
