import matplotlib.pyplot as plt
import numpy as np

# Load GPU exported data
points = np.loadtxt("points.txt")
triangles = np.loadtxt("triangles.txt", dtype=int)

plt.figure(figsize=(8, 8))
plt.triplot(points[:, 0], points[:, 1], triangles, color="crimson", lw=0.5)
plt.scatter(points[:, 0], points[:, 1], color="blue", s=5)
plt.title(f"GPU Delaunay Triangulation ({len(points)} Points)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("delaunay_gpu.png", dpi=300)
plt.show()