"""
Quick verification script to check mesh dimensions after fix
"""
import sys
sys.path.insert(0, 'c:/Users/leona/PIC_IPD_Python')

from pic_modules.parameters import Parameters
from pic_modules.mesh import Mesh

# Load parameters
params = Parameters("inputs.txt")
params.assign_inputs()

# Generate mesh
mesh = Mesh(params, "PIC")

# Check dimensions
print("="*60)
print("MESH DIMENSION VERIFICATION")
print("="*60)
print(f"\nInput parameters:")
print(f"  domainLength (axial, z): {params.domain_length*1000:.2f} mm")
print(f"  domainHeight (radial, r): {params.domain_height*1000:.2f} mm")
print(f"  PIC spacing: {params.pic_spacing*1000:.2f} mm")

print(f"\nGrid dimensions:")
print(f"  num_cells_x (radial): {mesh.num_cells_x}")
print(f"  num_cells_y (axial): {mesh.num_cells_y}")
print(f"  Total cells: {mesh.num_cells}")
print(f"  Total nodes: {mesh.num_nodes}")

print(f"\nNode coordinate ranges:")
print(f"  First node (0,0): r={mesh.nodes[0].x*1000:.3f} mm, z={mesh.nodes[0].y*1000:.3f} mm")
print(f"  Last node: r={mesh.nodes[-1].x*1000:.3f} mm, z={mesh.nodes[-1].y*1000:.3f} mm")

# Check actual mesh extents
min_r = min(node.x for node in mesh.nodes)
max_r = max(node.x for node in mesh.nodes)
min_z = min(node.y for node in mesh.nodes)
max_z = max(node.y for node in mesh.nodes)

print(f"\nMesh extents:")
print(f"  Radial (r): {min_r*1000:.3f} to {max_r*1000:.3f} mm")
print(f"  Axial (z): {min_z*1000:.3f} to {max_z*1000:.3f} mm")

# Verify correctness
print(f"\nVerification:")
r_expected = params.domain_height
z_expected = params.domain_length

r_correct = abs(max_r - r_expected) < 1e-10
z_correct = abs(max_z - z_expected) < 1e-10

print(f"  Radial extent correct: {'✓ YES' if r_correct else '✗ NO'}")
print(f"    Expected: {r_expected*1000:.2f} mm, Got: {max_r*1000:.3f} mm")
print(f"  Axial extent correct: {'✓ YES' if z_correct else '✗ NO'}")
print(f"    Expected: {z_expected*1000:.2f} mm, Got: {max_z*1000:.3f} mm")

if r_correct and z_correct:
    print("\n✓✓✓ MESH DIMENSIONS ARE CORRECT! ✓✓✓")
else:
    print("\n✗✗✗ MESH DIMENSIONS ARE STILL WRONG! ✗✗✗")

print("="*60)
