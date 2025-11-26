"""
Debug script to trace parameter reading and find index misalignment
"""
import os
import sys

# Read inputs.txt and parse to see what values_vector looks like
filename = "inputs.txt"

with open(filename, 'r') as f:
    lines = f.readlines()

tokens = []
for line in lines:
    line = line.strip()
    if not line or line.startswith('%'):
        continue
    tokens.extend(line.split())

print("="*70)
print("PARAMETER FILE PARSING DEBUG")
print("="*70)

# Simulate the C++ parsing logic
values_vector = []
i = 0
param_index = 0

print(f"\nTotal tokens: {len(tokens)}\n")
print("Token stream (first 100):")
for idx, token in enumerate(tokens[:100]):
    print(f"  [{idx:3d}] {token}")

print("\n" + "="*70)
print("PARAMETER EXTRACTION SIMULATION")
print("="*70)

i = 0
while i < len(tokens):
    name = tokens[i]
    if i + 1 < len(tokens):
        value = tokens[i+1]
        
        # C++ check: if (value.front() != '%' && value.back() != ':')
        if not value.startswith('%') and not value.endswith(':'):
            values_vector.append(value)
            print(f"[{param_index:3d}] {name:30s} = {value}")
            param_index += 1
            i += 2  # Consumed name and value
        else:
            # Value is missing or is next key
            values_vector.append("DEFAULT")
            print(f"[{param_index:3d}] {name:30s} = DEFAULT (value={value} looks like key)")
            param_index += 1
            i += 1  # Consumed name only
    else:
        # Last token, no value
        values_vector.append("DEFAULT")
        print(f"[{param_index:3d}] {name:30s} = DEFAULT (no value)")
        param_index += 1
        i += 1

print("\n" + "="*70)
print("EXPECTED PARAMETER POSITIONS")
print("="*70)

expected_params = [
    (0, "timeStep"),
    (1, "maximumNumberOfIterations"),
    (2, "numberOfPatches"),
    (3, "minimumParticlesPerCell"),
    (4, "maximumParticlesPerCell"),
    (5, "specificWeight"),
    (6, "simulationType"),
    (7, "axisymmetricFlag"),
    (8, "twoStreamFlag"),
    (9, "initialParticlesPerCell"),
    (10, "numCellsWithParticles"),
    (11, "particleDistribution"),
    (12, "initialTemperature"),
    (13, "initialPosition"),
    (14, "initialVelocity"),
    (15, "inletSource"),
    (16, "inletSize"),
    (17, "inletFlowRate"),
    (18, "inletVelocity"),
    (19, "propellant"),
    (20, "MCCFrequency"),
    (21, "electricField"),
    (22, "magneticField"),
    (23, "FDTDIterations"),
    (24, "FDTDFrequency"),
    (25, "useRFCoil"),
    (26, "coilTurns"),
    (27, "coilRadius"),
    (28, "coilLength"),
    (29, "coilCurrent"),
    (30, "rfFrequency"),
    (31, "userMesh"),
    (32, "domainLength"),
    (33, "domainHeight"),
]

print(f"\nLooking for domainLength and domainHeight:")
print(f"  Expected at indices: 32, 33")
print(f"  Values_vector length: {len(values_vector)}")

if len(values_vector) > 33:
    print(f"\n  values_vector[32] = {values_vector[32]} (should be domainLength = 0.04)")
    print(f"  values_vector[33] = {values_vector[33]} (should be domainHeight = 0.005)")
else:
    print(f"\n  ERROR: values_vector only has {len(values_vector)} elements!")

print("\n" + "="*70)
