"""
RF Coil Module - Analytical calculation of magnetic field from induction coil

For a cylindrical solenoid carrying RF current, calculates the  
magnetic field using analytical formulas.
"""

import math
from .constants import MU_0
import numpy as np

class RFCoil:
    def __init__(self, parameters):
        """
        Initialize RF coil with parameters from inputs.txt
        
        Args:
            parameters: Parameters object containing coil configuration
        """
        self.parameters = parameters
        self.n_turns = parameters.coil_turns
        self.radius = parameters.coil_radius  # m
        self.length = parameters.coil_length  # m
        self.current_amplitude = parameters.coil_current  # A
        self.omega = 2.0 * math.pi * parameters.rf_frequency  # rad/s
        
        # Turn density (turns per meter)
        self.n = self.n_turns / self.length if self.length > 0 else 0
        
        # Coil center position (assume centered on domain)
        self.z_center = parameters.domain_length / 2.0  # Axial center
        self.z_start = self.z_center - self.length / 2.0
        self.z_end = self.z_center + self.length / 2.0
        
    def get_current(self, time):
        """
        Get time-varying coil current.
        
        Args:
            time: Current simulation time (s)
            
        Returns:
            float: Instantaneous current (A)
        """
        return self.current_amplitude * math.sin(self.omega * time)
        
    def calculate_field_solenoid(self, r, z, time):
        """
        Calculate B-field using infinite solenoid approximation.
        Valid for points well inside the solenoid.
        
        For a solenoid:
        - Inside: B_z = μ₀ * n * I, B_r = 0
        - Outside: B_z ≈ 0, B_r ≈ 0
        
        Args:
            r: Radial position (m)
            z: Axial position (m) 
            time: Current time (s)
            
        Returns:
            tuple: (Br, Bz) magnetic field components (T)
        """
        I = self.get_current(time)
        
        # Check if inside coil axially
        if self.z_start <= z <= self.z_end and r < self.radius:
            # Inside solenoid
            Bz = MU_0 * self.n * I
            Br = 0.0
        else:
            # Outside or end regions - simplified to zero
            # (more accurate would use finite solenoid formula)
            Bz = 0.0
            Br = 0.0
            
        return Br, Bz
        
    def calculate_field_finite_solenoid(self, r, z, time):
        """
        Calculate B-field for finite solenoid using more accurate formula.
        
        For a finite solenoid, on axis (r=0):
        B_z = (μ₀*n*I/2) * (cos(θ₁) - cos(θ₂))
        where θ₁, θ₂ are angles from point to coil ends
        
        Off-axis is more complex (elliptic integrals).
        
        Args:
            r: Radial position (m)
            z: Axial position (m)
            time: Current time (s)
            
        Returns:
            tuple: (Br, Bz) magnetic field components (T)
        """
        I = self.get_current(time)
        
        if r < 1e-10:  # On axis
            # Calculate angles to coil ends
            z1 = z - self.z_start
            z2 = z - self.z_end
            
            # Distance to ends
            R = self.radius
            d1 = math.sqrt(R*R + z1*z1)
            d2 = math.sqrt(R*R + z2*z2)
            
            # Cosines
            cos_theta1 = z1 / d1 if d1 > 0 else 0
            cos_theta2 = z2 / d2 if d2 > 0 else 0
            
            Bz = (MU_0 * self.n * I / 2.0) * (cos_theta1 - cos_theta2)
            Br = 0.0
        else:
            # Off-axis - use simplified approximation
            # (Full solution requires elliptic integrals)
            # For now, use tapered field
            
            # Axial field (reduced off-axis)
            z1 = z - self.z_start
            z2 = z - self.z_end
            R = self.radius
            
            d1 = math.sqrt(R*R + z1*z1)
            d2 = math.sqrt(R*R + z2*z2)
            
            cos_theta1 = z1 / d1 if d1 > 0 else 0
            cos_theta2 = z2 / d2 if d2 > 0 else 0
            
            # Reduce field strength based on radius
            radial_factor = math.exp(-(r/R)**2) if r < R else 0.0
            
            Bz = (MU_0 * self.n * I / 2.0) * (cos_theta1 - cos_theta2) * radial_factor
            
            # Simple radial component (∇·B = 0 requires ∂Br/∂r + Br/r + ∂Bz/∂z = 0)
            # For axisymmetric case: Br ≈ -(r/2) * ∂Bz/∂z
            # Simplified estimate
            if abs(z2 - z1) > 1e-10:
                dBz_dz = (MU_0 * self.n * I / 2.0) * (
                    z1/(d1**3) * R*R - z2/(d2**3) * R*R
                ) * radial_factor
                Br = -(r / 2.0) * dBz_dz
            else:
                Br = 0.0
                
        return Br, Bz
        
    def get_field_at_position(self, x, y, time, use_finite=True):
        """
        Get B-field at a position in the simulation domain.
        
        Assumes axisymmetric geometry:
        - x → z (axial direction)
        - y → r (radial direction)
        
        Args:
            x: Axial position (m)
            y: Radial position (m)
            time: Current time (s)
            use_finite: Use finite solenoid formula (more accurate)
            
        Returns:
            tuple: (Bx, By, Bz) where Bx=Bz_axial, By=Br_radial, Bz=Btheta=0
        """
        # In axisymmetric coordinates: x→z, y→r
        z = x
        r = y
        
        if use_finite:
            Br, Bz_axial = self.calculate_field_finite_solenoid(r, z, time)
        else:
            Br, Bz_axial = self.calculate_field_solenoid(r, z, time)
            
        # Return in Cartesian-like format for PIC code
        # Bx = axial component (along x-axis in our coords)
        # By = radial component
        # Bz = azimuthal component (0 for axisymmetric)
        return Bz_axial, Br, 0.0

    def get_dI_dt(self, time):
        """
        Get time derivative of coil current.
        dI/dt = I0 * omega * cos(omega * t)
        """
        return self.current_amplitude * self.omega * math.cos(self.omega * time)

    def get_E_field_at_position(self, x, y, time):
        """
        Get induced E-field (E_theta) at a position.
        E_theta = -0.5 * r * dBz/dt (Faraday's Law for axisymmetric B)
        
        Args:
            x: Axial position (m)
            y: Radial position (m)
            time: Current time (s)
            
        Returns:
            tuple: (Ex, Ey, Ez) where Ez is E_theta
        """
        z = x
        r = y
        
        # Calculate dBz/dt
        # For infinite solenoid: Bz = mu0 * n * I
        # dBz/dt = mu0 * n * dI/dt
        
        dI_dt = self.get_dI_dt(time)
        
        # Use finite solenoid factor if inside
        # Bz_finite approx Bz_infinite * factor
        # Let's use the same geometric factor as for B-field
        
        # Calculate geometric factor (Bz / (mu0 * n * I))
        # If I=1, Bz_norm = Bz(I=1) / (mu0 * n)
        # But we can just reuse the logic.
        
        # On axis factor: 0.5 * (cos1 - cos2)
        z1 = z - self.z_start
        z2 = z - self.z_end
        R = self.radius
        d1 = math.sqrt(R*R + z1*z1)
        d2 = math.sqrt(R*R + z2*z2)
        cos_theta1 = z1 / d1 if d1 > 0 else 0
        cos_theta2 = z2 / d2 if d2 > 0 else 0
        
        geometric_factor = 0.5 * (cos_theta1 - cos_theta2)
        
        # Radial decay factor
        radial_factor = math.exp(-(r/R)**2) if r < R else 0.0
        
        # dBz/dt
        dBz_dt = MU_0 * self.n * dI_dt * geometric_factor * radial_factor
        
        # E_theta = -0.5 * r * dBz/dt
        # This assumes B is uniform in r (integral form of Faraday's law: 2*pi*r*E = -d/dt(integral B dA))
        # If B varies with r, we need integral(B(r') * 2*pi*r' dr')
        # For solenoid, B is roughly constant inside.
        # So E_theta = -0.5 * r * dBz/dt is good approx inside.
        
        E_theta = -0.5 * r * dBz_dt
        
        # Ex (axial) = 0
        # Ey (radial) = 0 (from induction)
        # Ez (azimuthal) = E_theta
        
        return 0.0, 0.0, E_theta
