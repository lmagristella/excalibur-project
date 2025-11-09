#!/usr/bin/env python3
"""
Backward ray tracing simulation using the excalibur library.

This script demonstrates backward ray tracing of multiple photons in a cone
towards a spherical mass within a cosmological (perturbed FLRW) spacetime.

Features:
- Multiple photons generated in a cone configuration
- Backward ray tracing (reverse time integration)
- Spherical mass perturbation in FLRW cosmology
- Collective trajectory saving to HDF5 file
"""

import numpy as np
from scipy import interpolate 
import sys
import os

sys.path.insert(0, '/home/magri/excalibur_project')

### excalibur imports ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
##########################

def main():
    """Main function for backward ray tracing simulation."""
    
    print("=== Backward Ray Tracing with Excalibur ===\n")
    
    # =============================================================================
    # 1. COSMOLOGICAL SETUP
    # =============================================================================
    print("1. Setting up cosmology...")
    
    # Define ΛCDM cosmology
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_r = 0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)
    
    # Create scale factor interpolation
    eta_sample = np.linspace(0.1, 10, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")
    
    print(f"   Cosmology: H0={H0} km/s/Mpc, Ωm={Omega_m}, ΩΛ={Omega_lambda}")
    print(f"   Scale factor range: a({eta_sample[0]:.1f}) = {a_sample[0]:.3f} to a({eta_sample[-1]:.1f}) = {a_sample[-1]:.3f}")
    
    # =============================================================================
    # 2. GRID AND MASS DISTRIBUTION SETUP
    # =============================================================================
    print("\n2. Setting up grid and mass distribution...")
    
    # Grid parameters
    N = 512  # Grid resolution (reduced for faster computation)
    grid_size = 1000 * one_Mpc  # 1000 Mpc box size in meters
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (0, 0, 0)
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays
    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define spherical mass (target for backward ray tracing)
    M = 1e30 * one_Msun  # Galaxy cluster mass in kg
    radius = 10 * one_Mpc   # Virial radius
    center = np.array([0.5, 0.5, 0.5]) * grid_size  # Off-center position
    spherical_halo = spherical_mass(M, radius, center)
    
    # Compute potential field
    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    
    print(f"   Grid: {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"   Mass: {M/one_Msun:.1e} M☉ at [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
    print(f"   Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m²/s²")
    
    # =============================================================================
    # 3. INTERPOLATOR AND METRIC SETUP
    # =============================================================================
    print("\n3. Setting up spacetime metric...")
    
    interpolator = Interpolator(grid)
    metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)
    
    # Test metric at mass center
    test_pos = [2.0, center[0], center[1], center[2]]  # [η, x, y, z]
    christoffel = metric.christoffel(test_pos)
    print(f"   Metric initialized successfully")
    print(f"   Test Christoffel Γ[0,0,0] = {christoffel[0,0,0]:.2e} at mass center")
    
    # =============================================================================
    # 4. BACKWARD RAY TRACING SETUP
    # =============================================================================
    print("\n4. Setting up backward ray tracing...")
    
    # Observer position and time (where photons are "detected")
    observer_eta = 46*one_Gyr  # Conformal time at observation
    observer_position = np.array([0.0, 0.0, 0.0]) * grid_size  # Observer location
    
    # Direction towards the mass (for backward tracing, we point towards the source)
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters for photon generation
    n_photons = 50  # Number of photons to trace
    cone_angle = np.pi / 12  # 15-degree half-angle cone
    energy = 1.0  # Photon energy scale
    
    print(f"   Observer at [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"   Observation time: η = {observer_eta:.1f}")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f}° half-angle")
    print(f"   Direction to mass: [{direction_to_mass[0]:.3f}, {direction_to_mass[1]:.3f}, {direction_to_mass[2]:.3f}]")
    
    # =============================================================================
    # 5. PHOTON GENERATION
    # =============================================================================
    print("\n5. Generating photons for backward ray tracing...")
    
    photons = Photons()
    
    # Generate photons in a cone pointing towards the mass
    observer_4d_position = np.array([observer_eta, *observer_position])
    
    photons.generate_cone_random(
        n_photons=n_photons,
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle,
        energy=energy
    )
    
    print(f"   Generated {len(photons)} photons in cone")
    
    # For backward ray tracing, we need to reverse the time component of 4-velocity
    # This makes photons travel backwards in time
    for photon in photons:
        photon.u[0] = -abs(photon.u[0])  # Make u^η negative for backward evolution
        photon.record()  # Record initial state
    
    print("   Configured photons for backward time evolution")
    
    # =============================================================================
    # 6. BACKWARD INTEGRATION
    # =============================================================================
    print("\n6. Performing backward ray tracing integration...")
    
    # Create integrator with negative time step for backward evolution
    dt = 1e-3 * one_Gyr   # cosmic time step to convert to conformal time step
    deta = -dt / a_of_eta(observer_eta)  
    integrator = Integrator(metric, dt=dt)
    
    # Integration parameters
    # Ensure photons spend at least a few time-steps inside each grid cell on average.
    # For one step a photon moves ~ c * |dt|, so crossing a cell of size dx takes ~ dx / (c*|dt|)
    min_steps_per_cell = 5  # desired minimum average steps inside one cell
    n_cells = int(np.ceil(grid_size / dx))  # cells along a straight path across the box

    # conservative light crossing time for whole box and corresponding steps
    light_travel_time = grid_size / c
    n_steps_travel = int(np.ceil(light_travel_time / abs(dt)))

    # enforce minimum total steps so that on average each cell gets min_steps_per_cell samples
    min_steps_total = int(min_steps_per_cell * n_cells)

    n_steps = max(n_steps_travel, min_steps_total, 1)

    print(n_steps)
    save_interval = 100  # Print progress every N photons
    
    print(f"   Integration: {n_steps} steps with dt = {dt}")
    print(f"   Total backward time: Δη = {n_steps * dt:.2f}")
    
    # Integrate all photons
    for i, photon in enumerate(photons):
        integrator.integrate(photon, n_steps)
        
        if (i + 1) % save_interval == 0 or (i + 1) == len(photons):
            print(f"   Progress: {i + 1}/{len(photons)} photons completed")
    
    # =============================================================================
    # 7. ANALYSIS AND RESULTS
    # =============================================================================
    print("\n7. Analyzing results...")
    
    # Calculate statistics
    trajectory_lengths = [len(photon.history.states) for photon in photons]
    avg_length = np.mean(trajectory_lengths)
    
    # Find final positions (earliest times in backward tracing)
    final_positions = []
    final_times = []
    
    for photon in photons:
        if len(photon.history.states) > 0:
            final_state = photon.history.states[-1]  # Last recorded state
            final_times.append(final_state[0])  # η coordinate
            final_positions.append(final_state[1:4])  # spatial coordinates
    
    if final_positions:
        final_positions = np.array(final_positions)
        final_times = np.array(final_times)
        
        # Calculate spread and distance from mass
        distances_from_mass = [np.linalg.norm(pos - center) for pos in final_positions]
        avg_distance_from_mass = np.mean(distances_from_mass)
        min_distance_from_mass = np.min(distances_from_mass)
        
        print(f"   Average trajectory length: {avg_length:.1f} states")
        print(f"   Final time range: η ∈ [{final_times.min():.2f}, {final_times.max():.2f}]")
        print(f"   Distance from mass: avg = {avg_distance_from_mass/one_Mpc:.2f} Mpc, min = {min_distance_from_mass/one_Mpc:.2f} Mpc")
        print(f"   Spatial spread: {np.std(final_positions, axis=0)/one_Mpc} Mpc")
    
    # =============================================================================
    # 8. SAVE RESULTS
    # =============================================================================
    print("\n8. Saving trajectories...")
    
    output_filename = "backward_raytracing_trajectories.h5"
    
    try:
        photons.save_all_histories(output_filename)
        print(f"   ✓ Saved all {len(photons)} photon trajectories to {output_filename}")
        
        # Get file size
        file_size = os.path.getsize(output_filename)
        print(f"   File size: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"   ✗ Error saving trajectories: {e}")
    
    # =============================================================================
    # 9. SUMMARY
    # =============================================================================
    print("\n" + "="*60)
    print("BACKWARD RAY TRACING SUMMARY")
    print("="*60)
    print(f"Cosmology:        ΛCDM (H₀={H0}, Ωₘ={Omega_m}, ΩΛ={Omega_lambda})")
    print(f"Grid:             {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"Mass:             {M/one_Msun:.1e} M☉")
    print(f"Observer:         η={observer_eta}, r=[{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"Photons:          {len(photons)} in {cone_angle*180/np.pi:.1f}° cone")
    print(f"Integration:      {n_steps} steps, Δη={n_steps*dt:.2f}")
    print(f"Output:           {output_filename}")
    print("="*60)
    print("✓ Backward ray tracing completed successfully!")
    

if __name__ == "__main__":
    main()
