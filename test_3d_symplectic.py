"""
High-resolution, long-time 3D PPF Navier-Stokes simulation using symplectic integration
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from ppf_navstokes.geometry.iot_3d import IOT3DGeometry
from ppf_navstokes.physics.fluid_state_3d import FluidState3D, VelocityField3D
from ppf_navstokes.physics.dlce_fluid_3d import DLCEFluidSolver3D
from ppf_navstokes.numerics.symplectic_integrator_3d import SymplecticIntegrator3D
from ppf_navstokes.numerics.time_integration import AdaptiveTimeStep


def _safe_gradient(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Compute gradient with periodic boundaries for u,v and natural for r"""
    grad = np.zeros_like(field)
    
    if axis == 0:  # u-direction (periodic)
        grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * spacing)
        grad[0, :, :] = (field[1, :, :] - field[-1, :, :]) / (2 * spacing)
        grad[-1, :, :] = grad[0, :, :]
    elif axis == 1:  # v-direction (periodic)
        grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * spacing)
        grad[:, 0, :] = (field[:, 1, :] - field[:, -1, :]) / (2 * spacing)
        grad[:, -1, :] = grad[:, 0, :]
    elif axis == 2:  # r-direction (natural boundaries)
        if field.shape[2] > 2:
            grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * spacing)
        if field.shape[2] > 1:
            grad[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / spacing
            grad[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / spacing
    
    return grad


def create_high_res_initial_condition(geometry: IOT3DGeometry,
                                    u_grid: np.ndarray,
                                    v_grid: np.ndarray,
                                    r_grid: np.ndarray) -> FluidState3D:
    """
    Create analytically divergence-free initial condition
    
    Uses stream function approach to ensure div(u) = 0
    """
    # Test with higher initial energy to verify robustness
    amplitude = 1.0  # Higher amplitude to test energy dissipation scaling
    
    # Normalized coordinates for boundary conditions
    r_normalized = (r_grid - r_grid.min()) / (r_grid.max() - r_grid.min() + 1e-10)
    
    # Create stream function for divergence-free flow in toroidal coordinates
    # For toroidal flow, we use two stream functions: ψ (poloidal) and χ (toroidal)
    
    # Poloidal stream function (creates u,w components)
    psi = amplitude * r_normalized * (1 - r_normalized) * (
        np.sin(u_grid) * np.cos(v_grid) + 
        0.5 * np.sin(2*u_grid) * np.sin(v_grid)
    )
    
    # Toroidal stream function (creates v component)  
    chi = amplitude * r_normalized * (1 - r_normalized) * (
        np.cos(u_grid) * np.sin(v_grid) +
        0.3 * np.cos(3*u_grid) * np.cos(2*v_grid)
    )
    
    # Grid spacings
    du = float(u_grid[1,0,0] - u_grid[0,0,0]) if u_grid.shape[0] > 1 else 0.1
    dv = float(v_grid[0,1,0] - v_grid[0,0,0]) if v_grid.shape[1] > 1 else 0.1
    dr = float(r_grid[0,0,1] - r_grid[0,0,0]) if r_grid.shape[2] > 1 else 0.1
    
    # Compute velocity from stream functions
    # In toroidal coordinates:
    # u = -∂ψ/∂r  (toroidal velocity)
    # v = ∂χ/∂r   (poloidal velocity)  
    # w = (1/r)∂ψ/∂v (radial velocity)
    
    # Create a truly 3D helical flow with non-zero helicity
    # Using ABC (Arnold-Beltrami-Childress) flow structure adapted to toroidal coordinates
    A, B, C = 1.0, 0.7, 0.5  # ABC flow parameters
    
    # Create a simple, smooth initial condition
    # Use direct construction to ensure low divergence
    
    # Simple, smooth initial condition with minimal divergence
    # Use simple trig functions with proper decay at boundaries
    
    # Smooth radial decay function
    radial_factor = r_normalized * (1 - r_normalized) * np.exp(-2*r_normalized)
    
    # Simple toroidal flow
    u_component = amplitude * np.sin(u_grid) * radial_factor
    v_component = amplitude * np.cos(v_grid) * radial_factor
    w_component = np.zeros_like(u_component)  # Start with no radial flow
    
    # Apply smooth cutoff near boundaries
    boundary_factor = np.ones_like(r_normalized)
    boundary_width = 0.1
    
    # Smooth cutoff at r=0
    mask_inner = r_normalized < boundary_width
    boundary_factor[mask_inner] *= (r_normalized[mask_inner] / boundary_width)**2
    
    # Smooth cutoff at r=r_max
    mask_outer = r_normalized > (1 - boundary_width)
    boundary_factor[mask_outer] *= ((1 - r_normalized[mask_outer]) / boundary_width)**2
    
    # Apply boundary factor
    u_component *= boundary_factor
    v_component *= boundary_factor
    w_component *= boundary_factor
    
    # Ensure w=0 at r=0 and r=r_max exactly
    w_component[:, :, 0] = 0.0
    w_component[:, :, -1] = 0.0
    
    # Initialize pressure (will be computed by solver)
    pressure = np.zeros_like(u_component)
    
    # Project to ensure divergence-free
    from ppf_navstokes.numerics.spectral_methods_3d import SpectralSolver3D
    spectral_solver = SpectralSolver3D(u_grid.shape[0], v_grid.shape[1], r_grid.shape[2], geometry)
    u_proj, v_proj, w_proj, pressure = spectral_solver.project_divergence_free_3d(
        u_component, v_component, w_component
    )
    
    velocity = VelocityField3D(u_proj, v_proj, w_proj, pressure)
    state = FluidState3D(velocity, geometry, u_grid, v_grid, r_grid, time=0.0)
    
    return state


def run_highres_long_simulation():
    """Run high-resolution 3D simulation for extended time"""
    
    print("="*70)
    print("HIGH-RESOLUTION 3D PPF NAVIER-STOKES (SYMPLECTIC INTEGRATION)")
    print("="*70)
    
    # Create high-resolution geometry
    print("\nSetting up high-resolution 3D IOT geometry...")
    geometry = IOT3DGeometry(R=30.0, r_max=1.0, n_r=4)  # More radial points
    
    # Higher resolution to test scaling with n_r=4 symmetry
    n_u, n_v = 32, 32  # Increased resolution to test robustness
    u_grid, v_grid, r_grid = geometry.create_grid_3d(n_u, n_v)
    
    total_points = np.prod(u_grid.shape)
    print(f"Grid: {u_grid.shape} = {total_points:,} total points")
    print(f"Memory estimate: ~{total_points * 8 * 10 / 1e6:.1f} MB")
    
    # Create initial condition
    print("\nCreating multi-scale initial condition...")
    initial_state = create_high_res_initial_condition(geometry, u_grid, v_grid, r_grid)
    
    print(f"\nInitial state:")
    print(f"  Energy: {initial_state.energy_3d():.4f}")
    print(f"  Helicity: {initial_state.helicity_integral():.4f}")
    print(f"  Max |u|: {np.max(initial_state.velocity.magnitude()):.4f}")
    
    # Create solver with stronger dissipation parameters
    print("\nInitializing PPF solver with enhanced stability parameters...")
    solver = DLCEFluidSolver3D(
        geometry=geometry, n_u=n_u, n_v=n_v,
        nu=0.1,        # Reasonable viscosity to see dissipation
        alpha=0.05,    # Moderate tautochrone coupling
        beta=0.05,     # Moderate PPF damping
        gamma=0.02,    # Moderate observational coupling
        delta=0.02     # Moderate involution coupling
    )
    
    # Check initial divergence using periodic gradients
    du = float(u_grid[1,0,0] - u_grid[0,0,0]) if u_grid.shape[0] > 1 else 0.1
    dv = float(v_grid[0,1,0] - v_grid[0,0,0]) if v_grid.shape[1] > 1 else 0.1
    dr = float(r_grid[0,0,1] - r_grid[0,0,0]) if r_grid.shape[2] > 1 else 0.1
    
    div_initial = (solver._periodic_gradient(initial_state.velocity.u_component, du, 0) +
                   solver._periodic_gradient(initial_state.velocity.v_component, dv, 1) +
                   solver._periodic_gradient(initial_state.velocity.w_component, dr, 2))
    print(f"Initial divergence (using periodic gradients): Max |div u|: {np.max(np.abs(div_initial)):.6f}")
    
    # Create symplectic integrator
    print("Initializing symplectic integrator...")
    integrator = SymplecticIntegrator3D(solver)
    
    # Create adaptive time stepper with reasonable parameters
    adaptive = AdaptiveTimeStep(solver, cfl_number=0.1, min_dt=1e-6, max_dt=1e-3)
    
    # Time integration parameters
    t_final = 1.0   # Focus on key dissipation physics
    save_interval = 0.05  # More frequent saves for better resolution
    
    # Storage arrays
    times = [0.0]
    energies = [initial_state.energy_3d()]
    helicities = [initial_state.helicity_integral()]
    max_velocities = [np.max(initial_state.velocity.magnitude())]
    divergences = [np.max(np.abs(div_initial))]
    
    # Run simulation
    print(f"\nRunning to t={t_final} with adaptive timestep...")
    print("\nTime    Energy    E/E0     Helicity   Max|u|   Max|div|   dt      Elapsed")
    print("-" * 80)
    
    current_state = initial_state
    step = 0
    E0 = energies[0] if energies[0] > 1e-10 else 1.0  # Prevent division by zero
    last_save_time = 0.0
    start_time = time.time()
    
    # Print initial state
    print(f"{0.0:5.2f}  {E0:8.4f}  {1.000:6.3f}  {helicities[0]:8.4f}  "
          f"{max_velocities[0]:7.4f}  {divergences[0]:8.6f}  ----  {0.0:7.1f}s")
    
    try:
        while current_state.time < t_final:
            # Compute adaptive timestep
            dt = adaptive.compute_dt(current_state)
            
            # Don't overshoot save time
            if current_state.time + dt > last_save_time + save_interval:
                dt = last_save_time + save_interval - current_state.time
                
            # Take symplectic step
            new_state = integrator.step(current_state, dt)
            
            # Compute divergence of new state using periodic gradients
            div = (solver._periodic_gradient(new_state.velocity.u_component, du, 0) +
                   solver._periodic_gradient(new_state.velocity.v_component, dv, 1) +
                   solver._periodic_gradient(new_state.velocity.w_component, dr, 2))
            max_div = np.max(np.abs(div))
            
            # Debug: Print divergence every 10 steps (on new line)
            if step % 10 == 0:
                new_energy = new_state.energy_3d()
                print(f"{new_state.time:5.4f}  {new_energy:8.4f}  {new_energy/E0:6.3f}  "
                      f"{new_state.helicity_integral():8.4f}  {np.max(new_state.velocity.magnitude()):7.4f}  "
                      f"{max_div:8.6f}  {dt:6.4f}  {time.time() - start_time:7.1f}s")
            
            # Check for instability
            new_energy = new_state.energy_3d()
            if new_energy > 100 * E0 or np.isnan(new_energy) or max_div > 100:
                print(f"\nInstability detected at t={new_state.time:.3f}")
                print(f"  Energy: {new_energy:.6f}")
                print(f"  Max divergence: {max_div:.6f}")
                
                # Detailed divergence analysis
                div_mean = np.mean(np.abs(div))
                div_std = np.std(div)
                print(f"  Divergence stats: mean={div_mean:.6f}, std={div_std:.6f}")
                
                # Check where divergence is highest
                max_idx = np.unravel_index(np.argmax(np.abs(div)), div.shape)
                print(f"  Max divergence at grid point: {max_idx}")
                print(f"  Velocity at max div point: u={new_state.velocity.u_component[max_idx]:.4f}, "
                      f"v={new_state.velocity.v_component[max_idx]:.4f}, "
                      f"w={new_state.velocity.w_component[max_idx]:.4f}")
                break
                
            current_state = new_state
            step += 1
            
            # Save data at intervals
            if current_state.time >= last_save_time + save_interval - 1e-10:
                times.append(current_state.time)
                energies.append(new_energy)
                helicities.append(current_state.helicity_integral())
                max_velocities.append(np.max(current_state.velocity.magnitude()))
                divergences.append(max_div)
                
                last_save_time = current_state.time
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"{current_state.time:5.2f}  {new_energy:8.4f}  {new_energy/E0:6.3f}  "
                      f"{helicities[-1]:8.4f}  {max_velocities[-1]:7.4f}  "
                      f"{max_div:8.6f}  {dt:6.4f}  {elapsed:7.1f}s")
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        
    # Final summary
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print(f"Final time: {current_state.time:.4f}")
    print(f"Total steps: {step}")
    print(f"Computation time: {time.time() - start_time:.1f} seconds")
    print(f"Performance: {step/(time.time() - start_time):.1f} steps/second")
    
    # Get actual final state values from current_state
    final_energy = current_state.energy_3d()
    final_helicity = current_state.helicity_integral()
    final_max_vel = np.max(current_state.velocity.magnitude())
    final_div = (np.gradient(current_state.velocity.u_component, du, axis=0) +
                 np.gradient(current_state.velocity.v_component, dv, axis=1) +
                 np.gradient(current_state.velocity.w_component, dr, axis=2))
    final_max_div = np.max(np.abs(final_div))
    
    print(f"\nFinal state:")
    print(f"  Energy: {final_energy:.4f} ({100*final_energy/E0:.1f}% of initial)")
    print(f"  Helicity: {final_helicity:.4f}")
    print(f"  Max velocity: {final_max_vel:.4f}")
    print(f"  Max divergence: {final_max_div:.6f}")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Energy evolution
    ax = axes[0, 0]
    times_array = np.array(times)
    energies_array = np.array(energies)
    ax.semilogy(times_array, energies_array, 'b-', linewidth=2)
    ax.axvline(1.0, color='r', linestyle='--', label='t=1.0')
    ax.axvline(2.0, color='orange', linestyle='--', label='t=2.0')
    ax.axvline(3.0, color='g', linestyle='--', label='t=3.0')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy (log scale)')
    ax.set_title('Energy Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Helicity conservation
    ax = axes[0, 1]
    if len(helicities) > 1:
        ax.plot(times_array, helicities, 'g-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Helicity')
        ax.set_title('Helicity Conservation')
        ax.grid(True, alpha=0.3)
    
    # Maximum velocity
    ax = axes[0, 2]
    if len(max_velocities) > 1:
        ax.plot(times_array, max_velocities, 'r-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |u|')
        ax.set_title('Maximum Velocity')
        ax.grid(True, alpha=0.3)
    
    # Normalized energy
    ax = axes[1, 0]
    ax.plot(times_array, energies_array/E0, 'b-', linewidth=2)
    ax.axhline(0.1, color='gray', linestyle=':', label='10%')
    ax.axhline(0.01, color='gray', linestyle=':', label='1%')
    ax.set_xlabel('Time')
    ax.set_ylabel('E(t)/E(0)')
    ax.set_title('Normalized Energy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.2)
    
    # Divergence evolution
    ax = axes[1, 1]
    if len(divergences) > 1:
        ax.semilogy(times_array, divergences, 'm-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |div u|')
        ax.set_title('Maximum Divergence')
        ax.grid(True, alpha=0.3)
    
    # Energy dissipation rate
    ax = axes[1, 2]
    if len(energies) > 2:
        dt_save = times_array[1] - times_array[0]
        dE_dt = np.gradient(energies_array, times_array)
        ax.plot(times_array[1:], -dE_dt[1:]/energies_array[1:], 'c-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('-dE/dt / E')
        ax.set_title('Energy Dissipation Rate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppf_3d_symplectic_analysis.png', dpi=150)
    
    # Create visualization of final state
    print("\nAnalysis plots saved to ppf_3d_symplectic_analysis.png")
    
    # Visualize final flow field
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get middle slices
    mid_r = current_state.velocity.u_component.shape[2] // 2
    
    # Velocity magnitude at r=r/2
    state = current_state
    vel_mag_r = np.sqrt(state.velocity.u_component[:, :, mid_r]**2 + 
                        state.velocity.v_component[:, :, mid_r]**2 +
                        state.velocity.w_component[:, :, mid_r]**2)
    
    ax1 = axes[0]
    im1 = ax1.contourf(state.u_grid[:, :, mid_r], state.v_grid[:, :, mid_r], 
                       vel_mag_r, levels=20, cmap='viridis')
    ax1.set_xlabel('u (toroidal)')
    ax1.set_ylabel('v (poloidal)')
    ax1.set_title(f'|u| at r={state.r_grid[0,0,mid_r]:.2f}, t={state.time:.2f}')
    plt.colorbar(im1, ax=ax1)
    
    # Vorticity (simplified for 3D)
    vorticity_mag = np.sqrt(
        np.gradient(state.velocity.w_component[:, :, mid_r], dv, axis=1)**2 +
        np.gradient(state.velocity.u_component[:, :, mid_r], dr, axis=0)**2
    )
    
    ax2 = axes[1]
    im2 = ax2.contourf(state.u_grid[:, :, mid_r], state.v_grid[:, :, mid_r],
                       vorticity_mag, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('u (toroidal)')
    ax2.set_ylabel('v (poloidal)')
    ax2.set_title(f'Vorticity magnitude at t={state.time:.2f}')
    plt.colorbar(im2, ax=ax2)
    
    # Factorization complexity
    ax3 = axes[2]
    complexity = state.state_cardinalities[:, :, mid_r]
    im3 = ax3.contourf(state.u_grid[:, :, mid_r], state.v_grid[:, :, mid_r],
                       complexity, levels=20, cmap='hot')
    ax3.set_xlabel('u (toroidal)')
    ax3.set_ylabel('v (poloidal)')
    ax3.set_title(f'Factorization Complexity at t={state.time:.2f}')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(f'ppf_3d_symplectic_state_t{state.time:.1f}.png', dpi=150)
    print(f"Final state visualization saved to ppf_3d_symplectic_state_t{state.time:.1f}.png")
    
    plt.show()
    

if __name__ == "__main__":
    run_highres_long_simulation()