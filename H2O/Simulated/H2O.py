import math
import numpy as np
import itertools
import hoomd
import gsd.hoomd

device = hoomd.device.CPU()
#check that all mpi processes are communicating with hoomd
print('Hello from Rank',device.communicator.rank)

# AKMA Units Convention See Hoomd Units Page
path = "water_sim_data"
N_molecules = 20_000
spacing = 6
timestep = 0.005
k   = 0.00198720426    # boltzmann constant    [kcal] [mol]^-1 [K]^-1  
av  = 6.0223e23        # avagadro number       [particle/mol]  
T   = 365.15           # temperature           [K]
kbT = k * T            #                       [kcal/mol]
density = 0.09681      #                       [number of atoms] [Å^3]^-1
eq_time = 50_000
production_time = 50_000
stride = 200

n_OO = 12                  # []
sigma_OO = 3.1507          # [Å]
epsilon_OO = 0.1521        # [kcal/mol]

n_OH = 12                  # []
sigma_OH = 1.7753          # [Å]
epsilon_OH = 0.0836        # [kcal/mol]

n_HH = 12                  # []
sigma_HH = 0.4             # [Å]
epsilon_HH = 0.0460        # [kcal/mol]

k_r = 450/4                # [kcal/mol]
r_0 = 0.9572               # [Å]

k_¸ = 55                   # [kcal/mol]
¸_0 = (104.52*np.pi)/180   # [rad]

O_q = -0.834/0.05487686461 # [0.05487686461 e]
H_q = 0.417/0.05487686461  # [0.05487686461 e]

O_m = 15.9994              # [amu]
H_m = 1.008                # [amu]

if device.communicator.rank == 0:
    
    O_r = np.zeros(3)
    H1_r = np.array([r_0,0,0])
    H2_r = np.array([r_0*np.cos(¸_0),r_0*np.sin(¸_0),0])
    
    CM_r = H1_r*H_m + H2_r*H_m + O_m*O_r
    CM_r /= O_m + 2*H_m
    
    O_r -= CM_r
    H1_r -= CM_r
    H2_r -= CM_r
    
    positions_molecule = np.array([O_r,H1_r,H2_r])
    
    K = math.ceil(N_molecules**(1 / 3))
    L = (K+1) * spacing * sigma_OO
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position_lattice = list(itertools.product(x, repeat=3)) 
    positions_molecule = np.array([O_r,H1_r,H2_r])
    
    positions = np.zeros((K**3*3,3))
    i = 0
    for pos in position_lattice:
        for atom in positions_molecule:
            positions[i] = pos + atom
            i += 1
    
    positions += [spacing / 2, spacing / 2, spacing / 2]
    
    snapshot = gsd.hoomd.Frame()
    # Place an h2o in the box.
    snapshot.particles.N = K**3 * 3
    snapshot.particles.position = positions
    snapshot.particles.types = ['O','H']
    snapshot.particles.typeid = [0,1,1] * K**3        
    snapshot.configuration.box = [L, L, L, 0, 0, 0]
    snapshot.particles.mass = [O_m,H_m,H_m]* K**3

    bonds_molecule = np.array([[0, 1], [0, 2]])
    bonds = np.zeros((2*K**3,2))
    
    for i in range(K**3):
        bonds[2*i:(2*i)+2] = bonds_molecule + 3*i
    
    # Connect particles with bonds.
    snapshot.bonds.N = K**3 * 2
    snapshot.bonds.types = ['O-H']
    snapshot.bonds.typeid = [0] * K**3  * 2
    snapshot.bonds.group = bonds
    
    angles_molecule = np.array([1,0,2])
    angles = np.zeros((K**3,3))
    
    for i in range(K**3):
        angles[i] = angles_molecule + 3*i
    
    # Connect particles with angles.
    snapshot.angles.N = K**3
    snapshot.angles.types = ['H-O-H']
    snapshot.angles.typeid = [0] * K**3
    snapshot.angles.group = angles    

    with gsd.hoomd.open(name=path+'/lattice/h2o_lattice.gsd', mode='w') as f:
        f.append(snapshot)    

# Pair hoomd to the cpu and create basic sim objects
sim = hoomd.Simulation(device=device, seed=42069)
sim.create_state_from_gsd(filename=path+'/lattice/h2o_lattice.gsd')

with sim.state.cpu_local_snapshot as snapshot:
    typeid = snapshot.particles.typeid
    snapshot.particles.charge[typeid == 1] = H_q
    snapshot.particles.charge[typeid == 0] = O_q

domain_decomposition = sim.state.domain_decomposition
split_fractions = sim.state.domain_decomposition_split_fractions
if device.communicator.rank == 0:
    # Print the domain decomposition.
    print("Domain decomposition:")
    print(domain_decomposition)
    print()
    
    # Print the location of the split planes.
    print("Split fractions:")
    print(split_fractions)
    print()
    
# Print the number of particles on each rank before equilibration.
with sim.state.cpu_local_snapshot as snap:
    N = len(snap.particles.position)
    print(f'{N} particles on rank {device.communicator.rank}')

integrator = hoomd.md.Integrator(dt = timestep)
cell = hoomd.md.nlist.Cell(buffer=0.4)

if device.communicator.rank == 0:
    print()
    print("Computed cells with buffer...")
    print()

# Non-Bonded potential energy function
mie = hoomd.md.pair.Mie(nlist=cell)

mie.params[('O', 'O')] = dict(epsilon=epsilon_OO, sigma=sigma_OO, n=n_OO, m=6)
mie.r_cut[('O', 'O')] = 3*sigma_OO

mie.params[('O', 'H')] = dict(epsilon=epsilon_OH, sigma=sigma_OH, n=n_OH, m=6)
mie.r_cut[('O', 'H')] = 3*sigma_OH

mie.params[('H', 'H')] = dict(epsilon=epsilon_HH, sigma=sigma_HH, n=n_HH, m=6)
mie.r_cut[('H', 'H')] = 3*sigma_HH
integrator.forces.append(mie)

# Bonded potentential energy function

harmonic_angle = hoomd.md.angle.Harmonic()
harmonic_angle.params['H-O-H'] = dict(k=k_¸, t0=¸_0)
integrator.forces.append(harmonic_angle)

harmonic_bond = hoomd.md.bond.Harmonic()
harmonic_bond.params['O-H'] = dict(k=k_r, r0=r_0)
integrator.forces.append(harmonic_bond)

real_space_force, reciprocal_space_force = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(nlist=cell, 
                                                                                             order = 4, 
                                                                                             resolution = (64,64,64), 
                                                                                             alpha=0,
                                                                                             r_cut = 3*sigma_OO)
integrator.forces.append(reciprocal_space_force)
integrator.forces.append(reciprocal_space_force)

if device.communicator.rank == 0:
    print("Created Potential Energy...")
    print()

# Create thermostat and fix the volume to make an NVT sim
mttk = hoomd.md.methods.thermostats.MTTK(kT=kbT, tau=100*timestep)
nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),thermostat=mttk)
integrator.methods.append(nvt) # Binds it to the integrator
sim.operations.integrator = integrator

if device.communicator.rank == 0:
    print("Created Thermostat...")
    print()

# Thermalize the simulation
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kbT)
sim.run(100)

if device.communicator.rank == 0:
    print("Thermalized Simulation...")
    print()

# Use the input target density to indicate how to rehsape the box
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000) # Indicates how fast to apply box reszizing
rho = sim.state.N_particles / sim.state.box.volume
initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box) 
final_rho = density
final_box.volume = sim.state.N_particles / final_rho
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
sim.run(10_000)
sim.operations.updaters.remove(box_resize)

if device.communicator.rank == 0:
    print("Resized Box...")
    print()

# Equilibrate Simulation
sim.run(eq_time)

if device.communicator.rank == 0:
    print("Equilibrated Simulation...")
    print()

# Print the number of particles on each rank after equilibration.
with sim.state.cpu_local_snapshot as snap:
    N = len(snap.particles.position)
    print(f'{N} particles on rank {device.communicator.rank}')

# Production Period to Gather Observations
gsd_writer = hoomd.write.GSD(filename= path + '/traj/h2o.gsd',
                          trigger=hoomd.trigger.Periodic(stride),
                          mode='wb')

if device.communicator.rank == 0:
    print()
    print("Starting Production Period...")
    print()

sim.operations.writers.append(gsd_writer)
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
logger = hoomd.logging.Logger()
logger.add(thermodynamic_properties)
gsd_writer.log = logger
sim.run(production_time)
gsd_writer.flush()

if device.communicator.rank == 0:
    print("Simulation Complete...")
    print()