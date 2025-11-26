import os
import sys
from .utils import log_messages, log_brief

class Parameters:
    def __init__(self, filename="inputs.txt"):
        self.filename = filename
        self.values_vector = []
        self.file_not_opened = False
        self.use_default_argument = False
        self.num_errors = 0

        # Global simulation parameters
        self.time_step = 0.0001
        self.maximum_number_of_iterations = 10
        self.number_of_patches = 1
        self.minimum_particles_per_cell = 4
        self.maximum_particles_per_cell = 10
        self.specific_weight = 1.0
        self.simulation_type = "electron"
        self.axisymmetric = False
        self.two_stream = False

        # Particle and collision parameters
        self.initial_particles_per_cell = 1
        self.num_cells_with_particles = 0
        self.particle_distribution = "random"
        self.initial_temperature = 1000.0
        self.initial_position = [0.5, 0.5]
        self.initial_velocity = [0.0, 0.0]
        self.inlet_source = False
        self.inlet_size_percent = 0.1
        self.inlet_flow_rate = 0.0
        self.inlet_velocity = 1.0
        self.propellant = "xenon"
        self.mcc_frequency = 10

        # Field and FDTD parameters
        self.e_field = [0.0, 0.0, 0.0]
        self.b_field = [0.0, 0.0, 0.0]
        self.fdtd_iterations = 1000
        self.fdtd_frequency = 10
        
        # RF Coil parameters
        self.use_rf_coil = False
        self.coil_turns = 10
        self.coil_radius = 0.06  # m
        self.coil_length = 0.08  # m
        self.coil_current = 100.0  # A
        self.rf_frequency = 13.56e6  # Hz (typical ICP frequency)

        # Mesh and domain parameters
        self.user_mesh = True
        self.domain_length = 0.1
        self.domain_height = 0.06
        self.pic_spacing = 0.02
        self.fdtd_spacing = 0.005
        self.mesh_file_path = "coarseMesh.su2"
        self.mesh_scaling_parameter = 1.0

        # Solver and boundary condition parameters
        self.solver_type = "GS"
        self.max_solver_iterations = 50
        self.residual_tolerance = 1e-20
        self.sor_parameter = 1.1
        self.left_bc_type = "dirichlet"
        self.left_bc_value = 0.0
        self.right_bc_type = "dirichlet"
        self.right_bc_value = 0.0
        self.top_bc_type = "dirichlet"
        self.top_bc_value = 0.0
        self.bottom_bc_type = "dirichlet"
        self.bottom_bc_value = 0.0

        # Parallelisation parameters
        self.num_threads = 1

        # Output parameters
        self.plot_frequency = 5
        self.tecplot_mesh = "cMesh"
        self.tecplot_particle_solution = "cSolution_P"
        self.tecplot_node_solution = "cSolution_N"
        self.tecplot_global_solution = "cSolution_G"

        self.read_inputs()

    def read_inputs(self):
        log_messages("Reading inputs", __file__, sys._getframe().f_lineno, 1)
        
        if not os.path.exists(self.filename):
            log_messages("Unable to open input file", __file__, sys._getframe().f_lineno, 3)
            self.file_not_opened = True
            return

        try:
            with open(self.filename, 'r') as f:
                content = f.read()
                
            # Tokenize based on whitespace
            tokens = content.split()
            
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                # Skip comments (start with %)
                if token.startswith('%'):
                    # Skip until next line? 
                    # C++ logic: if first char is %, ignore line.
                    # But here we already split by whitespace, so newlines are gone.
                    # This approach is tricky if we lost line info.
                    # Let's go back to line processing but handle space separation.
                    pass
                
                i += 1
            
            # Better approach: Line by line, then split
            self.values_vector = []
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    
                    # Remove comments within line? C++ doesn't seem to do that explicitly 
                    # except checking if first char is %.
                    
                    # Split by whitespace
                    parts = line.split()
                    if not parts:
                        continue
                        
                    # C++ logic:
                    # inputFile >> name >> value;
                    # So it expects pairs.
                    # But sometimes value might be missing?
                    
                    # My inputs.txt has "name: value" or "name value".
                    # "name:" is one token if no space? No, "name:" is usually "name" then ":" then "value"?
                    # Actually C++ code:
                    # inputFile >> name >> value;
                    # if (value.front() != '%' && value.back() != ':') -> push_back(value)
                    
                    # If line is "timeStep: 0.00004", parts are ["timeStep:", "0.00004"]
                    # name="timeStep:", value="0.00004"
                    # value doesn't end with ':', so push.
                    
                    # If line is "inletFlowRate 6e-26", parts=["inletFlowRate", "6e-26"]
                    # name="inletFlowRate", value="6e-26"
                    
                    # If line is empty or comment, skipped.
                    
                    # We need to process all tokens in the file sequentially.
                    pass

            # Re-implementing token stream approach correctly
            with open(self.filename, 'r') as f:
                lines = f.readlines()
            
            tokens = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                tokens.extend(line.split())
            
            i = 0
            while i < len(tokens):
                name = tokens[i]
                if i + 1 < len(tokens):
                    value = tokens[i+1]
                    
                    # Check if value looks like a key (ends with :) or comment
                    # C++ check: if (value.front() != '%' && value.back() != ':')
                    if not value.startswith('%') and not value.endswith(':'):
                        self.values_vector.append(value)
                        i += 2 # Consumed name and value
                    else:
                        # Value is missing or is next key
                        self.values_vector.append("DEFAULT")
                        i += 1 # Consumed name only
                else:
                    # Last token, no value
                    self.values_vector.append("DEFAULT")
                    i += 1

        except Exception as e:
            log_messages(f"Error reading file: {e}", __file__, sys._getframe().f_lineno, 3)
            self.file_not_opened = True

    def assign_inputs(self):
        log_messages("Printing input parameters", __file__, sys._getframe().f_lineno, 1)
        if self.file_not_opened:
            return

        index = 0
        
        # Helper to get value safely
        def get_value(idx, default, converter=str, validator=None):
            val = default
            try:
                if idx < len(self.values_vector) and self.values_vector[idx] != "DEFAULT":
                    val = converter(self.values_vector[idx])
                    if validator and not validator(val):
                        raise ValueError("Validation failed")
                else:
                    raise ValueError("Default requested")
            except Exception:
                log_brief(f"Using default for parameter at index {idx}", 2)
                val = default
            return val

        # Global simulation parameters
        self.time_step = get_value(index, 0.0001, float, lambda x: x > 0)
        log_brief(f"Time step: {self.time_step}", 1)
        index += 1

        self.maximum_number_of_iterations = get_value(index, 10, int, lambda x: x > 0)
        log_brief(f"Maximum number of iterations: {self.maximum_number_of_iterations}", 1)
        index += 1

        self.number_of_patches = get_value(index, 1, int, lambda x: x > 0)
        log_brief(f"Number of patches: {self.number_of_patches}", 1)
        index += 1

        self.minimum_particles_per_cell = get_value(index, 4, int, lambda x: x > 0)
        log_brief(f"Minimum particles per cell: {self.minimum_particles_per_cell}", 1)
        index += 1

        self.maximum_particles_per_cell = get_value(index, 6 + self.minimum_particles_per_cell, int, lambda x: x > self.minimum_particles_per_cell)
        log_brief(f"Maximum particles per cell: {self.maximum_particles_per_cell}", 1)
        index += 1

        self.specific_weight = get_value(index, 1.0, float, lambda x: x >= 1.0)
        log_brief(f"Specific weight: {self.specific_weight}", 1)
        index += 1

        self.simulation_type = get_value(index, "electron", str, lambda x: x in ["partial", "electron"])
        log_brief(f"Simulation type: {self.simulation_type}", 1)
        index += 1

        self.axisymmetric = get_value(index, 0, int, lambda x: x in [0, 1]) == 1
        log_brief(f"Axisymmetric flag: {self.axisymmetric}", 1)
        index += 1

        self.two_stream = get_value(index, 0, int, lambda x: x in [0, 1]) == 1
        log_brief(f"Two-stream flag: {self.two_stream}", 1)
        index += 1

        # Particle and collision parameters
        self.initial_particles_per_cell = get_value(index, 1, int, lambda x: x >= 0)
        log_brief(f"Initial particles per cell: {self.initial_particles_per_cell}", 1)
        index += 1

        self.num_cells_with_particles = get_value(index, 0, int, lambda x: x >= 0)
        log_brief(f"Number of cells with particles: {self.num_cells_with_particles}", 1)
        index += 1

        self.particle_distribution = get_value(index, "random", str, lambda x: x in ["uniform", "random", "precise"])
        log_brief(f"Particle distribution: {self.particle_distribution}", 1)
        index += 1

        self.initial_temperature = get_value(index, 1000.0, float, lambda x: x >= 0)
        log_brief(f"Initial temperature: {self.initial_temperature}", 1)
        index += 1

        def parse_list(s, dtype=float):
            return [dtype(x) for x in s.split(',')]

        self.initial_position = get_value(index, [0.5, 0.5], parse_list)
        log_brief(f"Initial position: {self.initial_position}", 1)
        index += 1

        self.initial_velocity = get_value(index, [0.0, 0.0], parse_list)
        log_brief(f"Initial velocity: {self.initial_velocity}", 1)
        index += 1

        self.inlet_source = get_value(index, 0, int, lambda x: x in [0, 1]) == 1
        log_brief(f"Inlet source: {self.inlet_source}", 1)
        index += 1

        self.inlet_size_percent = get_value(index, 0.1, float, lambda x: 0.0 <= x <= 1.0)
        log_brief(f"Inlet size: {self.inlet_size_percent}", 1)
        index += 1

        self.inlet_flow_rate = get_value(index, 0.0, float, lambda x: x >= 0)
        log_brief(f"Inlet flow rate: {self.inlet_flow_rate}", 1)
        index += 1

        self.inlet_velocity = get_value(index, 1.0, float)
        log_brief(f"Inlet velocity: {self.inlet_velocity}", 1)
        index += 1

        self.propellant = get_value(index, "xenon", str)
        log_brief(f"Propellant: {self.propellant}", 1)
        index += 1

        self.mcc_frequency = get_value(index, 10, int, lambda x: x > 0)
        log_brief(f"MCC frequency: {self.mcc_frequency}", 1)
        index += 1

        # Poisson solver parameter (added to fix index alignment)
        self.use_poisson = get_value(index, 0, int, lambda x: x in [0, 1]) == 1
        log_brief(f"Use Poisson solver: {self.use_poisson}", 1)
        index += 1

        # Field and FDTD parameters
        self.e_field = get_value(index, [0.0, 0.0, 0.0], parse_list)
        log_brief(f"E-field: {self.e_field}", 1)
        index += 1

        self.b_field = get_value(index, [0.0, 0.0, 0.0], parse_list)
        log_brief(f"B-field: {self.b_field}", 1)
        index += 1

        self.fdtd_iterations = get_value(index, 1000, int, lambda x: x > 0)
        log_brief(f"FDTD iterations: {self.fdtd_iterations}", 1)
        index += 1

        self.fdtd_frequency = get_value(index, 10, int, lambda x: x > 0)
        log_brief(f"FDTD frequency: {self.fdtd_frequency}", 1)
        index += 1
        
        # RF Coil parameters
        self.use_rf_coil = get_value(index, 0, int, lambda x: x in [0, 1]) == 1
        log_brief(f"Use RF coil: {self.use_rf_coil}", 1)
        index += 1
        
        self.coil_turns = get_value(index, 10, int, lambda x: x > 0)
        log_brief(f"Coil turns: {self.coil_turns}", 1)
        index += 1
        
        self.coil_radius = get_value(index, 0.06, float, lambda x: x > 0)
        log_brief(f"Coil radius: {self.coil_radius}", 1)
        index += 1
        
        self.coil_length = get_value(index, 0.08, float, lambda x: x > 0)
        log_brief(f"Coil length: {self.coil_length}", 1)
        index += 1
        
        self.coil_current = get_value(index, 100.0, float, lambda x: x > 0)
        log_brief(f"Coil current: {self.coil_current}", 1)
        index += 1
        
        self.rf_frequency = get_value(index, 13.56e6, float, lambda x: x > 0)
        log_brief(f"RF frequency: {self.rf_frequency}", 1)
        index += 1

        # Mesh and domain parameters
        self.user_mesh = get_value(index, 1, int, lambda x: x in [0, 1]) == 1
        log_brief(f"User mesh: {self.user_mesh}", 1)
        index += 1

        self.domain_length = get_value(index, 0.1, float, lambda x: x > 0)
        log_brief(f"Domain length: {self.domain_length}", 1)
        index += 1

        self.domain_height = get_value(index, 0.06, float, lambda x: x > 0)
        log_brief(f"Domain height: {self.domain_height}", 1)
        index += 1

        self.pic_spacing = get_value(index, 0.02, float, lambda x: x > 0)
        log_brief(f"PIC spacing: {self.pic_spacing}", 1)
        index += 1

        self.fdtd_spacing = get_value(index, 0.005, float, lambda x: x > 0)
        log_brief(f"FDTD spacing: {self.fdtd_spacing}", 1)
        index += 1

        self.mesh_file_path = get_value(index, "coarseMesh.su2", str)
        log_brief(f"Mesh file path: {self.mesh_file_path}", 1)
        index += 1

        self.mesh_scaling_parameter = get_value(index, 1.0, float, lambda x: x > 0)
        log_brief(f"Mesh scaling parameter: {self.mesh_scaling_parameter}", 1)
        index += 1

        # Solver and boundary condition parameters
        self.solver_type = get_value(index, "GS", str, lambda x: x in ["GS", "SOR", "MultiGrid"])
        log_brief(f"Solver type: {self.solver_type}", 1)
        index += 1

        self.max_solver_iterations = get_value(index, 50, int, lambda x: x > 0)
        log_brief(f"Max solver iterations: {self.max_solver_iterations}", 1)
        index += 1

        self.residual_tolerance = get_value(index, 1e-20, float, lambda x: x > 0)
        log_brief(f"Residual tolerance: {self.residual_tolerance}", 1)
        index += 1

        self.sor_parameter = get_value(index, 1.1, float, lambda x: 1.0 <= x < 2.0)
        log_brief(f"SOR parameter: {self.sor_parameter}", 1)
        index += 1

        self.left_bc_type = get_value(index, "dirichlet", str)
        log_brief(f"Left BC type: {self.left_bc_type}", 1)
        index += 1

        self.left_bc_value = get_value(index, 0.0, float)
        log_brief(f"Left BC value: {self.left_bc_value}", 1)
        index += 1

        self.right_bc_type = get_value(index, "dirichlet", str)
        log_brief(f"Right BC type: {self.right_bc_type}", 1)
        index += 1

        self.right_bc_value = get_value(index, 0.0, float)
        log_brief(f"Right BC value: {self.right_bc_value}", 1)
        index += 1

        self.top_bc_type = get_value(index, "dirichlet", str)
        log_brief(f"Top BC type: {self.top_bc_type}", 1)
        index += 1

        self.top_bc_value = get_value(index, 0.0, float)
        log_brief(f"Top BC value: {self.top_bc_value}", 1)
        index += 1

        self.bottom_bc_type = get_value(index, "dirichlet", str)
        log_brief(f"Bottom BC type: {self.bottom_bc_type}", 1)
        index += 1

        self.bottom_bc_value = get_value(index, 0.0, float)
        log_brief(f"Bottom BC value: {self.bottom_bc_value}", 1)
        index += 1

        # Parallelisation parameters
        self.num_threads = get_value(index, 1, int, lambda x: x > 0)
        log_brief(f"Number of threads: {self.num_threads}", 1)
        index += 1

        # Output parameters
        self.plot_frequency = get_value(index, 5, int, lambda x: x > 0)
        log_brief(f"Plot frequency: {self.plot_frequency}", 1)
        index += 1

        self.tecplot_mesh = get_value(index, "cMesh", str)
        log_brief(f"Tecplot mesh: {self.tecplot_mesh}", 1)
        index += 1

        self.tecplot_particle_solution = get_value(index, "cSolution_P", str)
        log_brief(f"Tecplot particle solution: {self.tecplot_particle_solution}", 1)
        index += 1

        self.tecplot_node_solution = get_value(index, "cSolution_N", str)
        log_brief(f"Tecplot node solution: {self.tecplot_node_solution}", 1)
        index += 1

        self.tecplot_global_solution = get_value(index, "cSolution_G", str)
        log_brief(f"Tecplot global solution: {self.tecplot_global_solution}", 1)
        index += 1

    def process_mesh(self, mesh_type):
        # Placeholder for mesh processing trigger
        log_messages(f"Processing mesh type: {mesh_type}", __file__, sys._getframe().f_lineno, 1)
