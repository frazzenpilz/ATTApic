import sys
from pic_modules.parameters import Parameters
from pic_modules.patch import VectorPatch
from pic_modules.utils import log_messages

def main():
    # Process input parameters
    parameters_list = Parameters("inputs.txt")
    
    # Clean output directory if it exists to overwrite previous results
    import os
    import shutil
    output_dir = "output"
    if os.path.exists(output_dir):
        try:
            # Remove all files in output directory but keep the directory
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            log_messages(f"Cleaned output directory: {output_dir}", __file__, sys._getframe().f_lineno, 1)
        except Exception as e:
            log_messages(f"Error cleaning output directory: {e}", __file__, sys._getframe().f_lineno, 3)
            
    log_messages("Starting simulation", __file__, sys._getframe().f_lineno, 1)

    if parameters_list.num_errors == 0:
        parameters_list.assign_inputs()
        parameters_list.process_mesh("PIC")

    # Commence simulation
    if parameters_list.num_errors == 0:
        patches_vector = VectorPatch(parameters_list)
        patches_vector.start_pic()
        
        parameters_list.num_errors = patches_vector.num_errors

    if parameters_list.num_errors != 0:
        log_messages("Simulation complete, exited UNSUCCESSFULLY", __file__, sys._getframe().f_lineno, 1)
        sys.exit(1)
    else:
        log_messages("Simulation complete, exited successfully", __file__, sys._getframe().f_lineno, 1)
        sys.exit(0)

if __name__ == "__main__":
    main()
