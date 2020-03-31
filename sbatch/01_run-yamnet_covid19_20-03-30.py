import glob
import os

script_name = os.path.basename(__file__)

# Define constants
in_data_dir = "/scratch/mc6591/covid_audio"
out_data_dir = "/scratch/vl1019/c19_data"
glob_regexp = os.path.join(in_data_dir, "sonycnode-*.sonyc")

# Define script name
script_name = os.path.basename(__file__)
script_path = os.path.join("..", "..", "..", "src", script_name)

# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)

# Create SLURM sbatch files. Loop over sensors.
sensor_dirs = glob.glob(glob_regexp)
for sensor_dir in sensor_dirs:
    sensor_dirname = os.path.split(sensor_dir)[1]
    sonycnode_str = os.path.splitext(sensor_dirname)[0]
    job_name = "_".join([script_name[:2], sonycnode_str])
    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)
    script_path_with_args = " ".join([script_name, sensor_dir, out_data_dir])
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + job_name + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --time=12:00:00\n")
        f.write("#SBATCH --mem=1GB\n")
        f.write("#SBATCH --output=" + "../slurm/" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("conda activate c19\n")
        f.write("\n")
        f.write("# The first argument is the path to the SONYC sensor.\n")
        f.write("# The second argument is the path to the output directory.\n")
        f.write("python " + script_path_with_args)

# Open shell file.
file_path = os.path.join(sbatch_dir, script_name[:2] + ".sh")
with open(file_path, "w") as f:
    # Print header
    f.write("# This shell script executes all Slurm jobs" +
            "for running YAMNet on SONYC recordings.\n")
    f.write("\n")

    # Loop over SONYC sensors.
    for sensor_dir in sensor_dirs:
        sensor_dirname = os.path.split(sensor_dir)[1]
        sonycnode_str = os.path.splitext(sensor_dirname)[0]
        job_name = "_".join([script_name[:2], sonycnode_str])
        sbatch_str = "sbatch " + job_name + ".sbatch"
        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")
