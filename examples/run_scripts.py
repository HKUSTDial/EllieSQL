import subprocess

def run_scripts():
    conda_env = "ElephantSQL"
    # spider dev
    # DC+R DC+OS+R
    # spider test
    # DC+R DC+OS+R

    # modules = ["examples.main", "examples.m2", "examples.m3", "examples.m4"]


    # bird dev 4o
    # vanilla, DC+R, OS+R, DC+OS+R
    modules = ["examples.main", "examples.m2", "examples.m3", "examples.m4"]
    for module in modules:
        cmd = f"conda activate {conda_env} && python -m {module}"
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run_scripts()
