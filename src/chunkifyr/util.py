import subprocess
import sys

# function to install a package
def install_package(pkg):
    subprocess.check_call([sys.executable, "-m" "pip", "install", "--upgrade", "--quiet", pkg])
