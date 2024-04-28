"""
setup.py is used to build an application (say ML) as a package.
    
"""

# find_packges identifies all the packages used in the application
from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."

# a funciton to read lines in requirements.txt (file_path)
def get_requirements(file_path):
    # requirements= []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

# metadata of the application
setup(
name= "MedicalCostPreditcton",
version="0.0.1",
author="Rohit",
author_email="rohitmtak@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)        