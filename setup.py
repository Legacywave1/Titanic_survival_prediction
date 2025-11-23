from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    try:
        with open(file_path, 'r') as file:
            requirements = file.readlines()
            requirements = [i.replace('\n', '') for i in requirements]
            requirements = [req for req in requirements if req and not req.startswith('#')]
            if hypen_e_dot in requirements:
                requirements.remove(hypen_e_dot)
    except FileNotFoundError:
        print(f'Warning: {file_path} not Found')
    return requirements

setup(
    name = 'Titanic-prediction',
    version = '0.0.2',
    author = 'Cyprian',
    author_email = 'cypriananku121@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )
