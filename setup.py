import setuptools

with open("requirements.txt") as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="Zoidberg2.0",
    description="Premier projet d'IA d'Epitech : - Classification de Chest XRay ",
    url="https://github.com/Lenoush/ZOIDBERG2.0",
    url_epitech="https://github.com/EpitechMscProPromo2025/T-DEV-810-PAR_26.git",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6",
)
