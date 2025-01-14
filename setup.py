import setuptools

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements_list = req_file.read().strip().split("\n")

setuptools.setup(
    name="u2net",
    packages=setuptools.find_packages(include=['u2net', 'u2net.*']),
    install_requires=requirements_list,
    include_package_data=True,
)