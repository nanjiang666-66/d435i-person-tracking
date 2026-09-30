from setuptools import setup


package_name = "person_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["bytetrack_recovery.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nanjiang666-66",
    maintainer_email="254610277+nanjiang666-66@users.noreply.github.com",
    description="D435i CPU person detection, tracking and aligned-depth output",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "person_tracker = person_vision.launcher:main",
        ],
    },
)
