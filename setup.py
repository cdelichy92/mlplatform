from distutils.core import setup

setup(
    name="mlplatform",
    version="1.0",
    author="Cyprien de Lichy",
    author_email="cyprien.delichy@gmail.com",

    packages=["mlplatform"],

    include_package_data=True,

    scripts=['bin/run_app'],

)
