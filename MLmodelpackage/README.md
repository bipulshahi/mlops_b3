# this is loan approval project

# create requirement.txt
# we will list all packages and modules here to be installed in server
# by executing "pip install -r requirements.txt" 
# we can install all required dependecies in any system in which we need to execute our project

## create virtual Environment
Install virtualenv

'''
python3 -m pip install virtualenv
'''

Check version
'''
virtualenv --version
'''

Create a virtual environment
'''
virtualenv ml_package
'''

# Activate virtual environment
**Windows**
ml_package\Scripts\activate
**Linux or Mac**
source ml_package/bin/activate

# Change and mode into the project directory
# install all dependecies in the virtual environemnt
'''pip install -r requirements.txt'''

# run training pipline file
'''python training_pipeline.py'''

# execute predict.py
'''python predict.py'''

# manifest.in => here we make a list of files and directory to be used in final distribution

# Virtual environment
* virtualenv ml_package
* ml_package\Scripts\activate
* pip install setuptools twine
# run it at location where setup.py is created
* python setup.py sdist  