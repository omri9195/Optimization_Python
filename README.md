Hey :)

Tested on python version 3.9.13. To avoid compatibility issues please use this python version. With high probability this will work well with other python versions as well.
To check python version run 'python --version'

In order to run the project please follow these steps:
1. Open terminal, clone the project as follows:
    'git clone https://github.com/omri9195/Optimization_Python.git'
2. Go to appropriate directory:
    'cd Optimization_Python'
3. Set up virtual environment as follows:
   Mac (Tested on Mac):
     'python3 -m venv venv'
     'source venv/bin/activate'
  Windows (Not tested on windows):
     'python -m venv venv'
     'venv\Scripts\activate'
4. Install dependencies:
     'pip install -r requirements.txt'
5. Run the project:
     'python -m tests.test_unconstrained_min'


You may observe iteration details in console (terminal), the plots will be created in a plots folder by function name as visible in this directory.

Thanks :)