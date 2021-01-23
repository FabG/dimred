# DimRed Packaging Instructions
The below instructions are for maintaining, updating and uploading dimred as a python package to PiPy.
You can go back to the main [README](readme.md) for using dimred package

#### 1. Automatic packaging and uploading to PiPy
Increment the version of the package in [setup.py](setup.py) following the `M.m.p` convention (M=Major, m=minor, p=patch).

Example:
`VERSION = '0.0.1'`

A Shell scripts was built to automate the build+upload to Pipy: `build_deploy_pipy.sh`
To package `dimred` and test it first at `test.pipy.org/dimred`, run:
```bash
./build_deploy_pipy.sh --test
```
You will be promoted for your `PiPy` username and password.

To package it and make it available to the pipy cominuty at `pipy.org/dimred`
```bash
./build_deploy_pipy.sh
```

Once this completes successfully, you can pip install your package as follow:
```bash
pip install -i https://test.pypi.org/simple/ dimred
```
or
```bash
pip install -i https://pypi.org/simple/ dimred
```

And you can import it to test in python:
```python
python
>>> import dimred
```

#### 2. Manual Packaging and uploading to PiPy
##### 2.1 Packaging
To generate distribution packages, make sure you have the latest versions of `setuptools` and `wheels` installed:
```bash
python3 -m pip install --user --upgrade setuptools wheel
```

Now run this command from the same directory where setup.py is located:
```bash
python3 setup.py sdist bdist_wheel
```

This command will create several directories: `dimred.egg-info`, `dist` and `build`.
It should output a lot of text and once completed should generate two files in the `dist` directory which is what we are interested into:
```
dist/
  dimred-0.1.0-py3-none-any.whl
  dimred-0.1.0.tar.gz
```

The `tar.gz` file is a Source Archive whereas the `.whl` file is a Built Distribution. Newer pip versions preferentially install built distributions, but will fall back to source archives if needed. You should always upload a source archive and provide built archives for the platforms your project is compatible with. In this case, our example package is compatible with Python on any platform so only one built distribution is needed.

Pro tip: Add these directories to your `.gitignore` file, to prevent pushing installation files to your repo.

##### 2.2 Building & Deploying
If not done already, go to [pypi.org](https://pypi.org/) and create an account. You’ll need it to upload your new library.
You can also do so at the test website [test.pypi.org](https://test.pypi.org/)

Install `twine`, which will allow us to deploy to PyPI. It is as simple as:
```bash
pip install twine
```

Next, we’ll create an installable package (that was done earlier so skip if you have done it already). Go to the directory where the `setup.py` file is, and run:
```bash
python setup.py sdist bdist_wheel
```

Next up, verify the distribution files you just created are okay by running:
```bash
python -m twine check dist/*
```

You should see something like:
```bash
Checking dist/dimred-0.1.0-py3-none-any.whl: PASSED
Checking dist/dimred-0.1.0.tar.gz: PASSED
```


Time to upload your package to PyPI. I recommend deploying first to the PyPI test domain, so you can verify everything looks as you intended.

Do this using:
```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

You can now view your package at:
https://test.pypi.org/project/<package>/<version>
For example for dimred v0.1.0:
View at:
https://test.pypi.org/project/dimred/0.1.0/

And you can pip install it with:
```bash
pip install -i https://test.pypi.org/simple/ dimred
```
