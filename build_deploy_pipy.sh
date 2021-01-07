#!/bin/bash
rm -r dist ;
python3 setup.py sdist bdist_wheel ;

if python3 -m twine check dist/* ; then
  if [ "$1" = "--test" ] ; then
    #twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    python3 -m twine upload --repository testpypi dist/*
  else
    #twine upload dist/* ;
    python3 -m twine upload dist/*
  fi
fi
