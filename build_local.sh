#!/bin/bash
rm -r dist ;

python3 setup.py sdist
pip3 uninstall dimred
pip3 install dist/dimred-*.gz
