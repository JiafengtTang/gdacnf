#!/bin/bash

python mainBaseline.py mnist sourceonly 0
python mainBaseline.py portraits sourceonly 0
python mainBaseline.py tox21a sourceonly 0
python mainBaseline.py tox21b sourceonly 0
python mainBaseline.py tox21c sourceonly 0
python mainBaseline.py shift15m sourceonly 0
python mainBaseline.py rxrx1 sourceonly 0

python mainBaseline.py mnist gst 0
python mainBaseline.py portraits gst 0
python mainBaseline.py tox21a gst 0
python mainBaseline.py tox21b gst 0
python mainBaseline.py tox21c gst 0
python mainBaseline.py shift15m gst 0
python mainBaseline.py rxrx1 gst 0

python mainBaseline.py mnist gift-low 0
python mainBaseline.py portraits gift-low 0
python mainBaseline.py tox21a gift-low 0
python mainBaseline.py tox21b gift-low 0
python mainBaseline.py tox21c gift-low 0
python mainBaseline.py shift15m gift-low 0
python mainBaseline.py rxrx1 gift-low 0

python mainBaseline.py mnist gift-mid 0
python mainBaseline.py portraits gift-mid 0
python mainBaseline.py tox21a gift-mid 0
python mainBaseline.py tox21b gift-mid 0
python mainBaseline.py tox21c gift-mid 0
python mainBaseline.py shift15m gift-mid 0
python mainBaseline.py rxrx1 gift-mid 0

python mainBaseline.py mnist gift-high 0
python mainBaseline.py portraits gift-high 0
python mainBaseline.py tox21a gift-high 0
python mainBaseline.py tox21b gift-high 0
python mainBaseline.py tox21c gift-high 0
python mainBaseline.py shift15m gift-high 0
python mainBaseline.py rxrx1 gift-high 0

python mainBaseline.py mnist aux-low 0
python mainBaseline.py portraits aux-low 0
python mainBaseline.py tox21a aux-low 0
python mainBaseline.py tox21b aux-low 0
python mainBaseline.py tox21c aux-low 0
python mainBaseline.py shift15m aux-low 0
python mainBaseline.py rxrx1 aux-low 0

python mainBaseline.py mnist aux-mid 0
python mainBaseline.py portraits aux-mid 0
python mainBaseline.py tox21a aux-mid 0
python mainBaseline.py tox21b aux-mid 0
python mainBaseline.py tox21c aux-mid 0
python mainBaseline.py shift15m aux-mid 0
python mainBaseline.py rxrx1 aux-mid 0

python mainBaseline.py mnist aux-high 0
python mainBaseline.py portraits aux-high 0
python mainBaseline.py tox21a aux-high 0
python mainBaseline.py tox21b aux-high 0
python mainBaseline.py tox21c aux-high 0
python mainBaseline.py shift15m aux-high 0
python mainBaseline.py rxrx1 aux-high 0