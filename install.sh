#!/bin/bash
apk add py3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt