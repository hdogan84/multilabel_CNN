#!/bin/sh

# This sets up a port forward from localhost to staging's mongodb instance.
# You will want to download robomongo from https://robomongo.org/download
# fire it up and connect to localhost 3307.

ssh -L 6007:127.0.0.1:6007 stana@stana
0
