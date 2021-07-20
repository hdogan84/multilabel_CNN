#!/bin/bash

TEST_DIR=./src/torchserve_handler/unit_tests
case $PWD/ in
*/src/torchserve_handler/unit_tests/) echo "Running tests" ;;
*)
    echo "Error! Must start in unit_tests directory"
    exit 1
    ;;
esac

cd ../../../

audio_handler() {
    python -m pytest $TEST_DIR/test_audio_handler.py
}

audio_handler
