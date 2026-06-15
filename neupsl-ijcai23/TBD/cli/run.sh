#!/bin/bash

# Configuration
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="${THIS_DIR}/TBD.json"
JAR_PATH="${THIS_DIR}/psl-cli-2.4.0.jar"

echo "Running NeuPSL with config: $CONFIG_PATH"
java -jar "$JAR_PATH" --learn --config "$CONFIG_PATH" --output inferred-predicates "$@"
