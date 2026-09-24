#!/usr/bin/env python3
"""Compatibility entry point: now generates damage-only single-class labels."""
import sys
from scripts.build_sam3_damage_dataset import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "generate":
        arguments = arguments[1:]
    main(arguments)
