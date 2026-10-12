#!/usr/bin/env bash
# UCI Adult (Census Income), mirrored as headerless CSV. Source data: UCI ML Repository, CC BY 4.0.
curl -L -o "$(dirname "$0")/adult.csv" https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
