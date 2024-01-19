#!/bin/bash
if [[ -d storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents storage/external
fi

cd storage/external

if [[ ! -f "MSR_data_cleaned.csv" ]]; then
    gdown https://drive.google.com/u/0/uc?id=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X&export=download
    unzip MSR_data_cleaned.zip
    rm MSR_data_cleaned.zip
else
    echo "Already downloaded bigvul data"
fi

if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v1.1.919/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi
