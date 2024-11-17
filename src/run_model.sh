#!/bin/bash

N='1000'
eta='4.5'
dinc_std='1.0'
alpha='0.5'
Penv='4.0'
mode='0'

while :; do
    case $1 in
        -n)
            if [ "$2" ]; then
                N=$2
                shift
            else
                echo 'ERROR: "-n" requires a non-empty option argument.'
            fi
            ;;
        -e|--eta)
            if [ "$2" ]; then
                eta=$2
                shift
            else
                echo 'ERROR: "--eta" requires a non-empty option argument.'
            fi
            ;;
        -d|--dinc_std)
            if [ "$2" ]; then
                dinc_std=$2
                shift
            else
                echo 'ERROR: "--dinc_std" requires a non-empty option argument.'
            fi
            ;;
        -a|--alpha)
            if [ "$2" ]; then
                alpha=$2
                shift
            else
                echo 'ERROR: "--alpha" requires a non-empty option argument.'
            fi
            ;;
        -p|--Penv)
            if [ "$2" ]; then
                alpha=$2
                shift
            else
                echo 'ERROR: "--Penv" requires a non-empty option argument.'
            fi
            ;;
        -m|--mode)
            if [ "$2" ]; then
                mode=$2
                shift
            else
                echo 'ERROR: "--mode" requires a non-empty option argument.'
            fi
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)
            break
    esac
    shift
done

ipython task.py $N $eta $dinc_std $alpha $Penv $mode