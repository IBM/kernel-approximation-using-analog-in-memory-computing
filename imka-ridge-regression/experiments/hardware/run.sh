#!/bin/bash

for TASK in ijcnn magic eeg cod-rna letter skin
do
    echo " >> Running" $TASK
    python experiments/hardware/hardware_general.py --config experiments/hardware/config/config_$TASK.yml
done

python experiments/hardware/hardware_attn.py