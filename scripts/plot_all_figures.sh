cd /home/lperon/cftp_dis_spin/scripts

# physics
echo "Plotting physics figures..."
python -m CW.CW_physics
echo "CW physics figures plotted."
python -m SK.SK_physics
echo "SK physics figures plotted."
python -m ER.ER_physics
echo "ER physics figures plotted."
python -m RR.RR_physics
echo "RR physics figures plotted."
python -m latt.lattice_physics
echo "Lattice physics figures plotted."

# coal_time
echo "Plotting coalescence time figures..."
python -m CW.CW_coal_time
echo "CW coalescence time figures plotted."
python -m SK.SK_coal_time
echo "SK coalescence time figures plotted."
python -m ER.ER_coal_time
echo "ER coalescence time figures plotted."
python -m RR.RR_coal_time
echo "RR coalescence time figures plotted."
python -m latt.lattice_coal_time
echo "Lattice coalescence time figures plotted."

# time_merging_config
echo "Plotting time merging config figures..."
python -m CW.CW_time_merging_config
echo "CW time merging config figures plotted."
python -m SK.SK_time_merging_config
echo "SK time merging config figures plotted."
python -m ER.ER_time_merging_config
echo "ER time merging config figures plotted."
python -m RR.RR_time_merging_config
echo "RR time merging config figures plotted."
python -m latt.latt_time_merging_config
echo "Lattice time merging config figures plotted."

# numb_star
echo "Plotting numb star figures..."
python -m CW.CW_numb_star
echo "CW numb star figures plotted."
python -m SK.SK_numb_star
echo "SK numb star figures plotted."
python -m ER.ER_numb_star
echo "ER numb star figures plotted."
python -m RR.RR_numb_star
echo "RR numb star figures plotted."
python -m latt.lattice_numb_star
echo "Lattice numb star figures plotted."