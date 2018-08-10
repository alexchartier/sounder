# example prc transmit, record, and processing

# see the following paper for a description and application of the technique:
# Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
# Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
# doi:10.5194/amt-9-829-2016, 2016.

# create a waveform
python create_waveform.py -l 10000 -b 10 -s 0

# tx
python tx_chirp.py -m 192.168.10.2 -d "A:A" -f 3.6e6 -G 0.25 -g 0 -r 1e6 code-l10000-b10-000000.bin

# rx
odin.py -m 192.168.30.2 -d "A:A" -c hfrx -f 3.6e6 -r 1e6 -i 10 ~/data/prc

# analysis
python prc_analyze.py /data/prc -c hfrx -l 10000 -s 0


# Chirpsounder Operating instructions
 	1. Edit /etc/network/interfaces and add the etc_network_interfaces to it
	2. Add the crontab line listed in run_tx to crontab -e
	3. tail -f run_tx.log
