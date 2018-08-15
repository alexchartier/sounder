# example prc transmit, record, and processing

# see the following paper for a description and application of the technique:
# Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
# Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
# doi:10.5194/amt-9-829-2016, 2016.

# Before running the following code:
     install gnuradio, uhd and all the many dependencies - do NOT upgrade pip at any point
     sudo ldconfig
     Using network manager, set the relevant ethernet port's IP to 192.168.10.X where X is NOT 2 (or the USRP's number)
        (note ifconfig provides only a temporary fix, but does let you check the IP has been set correctly)
     Plug in the USRP and run uhd_find_devices to make sure it's visible
     When running the code, don't worry about the occasional "failed to lock" from the GPS if the antenna is poorly located

# create a waveform
python create_waveform.py -l 10000 -b 10 -s 0

# tx
python tx_chirp.py -m 192.168.10.2 -d "A:A" -f 3.6e6 -G 0.25 -g 0 -r 1e6 code-l10000-b10-000000.bin

# rx
odin.py -m 192.168.10.3 -d "A:A" -c hfrx -f 3.6e6 -r 1e6 -i 10 ~/data/prc

# analysis
python prc_analyze.py /data/prc -c hfrx -l 10000 -s 0


# Automated running
# See run_tx. Automated running is achieved by: 
        1. Set the computer's BIOS to turn on after power outage 
            (press f2 during reboot, then go to power management)
        2. Edit the crontab (crontab -e) to specify automated running of the transmit/receive code
        3. tail -f the log files to see what's happening

# Networking with USB ethernet dongle
    For plugable USB ethernet adapter:
    Drivers available from http://www.asix.com.tw/products.php?op=pItemdetail&PItemID=131;71;112
    Also available in the repository - plugable_drivers.tar.gz
    Follow README instructions in there - note I did not need the modprobe usbnet command
