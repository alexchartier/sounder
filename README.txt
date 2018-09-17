# example prc transmit, record, and processing

# see the following paper for a description and application of the technique:
# Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
# Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
# doi:10.5194/amt-9-829-2016, 2016.


# In standard mode, 
    tx.py transmits a coded continuous wave made up of "l" bauds (usually 10x oversampled) at a specified sample rate. 
    thor.py records on a specified frequency at a specified sample rate. 
    prc_analyze.py analyzes the recordings for a specified number of ranges by convolving the coded wave with the received signal:

        1. Create estimation matrix B based on applying coded wave to signal in each range-gate 
        2. dot B and the signal (z) to get your output at each range. 
        3. Do an fft of each range-gate to get power in range-frequency coordinates

    other stuff: a) the code removes a DC offset from the signal, 
                 b) A blackman-harris window is applied to filter the signal
                 c) the code optionally removes RFI by "whitening" the signal
    

# Questions for Juha:
    1. Can the transmitter lose lock and keep on transmitting?
    2. Doppler frequency resolution?
    3. Can we do different baud oversampling on Tx and Rx side?


# Before running the following code:
     install gnuradio, uhd and all the many dependencies - do NOT upgrade pip at any point
     sudo ldconfig
     Using network manager, set the relevant ethernet port's IP to 192.168.10.X where X is NOT 2 (or the USRP's number)
        (note ifconfig provides only a temporary fix, but does let you check the IP has been set correctly)
     Plug in the USRP and run uhd_find_devices to make sure it's visible
     When running the code, don't worry about the occasional "failed to lock" from the GPS if the antenna is poorly located

     HDF5-specific:
        Install HDF5 from source with prefix /usr
            cd hdf5-1.10.1/; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ..
            make; sudo make install

            pip install --no-binary h5py -I h5py


# create a waveform
python create_waveform.py -l 10000 -b 10 -s 0

# tx
python tx_chirp.py -m 192.168.10.2 -d "A:A" -f 3.6e6 -G 0.25 -g 0 -r 1e6 code-l10000-b10-000000.bin

# rx
odin.py -m 192.168.10.3 -d "A:A" -c hfrx -f 3.6e6 -r 1e6 -i 10 ~/data/prc
    # NOTE: receiver sometimes drops a sample on retuning

# analysis
python analyze_chirp.py ~/data/prc -c hfrx -l 10000 -s 0 -n freqstep.log

-c channel, -l code length, -r samplerate, -s starttime, 

# Automated running
# See run_tx. Automated running is achieved by: 
    1. Set the computer's BIOS to turn on after power outage 
            (press f2 during reboot, then go to power management)
 	2. Edit /etc/network/interfaces and add the etc_network_interfaces to it
	3. Add the crontab line listed in run_tx to crontab -e
	4. tail -f run_tx.log

# Networking with USB ethernet dongle
    For plugable USB ethernet adapter:
    Drivers available from http://www.asix.com.tw/products.php?op=pItemdetail&PItemID=131;71;112
    Also available in the repository - plugable_drivers.tar.gz
    Follow README instructions in there - note I did not need the modprobe usbnet command


# Attaching an external hard drive automatically:
    Add the following to /etc/fstab (with correct UUID from blkid /dev/sdb)
    # dev/sdb1
    UUID=62dd164d-0150-4d07-a0c4-31417a1ab6d9 /data           ext4 nofail,auto,noatime,rw,user 0 0
