################################
##### Runtime instructions #####
################################

# CW example from digital_rf-master/python/examples/sounder
cd waveforms; python create_waveform.py -l 10000 -b 10 -s 0 -f; cd ..
python tx.py -m 192.168.10.3 -d "A:A" -f 3.6e6 -G 1 -g 0 -r 1e6 waveforms/code-l10000-b10-000000f.bin
thor.py -m 192.168.10.13 -d "A:A" -c hfrx -f 3.6e6 -r 1e6 -i 10 /data/prc
python prc_analyze.py /data/prc -c hfrx -l 10000 -s 0


# Stepped example using ./sounder
cd waveforms; python create_waveform.py -l 1000 -b 10 -s 0 -f; cd ..
python tx_chirp.py -m 192.168.10.3 -d "A:A" -f freq_lists/mcm_ops.flist -G 1 -g 0 -r 5E5 waveforms/code-l1000-b10-000000f.bin
python odin.py -m 192.168.10.13 -d "A:A" -c hfrx -f freq_lists/mcm_ops.flist -r 5E5 -i 10 /data/chirp
python analyze_prc.py /data/chirp -c hfrx -l 1000 -s 0 
python plot_rtd.py /data/chirp_notx/ -c hfrx



################################
###### Other Information  ######
################################

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
                 d) there is a DSP-related delay in the receiver. 
                    If you get a solid line on zero doppler, try changing the delay in analyze_chirp.py or prc_analyze.py

    
# Hardware instructions:

# If no dots from the receiver end (e.g nothing gets recorded):
    (or I think if sample is in the past)
	in /home/alex/gnuradio/gr-uhd/lib/usrp_source_impl.cc, comment out line 115: _tag_now = true
    recompile and install gnuradio

	1. Locate the receive antenna at least 100 metres from any other electronics (esp. air conditioning, transformers etc.)
	2. Test all cables for continuity with one end bridged, or with a cable tester
	3. Locate the GPS receiver somewhere that it can see satellites

# IP setting:
    Using network settings, set ethernet to 192.168.10.whatever and subnet to 255.255.255.0. Don't set the gateway
    Note that uhd_find_devices should report your device if it's working
    uhd_usrp_probe should tell you what's on it. 
    Try uhd_fft in gnuradio/gr_uhd/apps to see what signals are in your area


# USRP asks for firmware upgrade	
     In case of firmware upgrade, you have to power-cycle the USRP after upgrading the firmware.
         May also have to downgrade UHD to get it to upgrade
     To change USRP IP address:
            cd /usr/local/lib/uhd/utils
            ./usrp_burn_mb_eeprom --args="ip-addr=192.168.10.2" --values="ip-addr=192.168.10.11"


# Before running the following code:
    Feeling lucky...
        Get the disk image and restore using clonezilla

    Otherwise ...
        install gnuradio, uhd and all the many dependencies - do NOT upgrade pip at any point
        sudo ldconfig


	sudo apt-get install libhdf5-dev
     or 
        Install HDF5 from source with prefix /usr
            cd hdf5-1.10.1/; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ..
            make; sudo make install

            pip install --no-binary h5py -I h5py

    # UHD
        cd <uhd-repo-path>/host
        mkdir build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX=/opt/uhd ../
        make -j4
        make test
        sudo make install
        sudo ldconfig

# Automated running
# See run_tx. Automated running is achieved by: 
    1. Set the computer's BIOS to turn on after power outage 
            (press f2 during reboot, then go to power management)
 	2. Edit /etc/network/interfaces and add the etc_network_interfaces to it  --- not needed?
	3. Add the crontab line listed in run_tx to crontab -e
	4. tail -f run_tx.log

# Networking with USB ethernet dongle
    For plugable USB ethernet adapter:
    Drivers available from http://www.asix.com.tw/products.php?op=pItemdetail&PItemID=131;71;112
    Also available in the repository - plugable_drivers.tar.gz
    Follow README instructions in there - note I did not need the modprobe usbnet command

# Attaching an external hard drive automatically:
    Launch Disks
    Select external drive, click gears icon
    edit mount options
    Automatic off
    mount at startup, select mountpt

    (may also try sudo blkid, vi /etc/fstab)

# Receiver problems (no dots)
    The following error is fatal and needs to be fixed for the receiver to work
        gr::log :WARN: gr uhd usrp source0 - USRP Source Block caught rx error code: 2
    One possibility is that two conflicting PPS signals are being provided to the USRP. 
    In that case, disconnect the external or internal PPS and the problem should go away. 
    For octoclock operation, the PPS and 50MHz should both be plugged in externally, 
    and in that configuration the internal GPSDO (if present) SMA cables are unplugged. 
    That way the Octoclock provides 50MHz and PPS while the internal GPSDO provides timestamps
    
    gr_remez: too much integration and decimation?
    acks: launch time in the past or similar
    
    For both of these, try cleaning out the save directory
    
# Having git save your password:
    git config credential.helper store
    then push/pull and it will save your details

# SSH 
    sudo apt-get install openssh-server  (make sure it is on the source and target)

    Open a terminal session and run the command “ssh-keygen”
    Accept default path (note the path because it’s needed below) and hit “Enter”
    Leave passphrase blank and hit “Enter”
    Leave passphrase confirmation bland and hit “Enter”
    Run the command “ssh-copy-id –i /path_from_above/id_rsa.pub 5-2-1@sd-ssh.jhuapl.edu. 
    From my Mac the exact command is:
        ssh-copy-id -i /Users/chartat1/.ssh/id_rsa.pub chartat1@sd-ssh.jhuapl.edu
    You will be asked to accept the RSA key if you haven’t connected to SD-SSH before. Enter “yes”
    Enter your DMZ unix password.
    You should get verification that that one key was added. You can then log out.

    scp filename chartat1@sd-ssh.jhuapl.edu:/project/space_weather_imaging/alex/south_pole/


# Mismatching sample_rate_numerator
    You have to clear out the data storage directory - it contains files from a different sample-rate experiment


# rsync

    crontab -e
    Then, in the crontab, to backup every 10 minutes
    */10 * * * * rsync -av -e ssh /data/ch0/prc_analysis/ chartat1@sd-ssh.jhuapl.edu:/project/space_weather_imaging/alex/south_pole/

# how to rename a computer
sudo vi /etc/hostname
sudo vi /etc/hosts

###############################################
######## Computer Deployment Checklist ########
###############################################

1. Computer restarts and goes back to operating after a power outage
2. Can access GitHub
3. Local disk is not filling up
4. uhd_find_devices 
5. External drives mounted correctly
6. Loopback test (look for signal on gnuradio/gr-uhd/apps/uhd_fft -f 5E6)
7. DSP delay calibration (receive only)
