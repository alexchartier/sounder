#!/bin/bash
# Add the following to crontab -e to run this every minute:
#    */1 * * * * /home/alex/sounder/run_rx > /home/alex/sounder/run_rx.log &

echo "Started run_rx - tail -f /var/log/syslog for more information"
current_time="`date +%Y/%m/%d_%H:%M`";
chan=cha
outdir=/home/alex/data/hf12
backup_dir=/mnt/data/hf12/

# Rsync out everything over 5 mins old
rsync -avz --remove-sent-files --files-from=<(find $outdir/$chan/2021* -cmin +10) / $backup_dir



# Check output directory size as a function of time
sz1=`df $outdir/$chan |  tail -n1 | awk '{print $3}'`
sleep 1
sz2=`df $outdir/$chan |  tail -n1 | awk '{print $3}'`

# Start the receiver code if the directory is not getting bigger
if [ $sz1 == $sz2 ]; then
    echo "Directory not getting bigger. Launching THOR at ${current_time}"
    /usr/local/bin/thor.py -m 192.168.10.2 -d "A:A" -c $chan -f 7E6 -r 12.5E6 --type sc16 $outdir
else :
    pid="`pgrep -x thor.py`"
    echo "thor.py running at ${current_time} as pid ${pid}"
fi
exit 0
