import numpy as n

freqstiming=[
    [2.45, 2.55, 0],
    [3.275, 3.375, 0],
    [3.76, 3.86, 0],
    [4.95, 5.05, 0],
    [7.80, 7.90, 0],
    [9.95, 10.05, 0],
    [14.625, 14.725, 0],
    [14.95, 15.05, 0],
    [19.95, 20.05, 0],
    [3.76, 3.86, 0],
]
    

class sweep():
    def __init__(self,
                 freqs,    # list of frequencies. three values per frequency: min freq, max freq, code idx
                 freq_dur,
                 codes=["waveforms/code-l10000-b10-000000f_100k.bin", # code 0
                        "waveforms/code-l10000-b10-000000f_50k.bin",  # code 1
                        "waveforms/code-l10000-b10-000000f_30k.bin"], # code 2
                 sample_rate=1000000, # In Hz
                 code_amp=0.5):
        
        self.freq_dur=freq_dur
        self.n_freqs=len(freqs)
        self.freqs=freqs

        self.transmit_waveforms=[]
        self.code_len = 0
        self.sample_rate=sample_rate
        # check code lengths
        for c in codes:
            wf=n.fromfile(c,dtype=n.complex64)
            if self.code_len == 0:
                self.code_len=len(wf)
            else:
                if len(wf) != self.code_len:
                    print("Error. Not all waveforms are the same length!")
                    exit(0)
        
        n_reps=int(self.freq_dur*self.sample_rate/self.code_len)

        for c in codes:
            wf=code_amp*n.fromfile(c,dtype=n.complex64)
            self.transmit_waveforms.append(n.tile(wf,n_reps))
        
        self.determine_sweep_length()
        self.t0=n.arange(self.n_freqs,dtype=n.float)*self.freq_dur

    def determine_sweep_length(self):
        self.n_minutes=n.ceil((self.n_freqs*self.freq_dur)/60.0)
        # how many sweeps per day
        self.n_sweeps=n.floor(24*60/self.n_minutes)
        # how long is one ionosonde sweep
        self.sweep_len=24*60/self.n_sweeps
        self.sweep_len_s=self.sweep_len*60
        
    def t0s(self):
        return(self.t0)

    def freq(self,i):
        return(0.5*(self.freqs[i%self.n_freqs][0]+self.freqs[i%self.n_freqs][1])*1e6)
    
    def waveform(self,i):
        code_idx=self.freqs[i%self.n_freqs][2]
        return(self.transmit_waveforms[code_idx])

    def pars(self,i):
        return(self.freq(i),self.t0[i%self.n_freqs])
    
    def bw(self,i):
        return(self.freqs[i][1]-self.freqs[i][0])
    
