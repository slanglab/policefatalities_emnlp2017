import subprocess
from collections import defaultdict

def get_first_last(names, inputfile):
    '''
    returns dictionary where keys are the input names and values
    are each corresponding hapnis name
    '''
    hap = {}
    filename = inputfile+'.tmp'
    w = open(filename, 'w')
    for name in names:
        w.write(name.encode('utf-8'))
        w.write('\n')
    w.close()
    
    cmd = ('perl hap/hapnis.pl -names < {0}').format(filename)
    body = subprocess.check_output(cmd, shell=True).splitlines()
    
    assert len(body) == len(names)
      
    for i, line in enumerate(body):
        line = line.decode('utf-8') 
        if "Forename" not in line or "Surname" not in line: hap[names[i]] = names[i]; continue 
        #print line 
        name = line.strip().split()
        firstlast = u""
        for part in name:
            w = part.split("_")
            if w[1]=="Forename": firstlast = w[0]
            elif w[1]=="Surname": firstlast = firstlast + " " + w[0]
        if " " not in firstlast.strip(): hap[names[i]]= names[i]; continue 
        else: hap[names[i]] = firstlast.lstrip()
    
#     eek this is not true for non-unique names! 
#    assert len(hap) == len(names)
    #assert None not in hap.values() #we want no errors in the fatal encounters data! 
    return hap 

def get_hap_to_org(names, inputfile):
    '''
    returns dictionary where keys are the input names and values
    are each corresponding hapnis name
    '''
    hap = defaultdict(list)
    filename = inputfile+'.tmp'
    w = open(filename, 'w')
    for name in names:
        name = name.replace('\n', '')
        w.write(name.encode('utf-8'))
        w.write('\n')
    w.close()
    
    cmd = ('perl hap/hapnis.pl -names < {0}').format(filename)
    body = subprocess.check_output(cmd, shell=True).splitlines()
    
    assert len(body) == len(names)
      
    for i, line in enumerate(body):
        line = line.decode('utf-8') 
        if "Forename" not in line or "Surname" not in line: hap[names[i]].append(names[i]); continue 
        #print line 
        name = line.strip().split()
        firstlast = u""
        for part in name:
            w = part.split("_")
            if w[1]=="Forename": firstlast = w[0]
            elif w[1]=="Surname": firstlast = firstlast + " " + w[0]
        if " " not in firstlast.strip(): hap[names[i]].append(names[i]); continue 
        else: hap[firstlast.lstrip()].append(names[i])
    
    return hap  
