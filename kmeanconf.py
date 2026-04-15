def _hook_db1(params,order):
    sample=[]

    for i in range(len(order)):
        if order[i][0]!=1: continue
        if order[i][1]!='pdf': continue
        if 'g1'  in order[i][2]: continue 
        if 'uv1' in order[i][2]: continue 
        if 'dv1' in order[i][2]: continue 
        #if 'db1' in order[i][2]: continue 
        if 'ub1' in order[i][2]: continue 
        if 's1' in order[i][2]: continue 
        if 'sb1' in order[i][2]: continue
        if 'sea1' in order[i][2]: continue
        if 'sea1' in order[i][2]: continue

        #if order[i][2].split()[1]=='N': continue
        #if order[i][2].split()[1]=='a': continue
        #if order[i][2].split()[1]=='b': continue
        #if order[i][2].split()[1]=='c': continue
        #if order[i][2].split()[1]=='d': continue
        #print (order[i][2], params[i])
        sample.append(params[i])
    return sample

def _hook_off(params,order):
    sample=[]

    for i in range(len(order)):
        if order[i][0]!=1: continue
        if order[i][1]!='off': continue
        sample.append(params[i])
    return sample

def _hook_strange(params,order):
    sample=[]

    for i in range(len(order)):
        if order[i][0]!=1: continue
        if order[i][1]!='pdf': continue
        if 'g1'  in order[i][2]: continue 
        if 'uv1' in order[i][2]: continue 
        if 'dv1' in order[i][2]: continue 
        if 'db1' in order[i][2]: continue 
        if 'ub1' in order[i][2]: continue 
        if 'sp1' in order[i][2]: continue 
        #if 'sm1' in order[i][2]: continue
        if 'sea1' in order[i][2]: continue
        if 'sea2' in order[i][2]: continue

        #if order[i][2].split()[1]=='N': continue
        #if order[i][2].split()[1]=='a': continue
        #if order[i][2].split()[1]=='b': continue
        #if order[i][2].split()[1]=='c': continue
        #if order[i][2].split()[1]=='d': continue
        #print (order[i][2], params[i])
        #print params[i]
        sample.append(params[i])
    return sample

def _hook_ub(params,order):
    sample=[]

    for i in range(len(order)):
        if order[i][0]!=1: continue
        if order[i][1]!='ppdf': continue
        if 'g1'  in order[i][2]: continue 
        if 'uv1' in order[i][2]: continue 
        if 'dv1' in order[i][2]: continue 
        if 'db1' in order[i][2]: continue 
        #if 'ub1' in order[i][2]: continue 
        if 's1' in order[i][2]: continue 
        if 'sb1' in order[i][2]: continue
        if 'sea1' in order[i][2]: continue
        if 'sea2' in order[i][2]: continue

        #if order[i][2].split()[1]=='N': continue
        #if order[i][2].split()[1]=='a': continue
        #if order[i][2].split()[1]=='b': continue
        #if order[i][2].split()[1]=='c': continue
        #if order[i][2].split()[1]=='d': continue
        #print (order[i][2], params[i])
        #print params[i]
        sample.append(params[i])
    return sample

def _hook_g(params,order):
    sample=[]

    for i in range(len(order)):
        if order[i][0]!=1: continue
        if order[i][1]!='ppdf': continue
        #if 'g1'  in order[i][2]: continue 
        if 'uv1' in order[i][2]: continue 
        if 'dv1' in order[i][2]: continue 
        if 'db1' in order[i][2]: continue 
        if 'ub1' in order[i][2]: continue 
        if 's1' in order[i][2]: continue 
        if 'sb1' in order[i][2]: continue
        if 'sea1' in order[i][2]: continue
        if 'sea2' in order[i][2]: continue

        #if order[i][2].split()[1]=='N': continue
        #if order[i][2].split()[1]=='a': continue
        #if order[i][2].split()[1]=='b': continue
        #if order[i][2].split()[1]=='c': continue
        #if order[i][2].split()[1]=='d': continue
        #print (order[i][2], params[i])
        #print params[i]
        sample.append(params[i])
    return sample


nc={}
hooks={}

for i in range(100):
  nc[i+1]=1
  hooks[i+1]=None























