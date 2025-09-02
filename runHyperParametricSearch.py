import os
nHeads = [1,2,3,4]
nLayers = [1,2,3]
d_ff = [768]
d_k = [32] 
dropout = [0.05, 0.1]
'''
second run
nHeads = [i for i in range(1,20, 2)]
nLayers = [i for i in range(2, 8, 2)]
nHeads = [1] + nHeads
nLayers = [1] + nLayers
d_ff = [ 512, 768, 1024]
d_k = [32, 64] 
dropout = [0]
'''
'''
#initial values: resulted in NHeads 18, nLayers 1 (more layers didn't run as it reached the HPC limit), d_kk = 64, d_ff = 768
#first run
nHeads = [i for i in range(18,26, 2)]
nLayers = [i for i in range(2, 20, 2)]
nLayers = [1] + nLayers
d_ff = [512, 768, 1024]
d_k = [64] 
dropout = [0]'''

for head in nHeads:
    for layer in nLayers:
        for feedForw in d_ff:
            for dimInHeads in d_k:
                for d in dropout:
                    command = (
                            f"qsub -q v1_limited "
                            f"-v nHeads={head},nLayers={layer},d_k={dimInHeads},d_ff={feedForw},dropout={d} "
                            f"run5Fold.pbs"
                        )

                    
                    print("Submitting:", command)
                    os.system(command)