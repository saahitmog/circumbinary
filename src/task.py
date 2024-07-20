async def main(N: int=100_000, eta: float=4.5, sigma_di: float=1,   
               alpha: float=0, mode: int=0):
    
    N = int(N)
    eta = float(eta)
    sigma_di = float(sigma_di)
    alpha = float(alpha)
    mode = int(mode)

    n = os.getenv('SLURM_CPUS_ON_NODE')
    if n: n = int(n)
    args = [eta, sigma_di, alpha, mode]
    map_args = np.array(args*N).reshape(N, 4)

    rc = await ipp.Cluster().start_and_connect(n=n)
    view = rc.load_balanced_view()
    ar = view.map_async(wrapper, map_args, ordered=False)
    ar.wait_interactive()
    results = ar.get()
    rc.shutdown()

    data = np.array([r for r in results if r is not None])
    try:
        df = pd.DataFrame(data, columns=['Initial_Binary_Period',
                                         'Binary_Period',
                                         'Final_Binary_Period',
                                         'Binary_Semimajor_Axis',
                                         'Planet_Period',
                                         'Final_Planet_Period',
                                         'Planet_Semimajor_Axis',
                                         'Mass_A', 'Mass_B', 
                                         'Radius_A', 'Radius_B',
                                         'Initial_Binary_Eccentricity',
                                         'Binary_Eccentricity',
                                         'Final_Binary_Eccentricity',
                                         'Final_Planet_Eccentricity',
                                         'Binary_Inclination',
                                         'Planet_Inclination',
                                         'Planet_Sky_Inclination',
                                         'Binary_Omega',
                                         'Planet_Omega',
                                         'Planet_Sky_Omega',
                                         'Binary_omega',
                                         'Planet_omega',
                                         'Planet_Sky_omega',
                                         'Number_of_Orbits', 
                                         'Transit_Value'])
    except ValueError:
        df = None
    
    metadata = {'eta': eta, 'alpha': alpha, 'sigma_di': sigma_di, 'N': N, 
                'mode': mode, 'wall time': ar.wall_time, 
                'speed up': ar.serial_time/ar.wall_time}

    dir_name = 'data'
    if not os.path.exists(dir_name): 
        os.makedirs(dir_name)
    e, d, a, m, i = (str(eta).replace('.', ''), str(sigma_di).replace('.', ''), 
                     str(alpha).replace('.', ''), str(mode), 0)
    file = f'{dir_name}/e{e}_d{d}_a{a}_m{m}_{i}.h5'
    name_len = len(f'{dir_name}/e{e}_d{d}_a{a}_m{m}_')
    while os.path.exists(file):
        i += 1
        file = file[:name_len]+f'{i}.h5'
        
    transit = 0
    if df is not None:
        with pd.HDFStore(file, 'w') as hdf:
            hdf.put(dir_name, df, index=False)
            hdf.get_storer(dir_name).attrs.metadata = metadata

        transit = len(df[df.Transit_Value == 2])
        
    print(f'\n{(transit*1e2/N):.2f}% Transit Success Rate')
    print(f'{(ar.serial_time / ar.wall_time):.2f}x speedup')
    print(file+'\n')

if __name__ == '__main__': 

    import os
    import sys
    import asyncio
    import numpy as np
    import pandas as pd
    import ipyparallel as ipp
    from src.functions import wrapper

    inp = sys.argv[1:]

    if inp[0] != 'savio': asyncio.run(main(*inp))
    else:
        inps = [[2,   1,   0.5,    0],
                [4.5, 1,   0.5,    0],
                [7,   1,   0.5,    0],
                [4.5, 0.3, 0.5,    0],
                [4.5, 3,   0.5,    0],
                [4.5, 1,   -0.5,   0],
                [4.5, 1,   0,      0],
                [4.5, 1,   1,      0],
                [4.5, 1,   0.5,    1],
                [4.5, 1,   0.5,    2]]
        N = 100_000

        for i in inps: asyncio.run(main(N, *i))