import argparse
import json
import os
from datetime import datetime

import papermill as pm

if __name__ == '__main__':
    # python evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/aman/MicroSplit/NotebookRuns/')
    parser.add_argument('--sliding_window_flag',  action='store_true', help='Which inference to use')
    parser.add_argument('--tag_time_flag', action='store_false', help='Tag time flag')
    parser.add_argument('--override_kwargs', type=json.loads, default='{}', )
    args = parser.parse_args()
    param_dict = args.override_kwargs
    keys = sorted(param_dict.keys())
    param_str = '_'.join([f'{k}-{param_dict[k]}' for k in keys])
    # ckpt_dir = args.ckpt_dir
    # model_token = '-'.join(ckpt_dir.strip('/').split('/')[-3:])
    model_token = ''
    outputdir = os.path.join(args.outputdir, model_token)
    fname = os.path.basename(args.notebook)
    fname = fname.replace('.ipynb','')
    
    # subdirectory for the current month
    yearmonth = datetime.now().strftime("%Y%m")
    outputdir = os.path.join(outputdir, yearmonth)
    os.makedirs(outputdir, exist_ok=True)

    if args.tag_time_flag:
        now = datetime.now().strftime("%Y%m%d.%H.%M")
    else:
        now = datetime.now().strftime('%y%m%d')

    fname = f"{now}_{fname}_{param_str}.ipynb"

    output_fpath = os.path.join(outputdir, fname)
    output_config_fpath = os.path.join(outputdir,'config', fname.replace('.ipynb','.txt'))
    os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)
    # save as json
    with open(output_config_fpath, 'w') as f:
        f.write(str(args_dict))

    print(output_fpath, '\n', output_config_fpath)
    # param_dict['ckpt_dir'] = args.ckpt_dir
    pm.execute_notebook(
        args.notebook,
        output_fpath,
        # parameters = param_dict
    )
    