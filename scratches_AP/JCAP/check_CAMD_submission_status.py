from camd_apigw import s3_utils
from tqdm import tqdm
with open('submitted_combination.log') as f:
    submissions  = f.readlines()
submitted_candidates = [data.split(' : ')[1].strip('\n') for data in submissions]
print('Total of {} systems submitted, status are below ....'.format(len(submitted_candidates)))
for chemsys in submitted_candidates:
    print(chemsys, ":",  s3_utils.get_run_status(chemsys))
