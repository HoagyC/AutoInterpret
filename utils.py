import json
import os
from pathlib import Path
from typing import List, Tuple

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import numpy as np
from scipy import stats

BUCKET_NAME = "autointerpret"

def pearsonr_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Tuple[float, float, float, float]:
    ''' 
    Taken from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

    Calculate Pearson correlation along with the confidence interval using scipy and numpy

    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.1 by default d
    
    Returns
    -------
    r : float  =  Pearson's correlation coefficient
    pval : float  =  The corresponding p value
    lo, hi : float  =  The lower and upper bound of confidence intervals
    '''
    # TODO: More principled response to this?
    if max(x) == min(x) or max(y) == min(y):
        return -0.5, -0.5, -0.5, 1.0

    r, p = stats.pearsonr(x,y) # blank value is p-value
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return float(r), float(lo), float(hi), float(p)

def upload_to_aws(local_file_name, s3_file_name: str = "") -> bool:
    """"
    Upload a file to an S3 bucket
    :param local_file_name: File to upload
    :param s3_file_name: S3 object name. If not specified then local_file_name is used
    """
    secrets = json.load(open("secrets.json"))

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )

    if not s3_file_name:
        s3_file_name = local_file_name
    local_file_path = Path(local_file_name)
    try:
        if local_file_path.is_dir():
            _upload_directory(local_file_name, s3)
        else:
            s3.upload_file(str(local_file_name), BUCKET_NAME, str(s3_file_name))
        print(f"Upload Successful of {local_file_name}")
        return True
    except FileNotFoundError:
        print(f"File {local_file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
  
def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))


def download_from_aws(files: List[str], force_redownload: bool = False) -> bool:
    """
    Download a file from an S3 bucket
    :param files: List of files to download
    :param force_redownload: If True, will download even if the file already exists
    
    Returns:
        True if all files were downloaded successfully, False otherwise
    """
    secrets = json.load(open("secrets.json"))
    
    if not force_redownload:
        files = [f for f in files if not os.path.exists(f)]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    all_correct = True
    for filename in files:
        try:
            parent_dir = os.path.dirname(filename)
            if not os.path.exists(parent_dir) and parent_dir != "":
                os.makedirs(os.path.dirname(filename))
            with open(filename, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, filename, f)

            print(f"Successfully downloaded file: {filename}")
        except ClientError:
            print(f"File: {filename} does not exist")
            all_correct = False

    return all_correct

