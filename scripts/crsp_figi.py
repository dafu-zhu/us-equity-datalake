from quantdl.master.security_master import SecurityMaster
import wrds
import boto3
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("Connecting to WRDS...")
db = wrds.Connection(
    wrds_username=os.getenv('WRDS_USERNAME'),
    wrds_password=os.getenv('WRDS_PASSWORD')
)

print("Initializing SecurityMaster...")
s3 = boto3.client('s3')
sm = SecurityMaster(db=db)

print("Running overwrite_from_crsp...")
stats = sm.overwrite_from_crsp(db, s3, 'us-equity-datalake')
print(f"Done! Stats: {stats}")