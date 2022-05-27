from os import path, getenv
from argparse import ArgumentParser, BooleanOptionalAction
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import boto3
from fabric import Connection, ThreadingGroup
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(".env_test"))

region = getenv("WORKING_REGION")
availability_zone = getenv("WORKING_AZ")
subnet_id = getenv("WORKING_SUBNET")
security_group_ids = [getenv("WORKING_SECURITY_GROUP")]
key_name = getenv("WORKING_KEY_NAME")
results_bucket = getenv("WORKING_BUCKET")

min_worker_count = 1
max_worker_count = 1
worker_instance_type = "r5a.xlarge"
worker_tag = "worker"
worker_tmux_session = "worker"
worker_instance_ids = None
worker_dns_names = None

leader_instance_type = "r5a.large"
leader_tag = "leader"
leader_tmux_session = "leader"
leader_runner_tmux_session = "runner"
leader_instance_role = {"Arn": getenv("WORKING_EC2_INSTANCE_ROLE_ARN")}

base_ami = getenv("WORKING_BASE_AMI")
dry_run = False
ssh_key_location = getenv("WORKING_SSH_KEY_LOCATION")
instance_username = getenv("WORKING_INSTANCE_USERNAME")

leader_instance = None
leader_instance_dns_name = None
leader_instance_id = None
leader_instance_private_ip = None
can_terminate_leader = False

def create_workers(ec2_resource):
    global worker_instance_ids
    worker_instances = ec2_resource.create_instances(
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "VolumeSize": 15,
                    "VolumeType": "gp3",
                    "Throughput": 125
                }
            }
        ],
        ImageId = base_ami,
        InstanceType = worker_instance_type,
        MaxCount = max_worker_count,
        MinCount = min_worker_count,
        KeyName=key_name,
        SecurityGroupIds = security_group_ids,
        SubnetId = subnet_id,
        DryRun = dry_run,
        InstanceInitiatedShutdownBehavior="stop",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "cluster_id",
                        "Value": args.id
                    },
                    {
                        "Key": "cluster_role",
                        "Value": worker_tag
                    }
                ]
            }
        ]
    )
    worker_instance_ids = list([worker.id for worker in worker_instances])
    logger.info(f"{len(worker_instance_ids)} workers created")


def start_workers(ec2_client):
    global worker_dns_names
    logger.info(f"waiting for worker instances running")
    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=worker_instance_ids)
    logger.info(f"workers running, starting worker clients")
    worker_dns_names = list([instance["PublicDnsName"] for instance in ec2_client.describe_instances(InstanceIds=worker_instance_ids)["Reservations"][0]["Instances"]])
    with ThreadingGroup(*worker_dns_names, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as worker_group:
        worker_group.run(f"tmux new -d -s {worker_tmux_session}")
        worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'cd auto-stop-tar/' ENTER")
        if args.update:
            logger.info("updating source on worker")
            worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'git checkout . && git reset && git clean -f && git pull origin fuzzy_artmap' ENTER")
        worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'source .venv/bin/activate' ENTER")
        worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'export PYTHONPATH=/home/ubuntu/auto-stop-tar/autostop/tar_framework:/home/ubuntu/auto-stop-tar/autostop/:/home/ubuntu/auto-stop-tar/' ENTER")
        worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'cd /home/ubuntu/auto-stop-tar/autostop/tar_framework/' ENTER")
        worker_group.run(f"tmux send-keys -t {worker_tmux_session}:0 'python fuzzy_artmap_distributed_gpu.py -r {leader_instance_private_ip}:8786 -v' ENTER")    
    logger.info("Worker clients started")


def create_leader(ec2_resource, ec2_client):
    global leader_instance
    global leader_instance_dns_name
    global leader_instance_id
    global leader_instance_private_ip
    leader_instance = ec2_resource.create_instances(
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "VolumeSize": 15,
                    "VolumeType": "gp3",
                    "Throughput": 125
                }
            }
        ],
        ImageId = base_ami,
        InstanceType = leader_instance_type,
        MaxCount = 1,
        MinCount = 1,
        KeyName=key_name,
        SecurityGroupIds = security_group_ids,
        SubnetId = subnet_id,
        DryRun = dry_run,
        IamInstanceProfile = leader_instance_role,
        InstanceInitiatedShutdownBehavior="stop",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "cluster_id",
                        "Value": args.id
                    },
                    {
                        "Key": "cluster_role",
                        "Value": leader_tag
                    }
                ]
            }
        ]
    )[0]
    logger.info(f"Leader {leader_instance.id} created, waiting for instance running")
    # waiter = ec2_client.get_waiter('instance_status_ok')
    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[leader_instance.id])
    logger.info(f"leader instance running, starting registrar")
    leader_instance_id = leader_instance.id
    leader_instance_private_ip = leader_instance.private_ip_address
    leader_instance_dns_name = ec2_client.describe_instances(InstanceIds=[leader_instance_id])["Reservations"][0]["Instances"][0]["PublicDnsName"]
    start_leader()

def start_leader():
    # start registrar
    with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
        leader_connection.run(f"tmux new -d -s {leader_tmux_session}")
        leader_connection.run(f"tmux send-keys -t {leader_tmux_session}:0 'cd auto-stop-tar/' ENTER")
        if args.update:
            logger.info("updating source on leader")
            leader_connection.run(f"tmux send-keys -t {leader_tmux_session}:0 'git checkout . && git reset && git clean -f && git pull origin fuzzy_artmap' ENTER")
        leader_connection.run(f"tmux send-keys -t {leader_tmux_session}:0 'source .venv/bin/activate' ENTER")
        leader_connection.run(f"tmux send-keys -t {leader_tmux_session}:0 'cd autostop' ENTER")
        leader_connection.run(f"tmux send-keys -t {leader_tmux_session}:0 'python registrar.py' ENTER")

    logger.info("leader registrar started")


running_ec2_filter = {"Name": "instance-state-name", "Values": ["running"]}
cluster_tag_filter = None
leader_tag_filter = {"Name": f"tag:cluster_role", "Values": [leader_tag]}
worker_tag_filter = {"Name": f"tag:cluster_role", "Values": [worker_tag]}

def find_leader(ec2_client):
    logger.info("looking for leader")
    global leader_instance_dns_name
    global leader_instance_id
    global leader_instance_private_ip
    leader_instance_info = ec2_client.describe_instances(Filters=[running_ec2_filter, cluster_tag_filter, leader_tag_filter])["Reservations"][0]["Instances"][0]    
    leader_instance_dns_name = leader_instance_info["PublicDnsName"]
    leader_instance_id = leader_instance_info["InstanceId"]
    leader_instance_private_ip = leader_instance_info["PrivateIpAddress"]
    logger.info(f"leader found at {leader_instance_id} - {leader_instance_dns_name}")

def find_workers(ec2_client): 
    logger.info("looking for workers")   
    global worker_instance_ids
    worker_instance_ids = list([instance["InstanceId"] for instance in ec2_client.describe_instances(Filters=[running_ec2_filter, cluster_tag_filter, worker_tag_filter])["Reservations"][0]["Instances"]])
    logger.info(f"found {len(worker_instance_ids)} workers - {worker_instance_ids}")

def connect_to_cluster():
    ec2_client = boto3.client('ec2', region_name=region)
    find_leader(ec2_client)
    find_workers(ec2_client)
    start_leader()
    start_workers(ec2_client)


def create_cluster():
    ec2_resource = boto3.resource("ec2", region_name=region)
    ec2_client = boto3.client('ec2', region_name=region)
    logger.info("creating workers")
    create_workers(ec2_resource)

    create_leader(ec2_resource, ec2_client)
    start_workers(ec2_client)

def start_run(params_file_location: str):
    params_file_name = path.basename(params_file_location)
    destination_path = f"auto-stop-tar/autostop/tar_model/{params_file_name}"
    with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
        # upload params file to leader
        logger.info(f"Uploading parameter file {params_file_location} to ~/{destination_path}")
        leader_connection.put(params_file_location, destination_path)
        logger.info("params uploaded & starting remote run")
        run_start_time = datetime.now()
        leader_connection.run(f"source ~/auto-stop-tar/.venv/bin/activate && cd ~/auto-stop-tar/ && python ~/auto-stop-tar/autostop/tar_model/fuzzy_artmap_tar.py -p {params_file_name}")
        run_stop_time = datetime.now()
        run_duration = run_stop_time - run_start_time
        logger.info(f"run completed\nstarted at {run_start_time}\nended at: {run_stop_time}\nelapsed: {run_duration}")
        

def cleanup_cluster(terminate_workers=False):
    ec2_client = boto3.client('ec2', region_name=region)

    with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
        run_timestamp = datetime.now().isoformat().replace("-", "_").replace(":","_").replace(".", "_")
        results_archive_file_name = f"keepsake_results_{args.id}_{run_timestamp}.tar.gz"
        s3_upload_command = f"aws s3 cp {results_archive_file_name} s3://{results_bucket}"
        logger.info("compressing and saving results to S3")  # TODO: update base AMI - sudo apt install awscli
        save_results_result = leader_connection.run(f"cd ~/auto-stop-tar/ && tar -czf {results_archive_file_name} ./.keepsake/ && {s3_upload_command}")
        logger.info(f"Save succeeded and terminate leader? {save_results_result.ok}")
    
    if save_results_result.ok:
        ec2_client.terminate_instances(InstanceIds=[leader_instance.id])
    else:
        ec2_client.stop_instances(InstanceIds=[leader_instance.id])

    if terminate_workers:
        ec2_client.terminate_instances(InstanceIds=worker_instance_ids)
    else:
        ec2_client.stop_instances(InstanceIds=worker_instance_ids)


if __name__ == "__main__":
    # "args": ["-p", "~/auto-stop-tar/autostop/tar_model/params.json", "-i", "test_run", "-w", "4", "-t", "--update"]
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-p", "--params", help="params file location", required=True)
    arg_parser.add_argument("-i", "--id", help="id to tag cluster", required=True)
    arg_parser.add_argument("-c", "--connect", help="connect to existing cluster with 'id'", default=False, type=bool, action=BooleanOptionalAction)
    arg_parser.add_argument("-w", "--workers", help="number of workers to run", type=int)
    arg_parser.add_argument("-t", "--terminate", help="flag to terminate worker instances instead of stop", action=BooleanOptionalAction)
    arg_parser.add_argument("-u", "--update", help="update leader and workers to latest code from git repo", default=False, type=bool, action=BooleanOptionalAction)
    args = arg_parser.parse_args()
    
    logger.info(f"starting with args: {args}")
    if args.workers:
        max_worker_count = args.workers
        min_worker_count = args.workers

    if args.connect:
        cluster_tag_filter = {"Name": f"tag:cluster_id", "Values": [args.id]}
        connect_to_cluster()
    else:
        create_cluster()
    
    logger.info("starting run")
    start_run(args.params)
    logger.info("run complete")

    logger.info("cleaning up cluster")
    cleanup_cluster(args.terminate)
