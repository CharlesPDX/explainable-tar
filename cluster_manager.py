import enum
import os
from os import path, getenv
from argparse import ArgumentParser, BooleanOptionalAction
import logging
from datetime import datetime
from pathlib import Path
import tarfile
import shutil
from time import sleep

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_logging_handler = logging.FileHandler('cluster_manager.log')
file_logging_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logging_handler.setFormatter(file_logging_format)
logger.addHandler(file_logging_handler)


import boto3
import requests
from fabric import Connection, ThreadingGroup
from dotenv import load_dotenv
settings_file = ".env_test"
load_dotenv(dotenv_path=Path(settings_file))

region = getenv("WORKING_REGION")
availability_zone = getenv("WORKING_AZ")
subnet_id = getenv("WORKING_SUBNET")
security_group_ids = [getenv("WORKING_SECURITY_GROUP")]
key_name = getenv("WORKING_KEY_NAME")
results_bucket = getenv("WORKING_BUCKET")
topic_arn = getenv("TOPIC_ARN")

min_worker_count = 1
max_worker_count = 1
worker_instance_type = "r6g.xlarge" # "r6g.2xlarge" #"r6g.4xlarge"
worker_tag = "worker"
worker_tmux_session = "worker"
worker_instance_ids = []
worker_dns_names = []

leader_instance_type = "r6a.large"
leader_tag = "leader"
leader_tmux_session = "registrar"
leader_manager_tmux_session = "runner"
leader_working_tmux_session = "leader"
leader_instance_role = {"Arn": getenv("WORKING_EC2_INSTANCE_ROLE_ARN")}

sidecar_instance_type= "t3a.medium"

leader_ami = getenv("LEADER_AMI")
base_ami = getenv("WORKER_AMI")
dry_run = False
ssh_key_location = getenv("WORKING_SSH_KEY_LOCATION")
instance_username = getenv("WORKING_INSTANCE_USERNAME")

leader_instance = None
leader_instance_dns_name = None
leader_instance_id = None
leader_instance_private_ip = None
can_terminate_leader = False

worker_volume_size = 15
leader_volume_size = 35
sidecar_volume_size = 35


def sub_create_workers(ec2_resource, worker_count):
   return ec2_resource.create_instances(
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "DeleteOnTermination": True,
                        "Iops": 3000,
                        "VolumeSize": worker_volume_size,
                        "VolumeType": "gp3",
                        "Throughput": 125
                    }
                }
            ],
            ImageId = base_ami,
            InstanceType = worker_instance_type,
            MaxCount = worker_count,
            MinCount = worker_count,
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

def create_workers(ec2_resource):
    global worker_instance_ids
    worker_instance_ids = []
    if max_worker_count > 4:
        half = int(max_worker_count / 2)
        worker_instances = sub_create_workers(ec2_resource, half)
        worker_instance_ids.extend(list([worker.id for worker in worker_instances]))
        sleep(5)
        worker_instances = sub_create_workers(ec2_resource, half)
        worker_instance_ids.extend(list([worker.id for worker in worker_instances]))
    else:
        worker_instances = sub_create_workers(ec2_resource, max_worker_count)
        worker_instance_ids.extend(list([worker.id for worker in worker_instances]))

    logger.info(f"{len(worker_instance_ids)} workers created\n{worker_instance_ids}")


def start_workers(ec2_client):
    global worker_dns_names
    logger.info(f"waiting for worker instances running")
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=worker_instance_ids)
    logger.info(f"workers running, starting worker clients")
    worker_dns_names = []
    worker_info = ec2_client.describe_instances(InstanceIds=worker_instance_ids)
    logger.info(f"fetched worker information for {len(worker_info['Reservations'])} reservations and {len(worker_info['Reservations'][0]['Instances'])} instances")
    for reservation in worker_info["Reservations"]:
        worker_dns_names.extend(list([instance["PublicDnsName"] for instance in reservation["Instances"]]))

    logger.info(f"received dns names for workers:\n{worker_dns_names}")
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

    if not args.autotar:
        instance_type = leader_instance_type
    else:
        instance_type = worker_instance_type

    leader_instance = ec2_resource.create_instances(
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "VolumeSize": leader_volume_size,
                    "VolumeType": "gp3",
                    "Throughput": 125
                }
            }
        ],
        ImageId = leader_ami,
        InstanceType = instance_type,
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
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[leader_instance.id])
    logger.info(f"leader instance running, starting registrar")
    leader_instance_id = leader_instance.id
    leader_instance_private_ip = leader_instance.private_ip_address
    leader_instance_dns_name = ec2_client.describe_instances(InstanceIds=[leader_instance_id])["Reservations"][0]["Instances"][0]["PublicDnsName"]
    if not args.autotar:
        start_leader()
    else:
        if args.update:
            logger.info("updating source on leader")
            with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
                leader_connection.run(f"cd auto-stop-tar/ && git checkout . && git reset && git clean -f && git pull origin fuzzy_artmap")

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
    if not args.autotar:
        find_workers(ec2_client)
        start_leader()
        start_workers(ec2_client)


def create_cluster():
    ec2_resource = boto3.resource("ec2", region_name=region)
    ec2_client = boto3.client('ec2', region_name=region)
    if not args.autotar:
        logger.info("creating workers")
        create_workers(ec2_resource)

    create_leader(ec2_resource, ec2_client)
    if not args.autotar:
        start_workers(ec2_client)

def start_run(params_file_location: str, total_runs: int, run_number: int):
    params_file_name = path.basename(params_file_location)
    destination_path = f"auto-stop-tar/autostop/tar_model/{params_file_name}"
    with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
        # upload params file to leader
        logger.info(f"Uploading parameter file {params_file_location} to ~/{destination_path}")
        leader_connection.put(params_file_location, destination_path)
        logger.info("params uploaded & starting remote run")
        run_start_time = datetime.now()
        if not args.autotar:
            leader_connection.run(f"source ~/auto-stop-tar/.venv/bin/activate && cd ~/auto-stop-tar/ && python ~/auto-stop-tar/autostop/tar_model/fuzzy_artmap_tar.py -p {params_file_name}")
        else:
            leader_connection.run(f"source ~/auto-stop-tar/.venv/bin/activate && cd ~/auto-stop-tar/ && python ~/auto-stop-tar/autostop/tar_model/autotar.py -p {params_file_name}")
        run_stop_time = datetime.now()
        run_duration = run_stop_time - run_start_time
        run_type = "fuzzy"
        if args.autotar:
            run_type = "auto"
        completion_message = f"***complete {run_type} run completed\n{params_file_location}\n{run_number} of {total_runs}\nstarted at {run_start_time}\nended at: {run_stop_time}\nelapsed: {run_duration}"
        logger.info(completion_message)
        sns_client = boto3.client('sns', region_name="us-west-2")
        sns_client.publish(TopicArn=topic_arn, Message=completion_message)
        

def save_results(params_path):
    try:
        local_logs_path = Path.cwd() / "logs/"
        local_logs_path.mkdir(exist_ok=True)
        ec2_client = boto3.client('ec2', region_name=region)
        worker_dns_names = list([instance["PublicDnsName"] for instance in ec2_client.describe_instances(InstanceIds=worker_instance_ids)["Reservations"][0]["Instances"]])
        for worker_index, worker_dns_name in enumerate(worker_dns_names):
            with Connection(host=worker_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as worker_connection:
                logger.info(f"getting worker {worker_index} log file to {str(local_logs_path / f'worker_{worker_index}_log.log')}")
                worker_connection.get("/home/ubuntu/auto-stop-tar/autostop/tar_framework/fuzzy_artmap_gpu_distributed.log", str(local_logs_path / f"worker_{worker_index}_log.log"))
                try:
                    logger.info("removing logs from worker")
                    worker_connection.run("rm -f /home/ubuntu/auto-stop-tar/autostop/tar_framework/fuzzy_artmap_gpu_distributed.log")
                except Exception as e:
                    logger.warning(f"Error removing logs from workers: {e}")                
    except Exception as e:
        logger.warning(f"Error getting logs from workers: {e}")
    
    with Connection(host=leader_instance_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as leader_connection:
        params_file_name = path.basename(params_path)
        run_prefix = ""
        if "." in params_file_name:
            run_prefix = params_file_name.split(".")[0] + "_"
        if args.autotar:
            run_prefix = "autotar_" + run_prefix
        run_timestamp = datetime.now().isoformat().replace("-", "_").replace(":","_").replace(".", "_")
                
        try:
            logger.info("getting logs from leader")
            if not args.autotar:
                leader_connection.get("/home/ubuntu/auto-stop-tar/fuzzy_artmap_gpu_distributed.log", str(local_logs_path / f"leader_fuzzy_artmap_gpu_distributed.log"))
            leader_connection.get("/home/ubuntu/auto-stop-tar/auto_tar.log", str(local_logs_path / f"auto_tar.log"))
            logger.info("compressing and saving logs to S3")
            logs_archive_file_name = f"{run_prefix}keepsake_results_{args.id}_{run_timestamp}_logs.tar.gz"
            with tarfile.open(logs_archive_file_name, mode="w:gz") as logs_archive:
                logs_archive.add(str(local_logs_path))
            s3_client = boto3.client('s3', region_name=region)
            s3_client.upload_file(logs_archive_file_name, results_bucket, logs_archive_file_name)            
            logger.info("logs uploaded successfully, removing from cluster manager")
            shutil.rmtree(str(local_logs_path), ignore_errors=True)
            os.remove(logs_archive_file_name)
        except Exception as e:
            logger. warning(f"Error compressing logs and uploading to s3: {e}")
        finally:
            try:
                logger.info("removing logs from leader")
                leader_connection.run("rm -f /home/ubuntu/auto-stop-tar/auto_tar.log")
                leader_connection.run("rm -f /home/ubuntu/auto-stop-tar/fuzzy_artmap_gpu_distributed.log")
            except Exception as e:
                logger. warning(f"Error removing logs from leader: {e}")
            
        
        results_archive_file_name = f"{run_prefix}keepsake_results_{args.id}_{run_timestamp}.tar.gz"
        s3_upload_command = f"aws s3 cp {results_archive_file_name} s3://{results_bucket}"
        logger.info("compressing and saving results to S3") 
        save_results_result = leader_connection.run(f"cd ~/auto-stop-tar/ && tar -czf {results_archive_file_name} ./.keepsake/ && {s3_upload_command}")
        logger.info(f"Save succeeded and delete keepsake directory? {save_results_result.ok}")
        
        if save_results_result.ok:
            leader_connection.run(f"cd ~/auto-stop-tar/ && rm -f *.gz && rm -rf ./.keepsake/ && cd models && rm -f *.*")
        else:
            leader_connection.run(f"cd ~/auto-stop-tar/ && mv .keepsake {run_prefix}_keepsake")
            logger.warning(f"results saved on leader to: ~/auto-stop-tar/{run_prefix}_keepsake")
        
        return save_results_result.ok

def cleanup_cluster(terminate_leader=True, terminate_workers=False):
    ec2_client = boto3.client('ec2', region_name=region)

    try:
        logger.info(f"Terminate leader?: {terminate_leader}")
        if terminate_leader:
            ec2_client.terminate_instances(InstanceIds=[leader_instance_id])
        else:
            ec2_client.stop_instances(InstanceIds=[leader_instance_id])
        if not args.autotar:
            logger.info(f"Terminate workers?: {terminate_workers}")
            if terminate_workers:
                ec2_client.terminate_instances(InstanceIds=worker_instance_ids)
            else:
                ec2_client.stop_instances(InstanceIds=worker_instance_ids)

        logger.info("Checking for running as sidecar")
        try:
            logger.info("setting token")
            token_response = requests.put("http://169.254.169.254/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, timeout=5.0)
            logger.info(f"token response: {token_response.status_code} - {token_response.text}")
            if token_response.status_code != 200:
                token_response.raise_for_status()
            token = token_response.text
            logger.info("getting instance id")
            instance_response = requests.get("http://169.254.169.254/latest/meta-data/instance-id", headers={"X-aws-ec2-metadata-token": token}, timeout=1.0)
            logger.info(f"instance repsonse: {instance_response.status_code} - {instance_response.text}")
            sidecar_instance_id = instance_response.text
            if terminate_leader:
                ec2_client.terminate_instances(InstanceIds=[sidecar_instance_id])
            else:
                ec2_client.stop_instances(InstanceIds=[sidecar_instance_id])
        except Exception as e:
            logger.info(f"Error getting token or instance id: {e}")
    except Exception as e:
        logger.error(f"Error stopping or terminating cluster!! {str(e)}")
        sns_client = boto3.client('sns', region_name="us-west-2")
        sns_client.publish(TopicArn=topic_arn, Message=f"Error stopping or terminating cluster!! {str(e)}")

def bootstrap_sidecar(prefixes):
    ec2_resource = boto3.resource("ec2", region_name=region)
    ec2_client = boto3.client('ec2', region_name=region)
    sidecar_instance = ec2_resource.create_instances(
    BlockDeviceMappings=[
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "Iops": 3000,
                "VolumeSize": sidecar_volume_size,
                "VolumeType": "gp3",
                "Throughput": 125
            }
        }
    ],
    CreditSpecification={
        'CpuCredits': 'standard'
    },
    ImageId = leader_ami,
    InstanceType = sidecar_instance_type,
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
                    "Value": "sidecar"
                }
            ]
        }
    ]
    )[0]
    logger.info(f"Sidecar {sidecar_instance.id} created, waiting for instance running")
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[sidecar_instance.id])
    logger.info(f"sidecar instance running, starting bootstrap")
    sidecar_information = ec2_client.describe_instances(InstanceIds=[sidecar_instance.id])
    sidecar_dns_name = sidecar_information["Reservations"][0]["Instances"][0]["PublicDnsName"]
    with Connection(host=sidecar_dns_name, user=instance_username, connect_kwargs={"key_filename": ssh_key_location}) as sidecar_connection:
        # upload params file to sidecar
        for prefix in prefixes:
            params_path = f"autostop/tar_model/{prefix}_params.json"
            params_file_name = path.basename(params_path)
            destination_path = f"auto-stop-tar/autostop/tar_model/{params_file_name}"
            logger.info(f"Uploading parameter file {params_path} to ~/{destination_path}")
            sidecar_connection.put(params_path, destination_path)

        # upload cluster manager
        cluster_manager_path = path.abspath(__file__)
        cluster_manager_destination = f"auto-stop-tar/cluster_manager.py"
        logger.info(f"Uploading cluster manager {cluster_manager_path} to ~/{cluster_manager_destination}")
        sidecar_connection.put(cluster_manager_path, cluster_manager_destination)

        # read & write updated env file
        env_update_path = f"{os.getcwd()}/{settings_file}_update"
        env_destination = f"auto-stop-tar/{settings_file}"
        with open(f"{os.getcwd()}/{settings_file}", mode="r") as env_file, open(env_update_path, mode="w+") as updated_env_file:
            for line in env_file:
                if "SSH_KEY" not in line:
                    updated_env_file.write(line)
                else:
                    parts = line.split("=")
                    ssh_file_name = path.basename(parts[1].rstrip())
                    ssh_destination = f"/home/ubuntu/auto-stop-tar/{ssh_file_name}"
                    updated_env_file.write(f"{parts[0]}={ssh_destination}\n")
                    logger.info(f"Uploading ssh_key {parts[1].rstrip()} to {ssh_destination}")
                    sidecar_connection.put(parts[1].rstrip(), ssh_destination)
        
        logger.info(f"Uploading env file {env_update_path} to {env_destination}")
        sidecar_connection.put(env_update_path, env_destination)
        logger.info("Sidecar prepared!")
        logger.info(f"run command:\nssh -i {ssh_key_location} ubuntu@{sidecar_information['Reservations'][0]['Instances'][0]['PublicIpAddress']}")
        logger.info(f"follow on commands\ntmux new -d -s cluster_manager && tmux attach -t cluster_manager\ncd auto-stop-tar/ && source .venv/bin/activate\npython ./cluster_manager.py -i {args.id} -w 4 -t\nor\npython ./cluster_manager.py -i {args.id} -a -t")

if __name__ == "__main__":
    # "args": ["-p", "~/auto-stop-tar/autostop/tar_model/params.json", "-i", "test_run", "-w", "4", "-t", "--update"]
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--id", help="id to tag cluster", required=True)
    arg_parser.add_argument("-c", "--connect", help="connect to existing cluster with 'id'", default=False, type=bool, action=BooleanOptionalAction)
    arg_parser.add_argument("-w", "--workers", help="number of workers to run", type=int)
    arg_parser.add_argument("-t", "--terminate", help="flag to terminate worker instances instead of stop", action=BooleanOptionalAction)
    arg_parser.add_argument("-u", "--update", help="update leader and workers to latest code from git repo", default=False, type=bool, action=BooleanOptionalAction)
    arg_parser.add_argument("-b", "--bootstrap", help="bootstrap cluster manager on the leader", default=False, type=bool, action=BooleanOptionalAction)
    arg_parser.add_argument("-a", "--autotar", help="run AutoTAR baseline", default=False, type=bool, action=BooleanOptionalAction)
    args = arg_parser.parse_args()
    
    logger.info(f"starting with args: {args}")
    if args.workers:
        max_worker_count = args.workers
        min_worker_count = args.workers
    
    # prefixes = ["alpha", "beta", "charlie", "delta", "epsilon", "zeta"]
    # prefixes = ["beta", "charlie", "delta", "epsilon", "zeta"]
    # prefixes = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    # prefixes = ["foxtrot_18", "foxtrot_19", "foxtrot_20", "foxtrot_21", "foxtrot_22", "foxtrot_23", "foxtrot_24", "foxtrot_25", "foxtrot_26", "foxtrot_27", "foxtrot_28", "foxtrot_29", "foxtrot_30", "foxtrot_31", "foxtrot_32", "foxtrot_33", "foxtrot_34", "foxtrot_35", "foxtrot_36", "foxtrot_37", "foxtrot_38", "foxtrot_39", "foxtrot_40", "foxtrot_41", "foxtrot_42", "foxtrot_43", "foxtrot_44"]
    # prefixes = ["test_small_reuters"]
    # prefixes = ["golf_3", "golf_4", "golf_5", "golf_6", "hotel_0", "hotel_1", "hotel_2", "hotel_3", "hotel_4", "hotel_5", "hotel_6", "hotel_7", "hotel_8", "hotel_9", "hotel_10", "hotel_11", "hotel_12", "hotel_13", "hotel_14", "hotel_15", "hotel_16", "hotel_17", "hotel_18", "hotel_19", "hotel_20", "hotel_21", "hotel_22", "hotel_23", "hotel_24", "hotel_25", "hotel_26", "hotel_27", "hotel_28", "hotel_29", "hotel_30", "hotel_31", "hotel_32", "hotel_33", "hotel_34", "hotel_35", "hotel_36", "hotel_37", "hotel_38", "hotel_39", "hotel_40", "hotel_41", "hotel_42", "hotel_43", "hotel_44"]
    # prefixes = ["india_1", "india_2", "india_3", "india_4", "india_5", "india_6", "india_7", "india_8", "india_9", "india_10", "india_11", "india_12", "india_13", "india_14", "india_15", "india_16", "india_17", "india_18", "india_19", "india_20", "india_21", "india_22", "india_23", "india_24", "india_25", "india_26", "india_27", "india_28", "india_29", "india_30", "india_31", "india_32", "india_33", "india_34", "india_35", "india_36", "india_37", "india_38", "india_39", "india_40", "india_41", "india_42", "india_43", "india_44", "india_45", "india_46", "india_47", "india_48", "india_49", "india_50", "india_51", "india_52", "india_53", "india_54", "india_55", "india_56", "india_57", "india_58", "india_59", "india_60", "india_61", "india_62", "india_63", "india_64", "india_65", "india_66", "india_67", "india_68", "india_69", "india_70", "india_71", "india_72", "india_73", "india_74", "india_75", "india_76", "india_77", "india_78", "india_79", "india_80", "india_81", "india_82", "india_83", "india_84", "india_85", "india_86", "india_87", "india_88", "india_89", "india_90", "india_91", "india_92", "india_93", "india_94", "india_95", "india_96", "india_97", "india_98", "india_99", "india_100", "india_101", "india_102", "juliett_0", "juliett_1", "juliett_2", "juliett_3", "juliett_4", "juliett_5", "juliett_6", "juliett_7", "juliett_8", "juliett_9", "kilo_0", "kilo_1", "kilo_2", "kilo_3", "kilo_4", "kilo_5", "kilo_6", "kilo_7", "kilo_8", "kilo_9", "kilo_10", "kilo_11", "kilo_12", "kilo_13", "kilo_14", "kilo_15", "kilo_16", "kilo_17", "kilo_18", "kilo_19", "kilo_20", "kilo_21", "kilo_22", "kilo_23", "kilo_24", "kilo_25", "kilo_26", "kilo_27", "kilo_28", "kilo_29", "kilo_30", "kilo_31", "kilo_32", "kilo_33"]
    # prefixes = ["india_1"]
    # prefixes = ["bravo_echo_0", "bravo_echo_1", "bravo_echo_2", "bravo_echo_3", "bravo_echo_4", "bravo_echo_5", "bravo_echo_6", "bravo_foxtrot_0", "bravo_foxtrot_1", "bravo_foxtrot_2", "bravo_foxtrot_3", "bravo_foxtrot_4", "bravo_foxtrot_5", "bravo_foxtrot_6", "bravo_foxtrot_7", "bravo_foxtrot_8", "bravo_foxtrot_9", "bravo_foxtrot_10", "bravo_foxtrot_11", "bravo_foxtrot_12", "bravo_foxtrot_13", "bravo_foxtrot_14", "bravo_foxtrot_15", "bravo_foxtrot_16", "bravo_foxtrot_17", "bravo_foxtrot_18", "bravo_foxtrot_19", "bravo_foxtrot_20", "bravo_foxtrot_21", "bravo_foxtrot_22", "bravo_foxtrot_23", "bravo_foxtrot_24", "bravo_foxtrot_25", "bravo_foxtrot_26", "bravo_foxtrot_27", "bravo_foxtrot_28", "bravo_foxtrot_29", "bravo_foxtrot_30", "bravo_foxtrot_31", "bravo_foxtrot_32", "bravo_foxtrot_33", "bravo_foxtrot_34", "bravo_foxtrot_35", "bravo_foxtrot_36", "bravo_foxtrot_37", "bravo_foxtrot_38", "bravo_foxtrot_39", "bravo_foxtrot_40", "bravo_foxtrot_41", "bravo_foxtrot_42", "bravo_foxtrot_43", "bravo_foxtrot_44", "bravo_golf_0", "bravo_golf_1", "bravo_golf_2", "bravo_golf_3", "bravo_golf_4", "bravo_golf_5", "bravo_golf_6", "bravo_hotel_0", "bravo_hotel_1", "bravo_hotel_2", "bravo_hotel_3", "bravo_hotel_4", "bravo_hotel_5", "bravo_hotel_6", "bravo_hotel_7", "bravo_hotel_8", "bravo_hotel_9", "bravo_hotel_10", "bravo_hotel_11", "bravo_hotel_12", "bravo_hotel_13", "bravo_hotel_14", "bravo_hotel_15", "bravo_hotel_16", "bravo_hotel_17", "bravo_hotel_18", "bravo_hotel_19", "bravo_hotel_20", "bravo_hotel_21", "bravo_hotel_22", "bravo_hotel_23", "bravo_hotel_24", "bravo_hotel_25", "bravo_hotel_26", "bravo_hotel_27", "bravo_hotel_28", "bravo_hotel_29", "bravo_hotel_30", "bravo_hotel_31", "bravo_hotel_32", "bravo_hotel_33", "bravo_hotel_34", "bravo_hotel_35", "bravo_hotel_36", "bravo_hotel_37", "bravo_hotel_38", "bravo_hotel_39", "bravo_hotel_40", "bravo_hotel_41", "bravo_hotel_42", "bravo_hotel_43", "bravo_hotel_44"]
    # ranked down-reuters: 
    # prefixes = ["india_0", "india_1", "india_2", "india_3", "india_4", "india_5", "india_6", "india_7", "india_8", "india_9", "india_10", "india_11", "india_12", "india_13", "india_14", "india_15", "india_16", "india_17", "india_18", "india_19", "india_20", "india_21", "india_22", "india_23", "india_24", "india_25", "india_26", "india_27", "india_28", "india_29"]
    # random down-reuters: 
    # prefixes =["bravo_india_0", "bravo_india_1", "bravo_india_2", "bravo_india_3", "bravo_india_4", "bravo_india_5", "bravo_india_6", "bravo_india_7", "bravo_india_8", "bravo_india_9", "bravo_india_10", "bravo_india_11", "bravo_india_12", "bravo_india_13", "bravo_india_14", "bravo_india_15", "bravo_india_16", "bravo_india_17", "bravo_india_18", "bravo_india_19", "bravo_india_20", "bravo_india_21", "bravo_india_22", "bravo_india_23", "bravo_india_24", "bravo_india_25", "bravo_india_26", "bravo_india_27", "bravo_india_28", "bravo_india_29"]

    # ranked down-tr4: 
    # prefixes = ["juliett_0", "juliett_1", "juliett_2", "juliett_3", "juliett_4", "juliett_5", "juliett_6", "juliett_7", "juliett_8", "juliett_9", "juliett_10", "juliett_11", "juliett_12", "juliett_13", "juliett_14", "juliett_15", "juliett_16", "juliett_17", "juliett_18", "juliett_19", "juliett_20", "juliett_21", "juliett_22", "juliett_23", "juliett_24", "juliett_25", "juliett_26", "juliett_27", "juliett_28", "juliett_29"]
    # random down-tr4:
    # prefixes = ["bravo_juliett_0", "bravo_juliett_1", "bravo_juliett_2", "bravo_juliett_3", "bravo_juliett_4", "bravo_juliett_5", "bravo_juliett_6", "bravo_juliett_7", "bravo_juliett_8", "bravo_juliett_9", "bravo_juliett_10", "bravo_juliett_11", "bravo_juliett_12", "bravo_juliett_13", "bravo_juliett_14", "bravo_juliett_15", "bravo_juliett_16", "bravo_juliett_17", "bravo_juliett_18", "bravo_juliett_19", "bravo_juliett_20", "bravo_juliett_21", "bravo_juliett_22", "bravo_juliett_23", "bravo_juliett_24", "bravo_juliett_25", "bravo_juliett_26", "bravo_juliett_27", "bravo_juliett_28", "bravo_juliett_29"]
    # prefixes = ["charlie_juliett_1", "charlie_juliett_2", "charlie_juliett_3", "charlie_juliett_4", "charlie_juliett_5", "charlie_juliett_6", "charlie_juliett_7", "charlie_juliett_8", "charlie_juliett_9", "charlie_juliett_10", "charlie_juliett_11", "charlie_juliett_12", "charlie_juliett_13", "charlie_juliett_14", "charlie_juliett_15", "charlie_juliett_16", "charlie_juliett_17", "charlie_juliett_18", "charlie_juliett_19", "charlie_juliett_20", "charlie_juliett_21", "charlie_juliett_22", "charlie_juliett_23", "charlie_juliett_24", "charlie_juliett_25", "charlie_juliett_26", "charlie_juliett_27", "charlie_juliett_28", "charlie_juliett_29"]
    prefixes = ["charlie_juliett_0"]
    try:
        if args.bootstrap:
            bootstrap_sidecar(prefixes)
            logger.info("bootstrap complete, exiting")
            exit()
    except Exception as e:
        terminate_leader = False
        logger.warning(str(e))
        exit()
    
    last_param_path = ""
    try:
        terminate_leader = True
        if args.connect:
            cluster_tag_filter = {"Name": f"tag:cluster_id", "Values": [args.id]}
            connect_to_cluster()
        else:
            create_cluster()

        number_of_prefixes = len(prefixes)
        for prefix_number, prefix in enumerate(prefixes):
            params_path = f"autostop/tar_model/{prefix}_params.json"
            last_param_path = params_path
            
            logger.info(f"starting run: {params_path}")
            start_run(params_path, number_of_prefixes, prefix_number + 1)
            logger.info(f"run complete: {params_path}")

            logger.info("saving results")
            results_saved_successfully = save_results(params_path)
            terminate_leader = terminate_leader and results_saved_successfully
            logger.info(f"results saved successfully: {results_saved_successfully} - {params_path}")

        logger.info("cleaning up cluster")        
    except Exception as e:
        terminate_leader = False
        logger.warning(str(e))
        sns_client = boto3.client('sns', region_name="us-west-2")
        sns_client.publish(TopicArn=topic_arn, Message=f"error executing run <{last_param_path}>: {str(e)}")
    finally:
        cleanup_cluster(terminate_leader, args.terminate)