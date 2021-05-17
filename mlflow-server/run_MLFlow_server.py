import subprocess

if __name__ == "__main__":

	# For production
    subprocess.call(['mlflow', 'server', '--backend-store-uri', './mlruns/' ,'--default-artifact-root', '<s3://bucket/folder>', '--host', '0.0.0.0', '--port', '5000'], shell=False)

    # For local
    # subprocess.call(['mlflow', 'server', '--backend-store-uri', './mlruns/', '--default-artifact-root', './mlruns/', '--host', '0.0.0.0', '--port', '5000'], shell=False)





# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
