import subprocess

if __name__ == "__main__":

    subprocess.call(["gunicorn", "-b", "0.0.0.0:8200", "app:app"], shell=False)




# gunicorn -b 0.0.0.0:8200 app:app --workers=5
