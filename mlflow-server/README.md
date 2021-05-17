# Boston House Price Prediction (Demo)

This is a DEMO created for the purpose of testing MLflow and deploy the ML model in real time EC2 instance.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.txt.

```bash
pip install -r requirements.txt
pip install gevent
pip install xgboost
```

## Usage

* Open terminal.
* You will need to run the navigate to the folder containing all the files. (```cd <folder path>```)
* Start the MLflow server by running : ```python run_MLFlow_server.py```
* Open another terminal and navigate to the current directory.
* Run the ```run_app_server.py``` file : ```python run_app_server.py```
* [Watch DEMO](https://drive.google.com/file/d/1ulOly7t_D5hP4p55m8SlG1NIynVAomXQ/view?usp=sharing)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)

