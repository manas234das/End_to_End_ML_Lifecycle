# AWS_deployment

This is a DEMO created for the purpose of testing MLflow and deploy the ML model in real time EC2 instance.

## Installation

* All the files that are needed to create the containers are included. 
* Make sure you have docker installed in your system. If not follow the instruction.
    * Install docker engine -> [Docker engine installation](https://docs.docker.com/engine/install/ubuntu/)
    * Install docker compose -> [Docker compose installation](https://docs.docker.com/compose/install/)
* You just need to clone all the files and run the command : `docker-compose build --compress --force-rm`
* Make sure that you are in the directory containing the `docker-compose.yml` file before you execute the above command.


## Usage

* Open terminal.
* You will need to run the navigate to the folder containing all the files. (```cd <folder path>```)
* Execute : `docker-compose up`
* Ip address will be displayed where the server is running .i.e. `http://0.0.0.0:8200`
* You can now browse through the application.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)

