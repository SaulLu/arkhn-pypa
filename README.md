
#### Use of a virtual environment
For development purpose, we use a virtualenv to have the same library versions.

**Initiate the virtual environment**
This step has to be done only once.
```bash
virtualenv venv
```

**Start the virtual environment**
```bash
source venv/bin/activate
```
Install the project libraries
```bash
pip install -r requirements.txt
```

**Add library to the requirements**
Once you started the virtual environment, install the library using pip
```bash
pip install <name>
```


**Stop the virtual environment**
```bash
deactivate
```

#### Run the script
```bash
python main.py
```


