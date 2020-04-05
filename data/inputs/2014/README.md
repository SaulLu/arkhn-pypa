### How to extract 2014 database

1. Download the directories **training-PHI-Gold-Set1/** and from 2014 database to /workdir/#YOURNAME/pypa/data/inputs/2014/

2. Run the scripts script_phi.sh or script_rf.sh

It will create csv files containing the parsed database to the **current directory**

3. You probably want to move them to a new directory, you can use following cmd
```
mkdir PHI
mv $(ls | grep csv) PHI
```
