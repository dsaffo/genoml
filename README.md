# GenoML v2 Python Refactoring 
Branch Breakdown:
1. The workspace folder is the setup I have for my Visual Studio code so if you open it there things should be a lot easier 
2. I added the necessary example data but to test things the way I am testing them please see the `GettingStarted.sh` script 
3. The `GettingStarted.sh` script goes through how to setup a virtual conda environment to keep everything contained, along with how I am testing each component

**If you would like to contribute, please open a pull request and if you got something working, please update this TODO and `GettingStarted.sh` accordingly.**

# TODO List 
### CURRENTLY WORKING ON:
- Fix PLINK issue (PLINK has a redistributable license - just add to GenoML package and export PATH?)

### CRUCIAL:
- Add VIF calculation
- Address all FutureWarnings brought up in the training.py script (genoml.discrete.supervised.training)
    - These will likely cause issues for users in the future 
- Refactor continuous train + tune

### SECONDARY:
- Insert switch to check Python 3.7 
- Clean up formatting/files
- Change strings to format strings 
- Adding details comments to orient the user 
- Adding detailed argument comments to specify inputs and outputs 
- MIKE: Addition of new code?

### LAST:
- Update file_structure.txt in /docs 
- Update LICENSE.txt
- Update README.md 
- Fix grammatical/spelling errors 
- Make text more user-friendly
- Make a small tutorial 

### DONE:
- Easily install xgboost -- Done and updated the requirements file
- Munging has been refactored -- working on PLINK path issue + adding VIF 
- Discrete train has been refactored -- Some clean-up necessary
- Discrete tune has been refactored -- Some clean-up necessary

### TODO:
- Refactoring:
    - Munging 
        - Still need to add VIF
    - Discrete
        - Train 
        - Tune 
        - Test (needs to be added)
    - Continuous
        - Train
        - Tune
        - Test (needs to be added)
    - Add in Mike's additional switches 
    - Add in Hampton's UKBB script 

