# A Practical Introduction to Reinforcement Learning
## November 16th, 2022
### João G. Ribeiro


#### Pre-Requisites

1) Python 3 (Code tested on 3.9, but 3.6-3.10 should work fine)
2) An IDE or code-editor that allows you to quickly jump between files (I use JetBrains' PyCharm)

#### Installation Instructions

1) Create Virtual Environment (change 3.9 to your version)


      $ python3.9 -m venv resources/venv

2) Activate and test Virtual Environment

    
      $ source resources/venv/bin/activate

      $ which python
        resources/venv/bin/python

      $ python
         Python 3.9.15 (main, Nov 10 2022, 20:55:15) 
         [GCC 12.1.1 20220730] on linux
         Type "help", "copyright", "credits" or "license" for more information.
         >>> exit() [ENTER]
         
         

3) Install requirements
    

      $ pip install -r resources/requirements.txt


4) Test installation


      $ python demo.py
