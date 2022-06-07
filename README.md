# cvapr-facenet

## Versions

- [<b> Python </b>](https://www.python.org/downloads/release/python-368/) - 3.6.8
- [<b> Tensorflow </b>](https://files.pythonhosted.org/packages/72/b8/2ef7057c956f1062ffab750a90a6bdcd3de127fb696fb64583c2dfe77aab/tensorflow-2.2.0-cp36-cp36m-win_amd64.whl) - 2.2.0
- <b> Keras </b> - 2.3.0

## Setup Virtualenv on Windows

Download Tensorflow `*.whl` file and put it in `/resources` folder
```console
py -3.6 -m pip install --upgrade pip
py -3.6 -m pip install virtualenv
py -3.6 -m virtualenv venv
.\venv\Scripts\activate
py -3.6 -m pip install -r requirements.txt
py -3.6 -m pip install ./resources/tensorflow-2.2.0-cp36-cp36m-win_amd64.whl
```
Next configure your IDE. [Help](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#env-requirements)

If any problem with activating venv try:

```
For Windows 11, Windows 10, Windows 7, Windows 8, Windows Server 2008 R2 or Windows Server 2012, run the following commands as Administrator:

x86 (32 bit)
Open C:\Windows\SysWOW64\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned

x64 (64 bit)
Open C:\Windows\system32\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned
```