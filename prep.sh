cd homework/hw3
pip install -e .
pip install tensorboardX
apt update
apt install libglfw3 libglfw3-dev libosmesa6-dev
pip install gym==0.25.2
pip install mujoco
pip install swig
pip install opencv-python==4.6.0.66
pip install box2d-py
mkdir ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar xvf mujoco210-linux-x86_64.tar.gz 
ls
rm mujoco210-linux-x86_64.tar.gz 
cd mujoco210/
cd /workspace
cd homework/hw3
pip install mujoco-py
pip install opencv-python==4.5.5.64
python
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin >> ~/.bashrc
source ~/.bashrc

