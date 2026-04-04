mkdir -p ./data2/

# sun rgbd
mkdir -p ~/tmp/windows_share/sunrgbd
sudo mount -t cifs "//192.168.31.194/datasets/SUN RGB-D/sunrgbd" ~/tmp/windows_share/sunrgbd -o credentials=/root/.smbcredentials,iocharset=utf8,dir_mode=0777,file_mode=0777
ln -s  ~/tmp/windows_share/sunrgbd ~/ws/py/mmdetection3d/data2/sunrgbd

# TO Secne-down
mkdir -p ~/tmp/windows_share/TO-scannet
mkdir -p data2/TO-SCENE-down
sudo mount -t cifs "//192.168.31.194/datasets/TO-Scene-down/TO-scannet" ~/tmp/windows_share/TO-scannet -o credentials=/root/.smbcredentials,iocharset=utf8,dir_mode=0777,file_mode=0777
ln -s  ~/tmp/windows_share/TO-scannet data2/TO-SCENE-down/TO-scannet
