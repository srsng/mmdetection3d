sudo mount -t cifs "//192.168.31.194/datasets/SUN RGB-D/sunrgbd" ~/tmp/windows_share -o credentials=/root/.smbcredentials,iocharset=utf8,dir_mode=0777,file_mode=0777

mkdir -p ./data2/ && ln -s  ~/tmp/windows_share ~/ws/py/mmdetection3d/data2/sunrgbd
