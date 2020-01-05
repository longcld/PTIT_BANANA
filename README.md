# References

## Cách build

```
cd team805
sudo docker build -t banana .
```

## Cách chạy (chú ý sửa đường dẫn tuyệt đối)


```
sudo docker run --rm -d -it --privileged -e DISPLAY=$DISPLAY --gpus=all --network=host -v /home/long/DigitalRace2019/Banana/src:/catkin_ws/src --name team805 banana bash
```
```
sudo docker exec -it team805 /bin/bash
```

# Chạy code điều khiển xe 
```
roslaunch team805 team805.launch
```

