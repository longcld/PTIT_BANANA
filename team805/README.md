# References

## Cách build

```docker build -t banana .```

## Cách chạy (chú ý sửa đường dẫn tuyệt đối)


```docker run --rm -it --gpus=all --network=host -v /home/huy/DigitalRace2019/Banana/src:/catkin_ws/src --name banana banana bash```

# Dành cho ban tổ chức 
```roslaunch banana banana.launch```

