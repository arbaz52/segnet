# segnet
Keras, Tensorflow, Python, SegNet, Computer Vision, Image Segmentation, MaxPooling with UnMaxPooling

**Version 0.1**

***

## Contributors
- Arbaz Ajaz <arbaz5256@gmail.com>
- Kanwal Shariq

***

## Usage

### Using T.tf.Session() 

```Python
with K.tf.Session() as sess:
  batch = K.constant([[
      [[1],[2],[3],[4]],[[5],[6],[7],[8]],[[9],[10],[11],[12]],[[13],[14],[15],[16]]
  ],
                     [
      [[21],[22],[23],[24]],[[25],[26],[27],[28]],[[29],[210],[211],[212]],[[213],[214],[215],[216]]]])
  pooled, ind = MaxPoolingWithArgmax(batch)
  print(sess.run(UnMaxPooling([pooled, ind])))
```

### Using Lambda Layer in SegNet Model

```Python
  #Using the MaxPooling function
  layer = get_conv_block(layer, n_filters*2) #custom made function
  
  layer, m2 = Lambda(MaxPoolingWithArgmax)(layer)

  #Using the UnMaxPooling
  layer = get_conv_block(layer, n_filters*2) #custom made function
  layer = Lambda(UnMaxPoolingFixed)([layer, m2])
```