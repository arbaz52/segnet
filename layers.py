def MaxPoolingWithArgmax(inputs):
  output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
  return [output, argmax]

def UnMaxPooling(inputs):
  pooled, ind = inputs
  
  ind = K.cast(ind, "int32")
  input_shape = K.tf.shape(pooled, "int32")
  
  output_shape = [input_shape[0],
                 input_shape[1]*2,
                 input_shape[2]*2,
                 input_shape[3]]
  
  input_image_fatness = input_shape[1]*input_shape[2]
  ind = K.reshape(ind, [-1, input_image_fatness])
  batch_range = tf.range(tf.shape(ind)[0])
  output_image_fatness = output_shape[1]*output_shape[2]
  
  ind = K.expand_dims(output_image_fatness*batch_range) + ind
  
  unpooled = K.tf.scatter_nd(K.expand_dims(K.flatten(ind)),
                            K.flatten(pooled),
                            [K.prod(output_shape)])
  input_shape = pooled.shape
  output_shape = [-1,
                 input_shape[1]*2,
                 input_shape[2]*2,
                 input_shape[3]]
  return K.reshape(unpooled, output_shape)
  
#usage  
with K.tf.Session() as sess:
  batch = K.constant([[
      [[1],[2],[3],[4]],[[5],[6],[7],[8]],[[9],[10],[11],[12]],[[13],[14],[15],[16]]
  ],
                     [
      [[21],[22],[23],[24]],[[25],[26],[27],[28]],[[29],[210],[211],[212]],[[213],[214],[215],[216]]]])
  pooled, ind = MaxPoolingWithArgmax(batch)
  print(sess.run(UnMaxPooling([pooled, ind])))