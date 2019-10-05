import tensorflow as tf
#
#
#def restrct_price_throshuld(net_out,low,high):    
#    
#    lowRangeMask  = tf.less_equal(net_out,low)
#    HighRangeMask = tf.greater_equal(net_out,high)
#    mask = tf.logical_or(lowRangeMask , HighRangeMask);
#    v = tf.reduce_max(tf.boolean_mask(net_out, mask))
#    
#    return v
#
#x = tf.placeholder("float", None)
#z = tf.placeholder("float", None)
#
#with tf.Session() as session:
#    result = session.run(restrct_price_throshuld, feed_dict={x: [1, 2, 3],z: [1, 2, 3]})
#    print(result)

#a = True
#b = False
#c = (a*b)
#d = tf.cast(c, tf.bool)
#d1 = tf.logical_or(a,b)
#sess = tf.Session()
#vv = sess.run(d1)
#print(d)


x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.ones_like(x)
out = tf.greater(x, y)
os1 = tf.reduce_sum(tf.boolean_mask(y, out))
sess = tf.Session()
vv = sess.run(os1)