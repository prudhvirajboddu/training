def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

# EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
#         efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]

def build_model(dim=128):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = efn.EfficientNetB5(input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = [binary_focal_loss()] 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model


# **Learning rate scheduler**

def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

with strategy.scope():
        model = build_model(dim=IMG_SIZES[0])


model.summary()


es=tf.keras.callbacks.EarlyStopping(monitor='val_auc',patience=5,restore_best_weights=True,mode='max',verbose=1)


history = model.fit(
        get_dataset(files_valid, augment=True, shuffle=True, repeat=True,
                dim=IMG_SIZES[0], batch_size = BATCH_SIZES[0],
                droprate=DROP_FREQ[1], dropct=DROP_CT[1], dropsize=DROP_SIZE[1]), 
        epochs=2, callbacks = [es,get_lr_callback(BATCH_SIZES[0])], 
        steps_per_epoch=count_data_items(files_train)/BATCH_SIZES[0]//REPLICAS
#         validation_data=get_dataset(files_valid,augment=False,shuffle=False,
#                 repeat=False,dim=IMG_SIZES[0]), #class_weight = {0:1,1:2},
#         validation_steps=count_data_items(files_valid)/BATCH_SIZES[0]//REPLICAS
    )



model.save('train.h5')