import h5py

weights_path = '.../vgg16_weights.h5'
f=h5py.File(weights_path)
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):#attrs是指向f中的属性，点击右键可以看见这个属性
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]#format是格式化的意思，输出g就是format（k）填充到{}上
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #in range(g.attrs[‘nb_params’])]的含义
    #  得到的是layer下param_0、param_1等
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')