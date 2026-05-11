import os, numpy as np
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

from nnom import generate_model

def _mel_to_hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def _hz_to_mel(f): return 2595.0*np.log10(1.0+f/700.0)
def _mel_spaced_hz(n, min_hz=50.0, sr=8000):
    nyq=sr/2.0; mel=np.linspace(_hz_to_mel(min_hz),_hz_to_mel(nyq),n+1); hz=_mel_to_hz(mel)
    return hz[:-1].astype(np.float32), np.diff(hz).astype(np.float32)

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class MelBandpassInitializer(tf.keras.initializers.Initializer):
    def __init__(self, n_filters=32, kernel_size=129, sample_rate=8000, min_hz=50.0):
        self.n_filters=n_filters; self.kernel_size=kernel_size
        self.sample_rate=sample_rate; self.min_hz=min_hz
    def _build_kernel(self):
        centers,bws=_mel_spaced_hz(self.n_filters,self.min_hz,self.sample_rate)
        half=(self.kernel_size-1)/2.0
        n=np.arange(0,self.kernel_size,dtype=np.float32)-half; ns=n/float(self.sample_rate)
        win=(0.54-0.46*np.cos(2.0*np.pi*np.arange(self.kernel_size)/(self.kernel_size-1))).astype(np.float32)
        F=np.zeros((self.kernel_size,self.n_filters),np.float32)
        for i in range(self.n_filters):
            fl=max(centers[i],self.min_hz); fh=min(fl+max(bws[i],self.min_hz),self.sample_rate/2.0-1.0)
            with np.errstate(divide='ignore',invalid='ignore'):
                lp=np.where(np.abs(ns)<1e-7,np.float32(2*fl/self.sample_rate),2*fl*np.sin(np.pi*2*fl*ns)/(np.pi*2*fl*ns))
                hp=np.where(np.abs(ns)<1e-7,np.float32(2*fh/self.sample_rate),2*fh*np.sin(np.pi*2*fh*ns)/(np.pi*2*fh*ns))
            bp=(hp-lp)*win; F[:,i]=bp/(np.sum(np.abs(bp))+1e-8)
        return F[:,np.newaxis,:]
    def __call__(self,shape,dtype=None): return tf.constant(self._build_kernel(),dtype=dtype or tf.float32)
    def get_config(self): return {'n_filters':self.n_filters,'kernel_size':self.kernel_size,'sample_rate':self.sample_rate,'min_hz':self.min_hz}

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class SmoothnessRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,weight=1e-4): self.weight=weight
    def __call__(self,k): return self.weight*tf.reduce_sum(tf.abs(k[1:,:,:]-k[:-1,:,:]))
    def get_config(self): return {'weight':self.weight}

model = tf.keras.models.load_model('mel_compact_4cmd_int8norm.h5',
    custom_objects={'MelBandpassInitializer':MelBandpassInitializer,
                    'SmoothnessRegularizer':SmoothnessRegularizer}, compile=False)

rng = np.random.default_rng(42)
x_dummy = rng.uniform(-1, 1, (256, 8000, 1)).astype(np.float32)

generate_model(model=model, x_test=x_dummy, name='/tmp/weights_stat_dump.h',
               quantize_method='kld', per_channel_quant=True)
