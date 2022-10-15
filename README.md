# Tensorflow GPU Setup Windows 10 - without Anaconda

Based on this amazing guide by Prof Jeff Heaton: https://www.youtube.com/watch?v=qrkEYf-YDyI

## Step 1: Find which GPU do I have?

First we will need to detect the GPU we have present on our system

Open the **Start** menu on your PC, type "**Device Manager**," and press Enter. You should see an option near the top for **Display Adapters**. Click the drop-down arrow, and it should list the name of your GPU right there (credits: [pcmag](https://in.pcmag.com/cpus-components/84313/what-graphics-card-do-i-have))

For example, I have `NVIDIA GEFORCE GTX 1650`

## Step 2: Get NVIDIA GPU Drivers

These generic GPU Drivers are the ones we generally setup for gameplay graphics. 

1. Visit [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)
2. You should see something like this under "**Software requirements**"

    ![Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag1.png](Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag1.png)

3. These versions are really important and will be like a guide throughout the article. NOTE here we are installing **Tensorflow 2.1.0**.
4. First as you can see we need CUDA 10.1 for Tensorflow 2.1.0. and that requires NVIDIA GPU Driver 418.x or higher
5. Basically we need `Tensorflow 2.1.0 → ≥ NVIDIA GPU Driver 418.x → CUDA 10.1 → CUPTI → cuDNN SDK (≥ 7.6)`. This is what we have to keep in mind
6. Download the drivers for your GPU. For example, I have:

    ![Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag2.png](Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag2.png)

7. Complete the installation under the default **Express** settings

## Step 3: Get CUDA

Before downloading CUDA if u had any prior versions of CUDA already in your system, uninstall it. you can check by searching "Apps and features" in windows. For example:

![Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag4.png](Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag4.png)

Its best to clear out the earlier versions

Get CUDA toolkit from the link: [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

Note for our current configuration we need **CUDA 10.1**. This might be different for you based on the tensorflow version you are using. Again keep this hierarchy in mind:

`Tensorflow 2.1.0 → ≥ NVIDIA GPU Driver 418.x → CUDA 10.1 → CUPTI → cuDNN SDK (≥ 7.6)`

![Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag3.png](Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag3.png)

Download and install CUDA (standard procedure, just follow the instructions)

## Step 4: Get cuDNN

Go to this link: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

We have to sign up for an NVIDIA account before downloading cuDNN

Get the appropriate version. For example, **since I had CUDA 10.1, I need cuDNN 7.6.5**

![Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag5.png](Tensorflow%20GPU%20Setup%20Windows%2010%20without%20Anaconda%206a49ced466f041898a5da1145ef5676a/diag5.png)

## Step 5: Set PATH variables

Now we have cuDNN which is a zip file. Open it up and u should see a folder "cuda" inside. Now in the link [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) under "Windows setup" you can see something like:

```python
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

As we can see it expects cuda to be in the location `C:\tools\cuda\bin`

We create a folder `\tools` and copy the cuda folder in it.

Now we edit the environment variables

Go to **System Properties → Environment variables → System variables**

We can see that some variables like `CUDA_PATH` and `CUDA_PATH_V10_1` are already set, that is fine 

Go to `Path` variable

Set the following variables one by one (make sure to remove duplicates if any) from the snippet we had above

1. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
2. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64
3. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include
4. C:\tools\cuda\bin

Ok, so now we have all the dependencies and we can go on to creating a new virtual environment and installing tensorflow

## Step 6: Virtual environment setup

Let us setup a clean new virtual environment in which we will install tensorflow

1. Navigate to your project directory
2. Run the command:

    ```bash
    pip install virtualenv
    ```

3. Create a new environment using:

    ```bash
    python -m virtualenv gpu_acclerated

    ```

4. Activate the environment

    ```bash
    .\gpu_acclerated\Scripts\activate
    ```

5. Nice! You are in the `gpu_acclerated` environment now:

    ```bash
    (gpu_acclerated) PS C:\Users\shaun\Documents\my_projects> pip list
    Package    Version
    ---------- -------
    pip        20.1.1
    setuptools 47.1.1
    wheel      0.34.2
    ```

## Step 7: Install and run tensorflow

Run the command in your virtual environment:

```bash
(gpu_acclerated) PS C:\Users\shaun\Documents\my_projects> pip install tensorflow-gpu==2.1.0
```

Once it finishes, open up a python prompt from within the environment  and import tensorflow and check if gpu is found. It should look something like:

```bash
gpu_acclerated) PS C:\Users\shaun\Documents\my_projects> python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2020-07-05 14:50:17.103317: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
>>> tf.__version__
'2.1.0'
>>> tf.test.is_gpu_available()
WARNING:tensorflow:From <stdin>:1: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
2020-07-05 14:51:43.092301: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-07-05 14:51:43.366814: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-07-05 14:51:43.987028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1650 computeCapability: 7.5
coreClock: 1.56GHz coreCount: 16 deviceMemorySize: 4.00GiB deviceMemoryBandwidth: 119.24GiB/s
2020-07-05 14:51:44.014364: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2020-07-05 14:51:47.329817: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
2020-07-05 14:51:51.017566: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2020-07-05 14:51:52.647522: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll
2020-07-05 14:51:57.674345: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll
2020-07-05 14:51:59.307558: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll
2020-07-05 14:52:10.936559: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-07-05 14:52:10.950697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2020-07-05 14:52:12.815272: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-05 14:52:12.832900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2020-07-05 14:52:12.839752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
2020-07-05 14:52:12.847622: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/device:GPU:0 with 2917 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1650, pci bus id: 0000:01:00.0, compute capability: 7.5)
True
```

Apart from all the log messages the key things to spot is that the correct version of tensorflow is there: '**2.1.0**' and the `True` printed out in the end signifying that GPU is available and ready to be used. 

```bash
>>> tf.config.list_physical_devices('GPU')
2020-07-05 14:52:31.837948: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1650 computeCapability: 7.5
coreClock: 1.56GHz coreCount: 16 deviceMemorySize: 4.00GiB deviceMemoryBandwidth: 119.24GiB/s
2020-07-05 14:52:31.861074: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2020-07-05 14:52:31.883347: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
2020-07-05 14:52:31.902797: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2020-07-05 14:52:31.914896: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll
2020-07-05 14:52:31.926633: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll
2020-07-05 14:52:31.938539: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll
2020-07-05 14:52:32.014348: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-07-05 14:52:32.029138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

```

Lets see how much memory is allocated to our GPU:

```bash
>>> from tensorflow.python.client import device_lib
>>> print (device_lib.list_local_devices())
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 16190646232416109032
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 3059115622
locality {
  bus_id: 1
  links {
  }
}
incarnation: 2772747070698555022
physical_device_desc: "device: 0, name: GeForce GTX 1650, pci bus id: 0000:01:00.0, compute capability: 7.5"
]
```

The memory limit is 3GB

Great! Now we have GPU enabled in tensorflow and everything should work fine! :)