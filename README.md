# Cross-platform deployment of machine learning models using Tensorflow C API

Created: July 1, 2022 3:44 PM

Ref

[https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn](https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn)

[https://wizardforcel.gitbooks.io/mastering-tf-1x-zh/content/173.html](https://wizardforcel.gitbooks.io/mastering-tf-1x-zh/content/173.html) 精通Tensorflow 

[直接model.save](http://直接model.save) 和 使用saved_model_builder持久化模型的区别？

模型导出： [https://zhuanlan.zhihu.com/p/113734249](https://zhuanlan.zhihu.com/p/113734249)

自定义模型:

```python
# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
```

## 保存和恢复模型

Motivation

可以在训练期间和之后保存模型进度。这意味着模型可以从停止的地方恢复，避免长时间的训练。此外，保存还意味着您可以分享您的模型，其他人可以重现您的工作。在发布研究模型和技术时，大多数机器学习从业者会分享：

- 用于创建模型的代码
- 模型训练的权重 (weight) 和参数 (parameters) 。

共享数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。

### 在训练期间保存模型(以checkpoints形式保存)

see [Document](https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn)

跨平台部署， 为什么不使用checkpoint 进行模型持久化？

`Saver.restore()`需要提前建立好计算图， 这在理论上可行， 但对于模型跨平台来说，成本和效率都存在问题， 当模型趋于复杂， 序列模型， 深度卷积，复杂全连接以及种种超参数以及优化技术都需要两端完全匹配， 目前来看得不偿失。

### 保存整个模型

1. tensorflow SavedModel 格式 `model.save('model', save_format = 'tf')`
2. Keras H5(HDF5) 格式

调用 `[model.save](https://tensorflow.google.cn/api_docs/python/tf/keras/Model?hl=zh-cn#save)` 将保存模型的结构，权重和训练配置保存在单个文件/文件夹中。这可以让您导出模型，以便在不访问原始 Python 代码*的情况下使用它。因为优化器状态（optimizer-state）已经恢复，您可以从中断的位置恢复训练。

整个模型可以保存为两种不同的文件格式（`SavedModel` 和 `HDF5`）。TensorFlow `SavedModel` 格式是 TF2.x 中的默认文件格式。但是，模型能够以 `HDF5` 格式保存。下面详细介绍了如何以两种文件格式保存整个模型。

HDF5 和 SavedModel 之间的主要区别在于，HDF5 使用对象配置来保存模型架构，而 SavedModel 则保存执行计算图。因此，SavedModel 能够在不需要原始代码的情况下保存自定义对象，如子类模型和自定义层。 

# Using the SavedModel format to save and load a model

Tensorflow can store the model in more than one format. If C program needs to load and use the model, it must be saved in `SavedModel` format. (a single “frozen” model file is not enough).

A SavedModel contains a complete TensorFlow program, including trained parameters (i.e, `[tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)`s) and computation.

SavedModel包含一个完整的Tensorflow程序——不仅包含权重值， 还包含计算。它不需要原始模型构建代码就可以运行， 因此， 对共享和部署（e.g. TFLite, Tensorflow Serving, Tensorflow Hub) 非常有用.

## **SavedModel格式**

SavedModel格式是另一种序列化模型的方式

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')
```

### 保存格式

SavedModel 格式是**一个包含 protobuf(.pb) 二进制文件和 TensorFlow 检查点的目录**。检查保存的模型目录：

```
my_model
assets  keras_metadata.pb  saved_modelsave_model.pb variables
```

- `save_model.pb`: 模型架构和训练配置(优化器, 损失和指标).  **用于存储实际Tensorflow程序或模型，以及一组已命名的签名 ——每个签名标识一个接受张量输入和产生张量输出的函数**
- `variables/`: 权重， 包含一个标准训练checkpoint
- `assets`: 目录包含Tensorflow计算图使用的文件(不一定有)
- `keras_metadata.pb`:

---

### **线上服务签名**

tf serving 之类的工具和saved_model_cli 可以与SavedModel交互，为了帮助这些工具要使用的ConcreteFunction， 我们需要服务线上签名。要声明服务上线签名，可使用signatures关键参数指定concreteFunction. 当指定单个签名时， 签名键key为’serving_default’， 并将保存为常量 `tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY`

e.g.  使用`tf.saved_model.load`将SavedModel加载回python

```python
loaded = tf.saved_model.load(mobilenet_save_path)
print(list(loaded.signatures.keys()))  # ["serving_default"]
```

```python
# output
['serving_default']
```

导入的签名总是会返回字典.  (还自定义签名名称和输出字典键)

从SavedModel运行推断

```python
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
```

```python
{'predictions': TensorSpec(shape=(None, 1000), dtype=tf.float32, name='predictions')}
```

从SavedModel 运行推断产生与原始模型相同的结果

```python
labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]

decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]

print("Result after saving and loading:\n", decoded)

#output
Result after saving and loading:
 ['military uniform' 'bow tie' 'suit' 'bearskin' 'pickelhaube']
```

---

### 磁盘上的SavedModel格式

1. What is MetaGraphDef?

Tensorflow的计算图在运行时， 以MetaGraphDef的形式实行计算图， 并且在进行计算图保存时，将MetaGraphDef以二进制的形式写入磁盘

1. saved_model_cli

saved_model_cli 提供了一种通过命令行检查并恢复模型的机制

SavedModel 可能包含模型的多个变体（**多个 `v1.MetaGraphDefs`，通过 `saved_model_cli` 的 `--tag_set` 标记进行标识**），但这种情况很少见。可以为模型创建多个变体的 API 包括 [tf.Estimator.experimental_export_all_saved_models](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator?hl=zh-cn#experimental_export_all_saved_models) 和 TensorFlow 1.x 中的 `tf.saved_model.Builder`。

```
saved_model_cli show --dir {mobilenet_save_path} --tag_set serve

```

```
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"

```

`variables` 目录包含一个标准训练检查点（参阅[训练检查点指南](https://tensorflow.google.cn/guide/checkpoint?hl=zh-cn)）。

```
ls {mobilenet_save_path}/variables

```

```
variables.data-00000-of-00001  variables.index

```

`assets` 目录包含 TensorFlow 计算图使用的文件，例如，用于初始化词汇表的文本文件。本例中没有使用这种文件。

SavedModel 可能有一个用于保存 TensorFlow 计算图未使用的任何文件的 `assets.extra` 目录，例如，为使用者提供的关于如何处理 SavedModel 的信息。TensorFlow 本身并不会使用此目录。

# 在C/C++中加载SavedModel

saved_model_cli 提供了一种通过命令行检查并恢复模型的机制。 使用SavedModel命令行界面(CLI) 可以检查和执行SavedModel。 例如，可以使用CLI来检查模型的SignatureDef. 通过CLI，可以快速确认与模型相符的输入张量的dtype和形状。 此外， 如果要测试模型， 可以通过CLI传入各种格式的样本输入(如，python表达式）， 然后获取输出， 从而执行健全性检查。

Tensorflow has a tool to dive into the saved model files for us to match the input tensor and the output tensor. It is called `saved_model_cli`. It is a command-line tool and comes together when you install Tensorflow.

## SavedModel CLI

SavedModel CLI 支持在SavedModel上使用以下两个命令:

- show: 用于显示SavedModel中可用的计算
- run: 用于从SavedModel运行计算

### 1. show 命令

1. SavedModel 包含一个或多个模型变体（技术为 `[v1.MetaGraphDef](https://tensorflow.google.cn/api_docs/python/tf/compat/v1/MetaGraphDef?hl=zh-cn)`），这些变体通过 tag-set 进行标识。要为模型提供服务，您可能想知道每个模型变体中使用的具体是哪一种 `SignatureDef` ，以及它们的输入和输出是什么。那么，利用 `show` 命令，您就可以按照层级顺序检查 SavedModel 的内容。具体语法如下：
    
    ```
    usage: saved_model_cli show [-h] --dir DIR [--all] [--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
    ```
    
    e.g.  显示SavedModel中的所有可用tag-set
    
    ```python
    $ saved_model_cli show --dir <path_to_saved_model_folder> 
    
    #output
    The given SavedModel contains the following tag-sets
    serve
    ```
    
2. 显示tag-set的所有可用SignatureDef键
    
    SavedModel → 一个或多个模型变体(MetaGraphDef) 
    
    每个MetaGraphDef → tag-set标识
    
    每个MetaGraphDef → 都有不同的SignatureDef
    
    ```python
    `$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve 
    #output
    The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the following keys:
    	SignatureDef key: "classify_x2_to_y3" 
    	SignatureDef key: "classify_x_to_y" 
    	SignatureDef key: "regress_x2_to_y3" 
    	SignatureDef key: "regress_x_to_y" 
    	SignatureDef key: "regress_x_to_y2" 
    	SignatureDef key: "serving_default"`
    ```
    
    如果 tag-set 中有*多个tags*，则必须指定所有标记（标记之间用**逗号**分隔）。例如：
    
    ```
    $ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
    ```
    
    1. 要显示特定SignatureDef的所有输入和输出TensorInfo， 请将SignatureDef键传递给signature_def 选项。 如果想指导输入张量和张量键值，dtype和形状， 以便随后执行计算图， 这很有用
        
        要显示特定 `SignatureDef` 的所有输入和输出 TensorInfo，请将 `SignatureDef` 键传递给 `signature_def` 选项。如果您想知道输入张量的张量键值、dtype 和形状，以便随后执行计算图，这会非常有用。例如：
        
        using `serving_default` signature key into the command to print out the tensor node:
        
        ```bash
        $ saved_model_cli show --dir <path_to_saved_model_folder> --tag_set serve --signature_def serving_default 
        
        #output
        The given SavedModel SignatureDef contains the following input(s):
           inputs['x'] tensor_info:
               dtype: DT_FLOAT
               shape: (-1, 1)
               name: x:0
        The given SavedModel SignatureDef contains the following output(s):
           outputs['y'] tensor_info:
               dtype: DT_FLOAT
               shape: (-1, 1)
               name: y:0
        Method name is: tensorflow/serving/predict
        
        ```
        
        we would need the name `x` `y` later to be used in the C API.
        

### 2. run 命令

## Tensorflow C API

### Motivation

Imagine that we used the beautiful Python tools to design, train and test a Tensorflow model. The model performance is great. Now it is time to use it in an application. What if the target device cannot run Python or takes too much of resources? What if all we have is a very limited environment that can run only compiled executables? Time to write a C program!

### Building C/C++ code

**The Roadmap**

1. Create the computation graph
2. Instantiate/reload a session and associate it with the graph 
    
    The trained model is also loaded here
    
3. Define graph inputs/outputs
    1. Types( e.g. `TF_FLOAT` as floating number)
    2. Shapes
    3. Names of nodes
4. Create input tensor(s) and populate with data
5. Run the session
6. Get data from output tensor(s)

### 1. Write C code

```c
#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}
int main() {

}
```

### 2. Load the savedmodel and the session using `TF_LoadSessionFromSavedModel` API

1. Create a Graph

start with creating a computation graph that can be instantiated as simple as

```c
TF_Graph* Graph = TF_NewGraph();
```

Tensorflow API also provides a handy data structure for catching errors.

```c
TF_Status* Status = TF_NewStatus();
```

1. Prepare a Session

A session is the frontend of Tensorflow that can be used to perform computation. In our case, it will return predictions given some input.

1. create the necessary structures for options
    
    ```c
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;
    ```
    
2. provide information about the model to be loaded
    
    ```c
    const char* saved_model_dir = "<path_to_dir_with_saved_model>";
    const char* tags = "serve"; // default model serving tag;
    in future 
    int ntags = 1;
    ```
    
    This is where we provide full path to an exported Tensorflow model in SavedModel format.
    
3. instantiate the seesion
    
    ```c
    TF_Session* Session = TF_LoadSessionFromSavedModel(
    																									SessionOpts, RunOpts,
    																									saved_model_dir, &tags,
    																									ntags, Graph,
    																									NULL, Status);
    
    ```
    
    Technically, the session has been restored from the SavedModel.
    
    Note that the session object is created but is not ready yet to accept any input.
    

```c
TF_Graph* Graph = TF_NewGraph();
TF_Status* Status = TF_NewStatus();
TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
TF_Buffer* RunOpts = NULL;

const char* saved_model_dir = "<path_to_dir_with_saved_model>";
const char* tags = "serve"; // default model serving tag;

in future 
int ntags = 1;
TF_Session* Session = TF_LoadSessionFromSavedModel(
																									SessionOpts, RunOpts,
																									saved_model_dir, &tags,
																									ntags, Graph,
																									NULL, Status);

```

### 3. Define inputs and outputs

A useful computation graph accepts at least one tensor. A user is responsible to provide complete information about inputs. This information includes node names, data types and shape of the tensor.

SO we grab the tensor node from the graph by their name. Remember earlier we search for a tensor name using `saved_model_cli`. Here where we use it back when we call `TF_GraphOperationByName().` 

1. **Define Inputs.**

Previously:

```c
$ saved_model_cli show --dir <path_to_saved_model_folder> --tag_set serve --signature_def serving_default

#output
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_INT64
      shape: (-1, 1)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

→

```c
// ##### Get input tensor
int NumInputs = 1;
TF_Output* Input = malloc(sizeof(TF_Output) * NumInputs);

```

tell what nodes in the graphs will be accepting the input:

```c

TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1), 0};

if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
```

replace the node name placeholders and indices with actual values. 

Finally, register the inputs:

```c
Input[0] = t0;
```

Note that data hasn’t been still provided yet.

1. **Define outputs.**

In a similar fashion, we tell our program what nodes in the graph will output. Our graph will have one output node.

```c
// ############ Get Output tensor
int NumOutputs = 1;
TF_Output* Output = malloc(sizeof(TF_Output) * NumOutputs);
```

Provide information about the output node

```c
TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
else
        printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
```

finally:

```c
Output[0] = t2;
```

**We haven’t provided yet a pointer that will point to the computation graph result.**

---

Next, we will need to allocate the new tensor locally using `TF_NewTensor`, set the input value and later we will pass to session run. 

Note that `ndata` is total byte size of your data ,not length of the array.

### 4. Provide/Allocate data for inputs & outputs

Create the pointers to the arrays:

```c
TF_Tensor** InputValues = malloc(sizeof(TF_Tensor*) * NumInputs);
TF_Tensor** OutputValues = malloc(sizeof(TF_Tensor*) * NumOutputs);

```

Here we set the input tensor with a value of 20.0 and we should see the output value as 20 as well.

```c
# Create tensors with data here

int ndims = 2;
int64_t dims[] = {1, 1};
int64_t data[] = {20};

int ndata = sizeof(int64_t);
TF_Tensor* int_tensor = TF_NewTensor(TF_INT64, dims, ndims, data, ndata, &NoOpDeallocator, 0); 
```

Assign input tensors with the actual data:

```c
InputValues[0] = int_tensor;
```

### 5. Run the Session

run the computation graph on the provided inputs.

```c
TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

if(TF_GetCode(Status) == TF_OK)
      printf("Session is OK\n");
else
      printf("%s",TF_Message(Status));
```

Free Allocated Memory

```c
TF_DeleteGraph(Graph);
TF_DeleteSession(Session, Status);
TF_DeleteSessionOptions(SessionOpts);
TF_DeleteStatus(Status);
```

Lastly, we get back the output value from the output tensor using `TF_TensorData`
 that extract data from the tensor object. Since we know the size of the output which is 1, I can directly print it. Else use `TF_GraphGetTensorNumDims`
 or other API that is available in `c_api.h`
 or `tf_tensor.h`

```c
void* buff = TF_TensorData(OutputValues[0]);
float* offsets = buff;
printf("Result Tensor :\n");
printf("%f\n",offsets[0]);
return 0;
```

# Compile & Run the code

## **Step A: Compile the code**

Compile it as below:

```
gcc -I<path_of_tensorflow_api>/include/ -L<path_of_tensorflow_api>/lib main.c -ltensorflow -o main.out
```

## **Step B: Run it**

Before you run it. You’ll need to make sure the C library is exported in your environment

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_of_tensorflow_api>/lib
```

RUN IT

```
./main.out
```

You should get an output like below. Notice that the output value is 20, just like our input. you can change the model and initialize the kernel with a weight of value 2 and see if it reflected other value.

```
TF_LoadSessionFromSavedModel OK
TF_GraphOperationByName serving_default_input_1 is OK
TF_GraphOperationByName StatefulPartitionedCall is OK
TF_NewTensor is OK
Session is OK
Result Tensor :
20.000000
```