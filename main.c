#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
    //********* 1. Create a Graph
    TF_Graph* Graph = TF_NewGraph(); // Create a graph
    TF_Status* Status = TF_NewStatus(); // data structure for catching errors

    // 2. Prepare a Session: a session is the frontend of Tensorflow that can be used to perform computation. It will return predictions given some input.
        // 2.1. Create the necessary structures for options
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions(); 
    TF_Buffer* RunOpts = NULL;
        // 2.2 provide information about the model to be loaded.
    const char* saved_model_dir = "model/";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
        //2.3 instantiate the session
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    //****** 3. Define input/output tensor
     // need to use saved_model_cli to read saved_model arch
     // input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_lstm_input"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
	printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0;
    
    //Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;

    //********* 4. Allocate data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    int ndims = 3;
    int64_t dims[] = {1, 10, 1};
    float data[10*1];//= {1,1,1,1,1,1,1,1,1,1};
    for(int i=0; i< (10*1); i++)
    {
        data[i] = 0.5600;
    }
    int ndata = sizeof(float)*10*1;// This is tricky, it number of bytes not number of element


    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
        printf("TF_NewTensor is OK\n");
    }
    else
	printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;
    
    // *******5. Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);

    if(TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    // ***** 6. Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);


    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;
    printf("Result Tensor :\n");
    for(int i=0;i<10;i++)
    {
        printf("%f\n",offsets[i]);
    }
    
    
}
