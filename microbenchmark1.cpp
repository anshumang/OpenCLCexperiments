#include <mpi.h>
#include <CL/cl.h>
#include <stdio.h>

const char *KernelSource = "\n" \
"typedef struct mystruct2{ \n" \
"int *a2;\n" \
"} mystruct2_t;\n" \
"typedef struct mystruct{ \n" \
"int *a;\n" \
"mystruct2_t *t2;\n" \
"} mystruct_t;\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                            \n" \
"   __global mystruct_t* t)\n" \ 
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int *a, *a2;\n" \
"   mystruct2_t *t2;\n" \
"   a = t->a;\n" \
"   t2 = t->t2;\n" \
"   a2 = t2->a2;\n" \
"       output[i] = input[i];                                \n" \
"}                                                                      \n" \
"\n";


int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    cl_int err;
    cl_platform_id platform;
    err  = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS)
      fprintf(stderr, "clGetPlatformIDs failed with err %d\n", err);
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err != CL_SUCCESS)
      fprintf(stderr, "clGetDeviceIDs failed with err %d\n", err);
   cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS)
      fprintf(stderr, "clCreateContext failed with err %d\n", err);


   cl_program program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if(err != CL_SUCCESS)
      fprintf(stderr, "clCreateProgramWithSource failed with err %d\n", err);
   err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
      fprintf(stderr, "clBuildProgram failed with err %d (CL_BUILD_PROGRAM_FAILURE %d) \n", err, CL_BUILD_PROGRAM_FAILURE);
   char param_value[1000];
   for(int i=0; i<1000; i++)
      param_value[i] = '0';
   size_t param_value_size_ret;
   err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1000, param_value, &param_value_size_ret);
   if(err != CL_SUCCESS)
     fprintf(stderr, "clGetProgramBuildInfo failed with err %d\n", err);
   fprintf(stderr, "build log of %d bytes\n", param_value_size_ret);
   for(int i=0; i<1000; i++)
     fprintf(stderr, "%c", param_value[i]);
   fprintf(stderr, "\n");
   return 0;
}
