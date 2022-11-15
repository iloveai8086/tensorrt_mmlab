#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cassert>
#include <NvOnnxParser.h>
#include "include/common/logger.h"


// nvinfer1:: createInferBuilder 对应 Python 中的 tensorrt.Builder，需要传入 ILogger 类的实例，
// 但是 ILogger 是一个抽象类，需要用户继承该类并实现内部的虚函数。   杜老就是自己实现的 noexcept override
// 不过此处我们直接使用了 TensorRT 包解压后的 samples 文件夹 ../samples/common/logger.h 文件里的实现 Logger 子类。

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
using namespace sample;
using namespace nvonnxparser;


const char *IN_NAME = "input";
const char *OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int) (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 1

void build_engine_with_trt_api() {
    // Create builder
    Logger m_logger;
    IBuilder *builder = createInferBuilder(m_logger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network
    INetworkDefinition *network = builder->createNetworkV2(EXPLICIT_BATCH);  // 1
    ITensor *input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{BATCH_SIZE, 3, IN_H, IN_W});
    IPoolingLayer *pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{2, 2});
    pool->setStrideNd(DimsHW{2, 2});
    pool->getOutput(0)->setName(OUT_NAME);
    network->markOutput(*pool->getOutput(0));

    // Build engine
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    // IOptimizationProfile 需要用 createOptimizationProfile 函数，对应 Python 的 create_builder_config 函数。
    // 设置 TensorRT 模型的输入尺寸，需要多次调用 IOptimizationProfile 的 setDimensions 方法，比 Python 略繁琐一些
    profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    // Serialize the model to engine file
    IHostMemory *modelStream{nullptr};
    assert(engine != nullptr);
    modelStream = engine->serialize();

    std::ofstream p("model.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    std::cout << "generating file done!" << std::endl;

    // Release resources
    modelStream->destroy();
    network->destroy();
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void build_engine_with_onnx() {
    // Create builder
    Logger m_logger;
    IBuilder *builder = createInferBuilder(m_logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    // Parse ONNX file
    IParser *parser = nvonnxparser::createParser(*network, m_logger);
    bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    // Get the name of network input
    Dims dim = network->getInput(0)->getDimensions();
    if (dim.d[0] == -1)  // -1 means it is a dynamic model
    {
        const char *name = network->getInput(0)->getName();
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        config->addOptimizationProfile(profile);
    }

    // Build engine
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    // Serialize the model to engine file
    IHostMemory *modelStream{nullptr};
    assert(engine != nullptr);
    modelStream = engine->serialize();

    std::ofstream p("model2.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    std::cout << "generate file success!" << std::endl;

    // Release resources
    modelStream->destroy();
    network->destroy();
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void build_engine_with_onnx2() {
    // Create builder
    Logger m_logger;
    IBuilder *builder = createInferBuilder(m_logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    // Parse ONNX file
    IParser *parser = nvonnxparser::createParser(*network, m_logger);
    bool parser_status = parser->parseFromFile("model_gat.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    // Get the name of network input
    Dims dim = network->getInput(0)->getDimensions();
    Dims dim2 = network->getInput(1)->getDimensions();
    if (dim.d[0] == -1)  // -1 means it is a dynamic model
    {
        const char *name = network->getInput(0)->getName();
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        config->addOptimizationProfile(profile);
    }

    // Build engine
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    // Serialize the model to engine file
    IHostMemory *modelStream{nullptr};
    assert(engine != nullptr);
    modelStream = engine->serialize();

    std::ofstream p("model2.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    std::cout << "generate file success!" << std::endl;

    // Release resources
    modelStream->destroy();
    network->destroy();
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    // context 两块内存，batch，内存是host上面的内存
    const ICudaEngine &engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);  //  bindings是tensorRT对输入输出张量的描述，bindings = input-tensor + output-tensor。
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(IN_NAME);
    const int outputIndex = engine.getBindingIndex(OUT_NAME);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float)));
    // 上面的第一个参数其实就是void** 那种的，杜老是初始化为空指针
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));  // 有一个流，异步加了stream 后面的操作都带这个
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host  之前分配的显存就是结果
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);   // 同步，好像也是套路，stream完全结束后，推理也结束了，内存拷贝也结束了

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));  // 显存需要手动释放 逆序释放的 杜老是写一起了，他这边是函数结束了自动释放

}

void doInference2(IExecutionContext &context, float *input1, float *input2, float *output, int batchSize) {
    // context 两块内存，batch，内存是host上面的内存
    const ICudaEngine &engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);  //  bindings是tensorRT对输入输出张量的描述，bindings = input-tensor + output-tensor。
    float *buffers[3];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = engine.getBindingIndex("one");
    const int inputIndex2 = engine.getBindingIndex("two");
    const int outputIndex = engine.getBindingIndex("prob");
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * 10 * 2 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[inputIndex2], batchSize * 128 * 10 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 64 * 10 * sizeof(float)));
    // 上面的第一个参数其实就是void** 那种的，杜老是初始化为空指针
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));  // 有一个流，异步加了stream 后面的操作都带这个
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host  之前分配的显存就是结果
    CHECK(cudaMemcpyAsync(buffers[inputIndex1], input1, batchSize * 10 * 2 * sizeof(float), cudaMemcpyHostToDevice,
                          stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, batchSize * 128 * 10 * sizeof(float), cudaMemcpyHostToDevice,
                          stream));
    float *bindings[] = {buffers[inputIndex1], buffers[inputIndex2], buffers[outputIndex]};
    context.enqueueV2((void **) bindings, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 64 * 10 * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);   // 同步，好像也是套路，stream完全结束后，推理也结束了，内存拷贝也结束了

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[inputIndex2]));
    CHECK(cudaFree(buffers[outputIndex]));  // 显存需要手动释放 逆序释放的 杜老是写一起了，他这边是函数结束了自动释放

}

void test_gcn_infer(){
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("model2.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    Logger m_logger;
    IRuntime *runtime = createInferRuntime(m_logger);  // 杜老是全部用这种给它封装了
    // auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // generate input data    连续的内存   batchSize * 10 * 2  batchSize * 128 * 10  batchSize  * 64 * 10
    float data1[BATCH_SIZE * 10 * 2];
    for (int i = 0; i < BATCH_SIZE * 10 * 2; i++)
        data1[i] = 1;
    float data2[BATCH_SIZE * 128 * 10];
    for (int i = 0; i < BATCH_SIZE * 128 * 10; i++)
        data2[i] = 1;  // 之前这边写错了，写成1，一直seg fault，为什么呢？

    // Run inference  输出 由于224*224->112*112 所以内存占用除以4
    float prob[BATCH_SIZE * 64 * 10];
    doInference2(*context, data1, data2, prob, BATCH_SIZE);
    for (int i = 0; i < 40; ++i) {
        std::cout << prob[i] << std::endl;
    }

    // generate input data
//    float data[BATCH_SIZE * 3 * IN_H * IN_W];
//    for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++)
//        data[i] = 1;
//
//    // Run inference
//    float prob[BATCH_SIZE * 3 * IN_H * IN_W / 4];
//    doInference(*context, data, prob, BATCH_SIZE);
//    for (int i = 0; i < 10; ++i) {
//        std::cout << prob[i] << std::endl;
//    }
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "done !" << std::endl;
}

int main(int argc, char **argv) {
    // build_engine_with_trt_api();
    // build_engine_with_onnx();
    // build_engine_with_onnx2();

    // test_gcn_infer();



    return 0;
}


// LD_LIBRARY_PATH /media/ros/A666B94D66B91F4D/ros/new_deploy/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib:/media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib:$LD_LIBRARY_PATH

