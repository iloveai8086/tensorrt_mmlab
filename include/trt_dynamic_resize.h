//
// Created by ros on 11/13/22.
//

#ifndef TENSORRT_MMLAB_TRT_DYNAMIC_RESIZE_H
#define TENSORRT_MMLAB_TRT_DYNAMIC_RESIZE_H

#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>
#include "trt_plugin_base.hpp"

namespace mmdeploy {
    class DynamicTRTResize : public TRTPluginBase {  // 继承自nvinfer1::IPluginV2DynamicExt
    public:
        /*
         * 在插件类 DynamicTRTResize 中，我们定义了私有变量 mAlignCorners，该变量表示是否 align corners。
         * 此外只要实现构造函数、析构函数和 TensoRT 中三个基类的方法即可。其中构造函数有二，分别用于创建插件和反序列化插件
         * 而基类方法中：
         * 基类 IPluginV2DynamicExt 的方法较为值得关注
         * getOutputDimensions 获取输出张量的形状，enqueue 真正负责执行我们的算法，内部一般会调用 CUDA 核函数。
         * 本文实现的插件直接调用 MMDeploy 已定义在 csrc/backend_ops/tensorrt/bicubic_interpolate 的核函数 bicubic_interpolate。
         * 基类 IPluginV2Ext 的方法，我们只要实现获取输出数据类型的 getOutputDataType 即可。
         * 基类 IPluginV2 则是些获取插件类型和版本号的方法
         * 此外则是序列化输入插件的参数的函数 serialize 和计算该参数的序列化后 buffer 大小的函数 getSerializationSize
         * 以及获取输出张量个数的方法 getNbOutputs。还有部分公共方法被定义在 TRTPluginBase 类内了。
         *
         * */
        DynamicTRTResize(const std::string &name, bool align_corners);//parse + clone 阶段
        DynamicTRTResize(const std::string name, const void *data, size_t length);//用于在deserialize阶段
        DynamicTRTResize() = delete;// 注意需要把默认构造函数删掉：    为什么要删掉？


        // IPluginV2DynamicExt Methods
        nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;

        //// clone成员函数主要用于传递不变的权重和参数，将plugin复制n多份，从而可以被不同engine或者builder或者network使用。
        nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
        TRT_NOEXCEPT override;//我们要做的就是在这个成员函数中根据输入维度推理出模型的输出维度
        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                       int nbOutputs) TRT_NOEXCEPT override;

        //TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                             const nvinfer1::DynamicPluginTensorDesc *out,
                             int nbOutputs) TRT_NOEXCEPT override;//配置这个插件op，判断输入和输出类型数量是否正确。
        //这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，运行时候自动获取显存空间
        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                const nvinfer1::PluginTensorDesc *outputs,
                                int nbOutputs) const TRT_NOEXCEPT override;

        int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;
        // 实际插件op的执行函数，我们自己实现的cuda操作就放到这里(当然C++写的op也可以放进来，不过因为是CPU执行，速度就比较慢了)
        // 默认写的.cu是fp32的，TensorRT在fp16运行模式下，运行到不支持fp16的插件op时，会自动切换到fp32模式，等插件op运行完再切换回来。

        // -------------------------------------------------------------------------------------------------------------
        // IPluginV2Ext Methods  返回结果的类型，一般来说我们插件op返回结果类型与输入类型一致：
        nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                             int nbInputs) const TRT_NOEXCEPT override;

        // IPluginV2 Methods
        const char *getPluginType() const TRT_NOEXCEPT override;

        const char *getPluginVersion() const TRT_NOEXCEPT override;

        int
        getNbOutputs() const TRT_NOEXCEPT override;  // 插件op返回多少个Tensor，比如MyCustomPlugin这个操作只输出一个Tensor(也就是一个output)，所以直接return 1
        size_t getSerializationSize() const TRT_NOEXCEPT override;  // 返回序列化时需要写多少字节到buffer中。
        void serialize(void *buffer) const TRT_NOEXCEPT override;  // 把需要用的数据按照顺序序列化到buffer里头。
    private:
        // 这边一般会用到自己的一些参数，比如pillar scatter 就是有一些grid size
        bool mAlignCorners;  // 我们定义了私有变量 mAlignCorners，该变量表示是否 align corners。
        // 此外只要实现构造函数、析构函数和 TensoRT 中三个基类的方法即可。其中构造函数有二，分别用于创建插件和反序列化插件。而基类方法中：
    };

    /*
     * 在插件工厂类 DynamicTRTResizeCreator 中，我们需要声明获取插件名称和版本的方法 getPluginName 和 getPluginVersion。
     * 同时我们还需要声明创建插件和反序列化插件的方法 createPlugin 和 deserializePlugin
     * 前者调用 DynamicTRTResize 中创建插件的方法，后者调用反序列化插件的方法。
     * */
    class DynamicTRTResizeCreator : public TRTPluginCreatorBase {  // 继承自public nvinfer1::IPluginCreator
    public:
        DynamicTRTResizeCreator();

        const char *getPluginName() const TRT_NOEXCEPT override;  // 具体看实现
        const char *getPluginVersion() const TRT_NOEXCEPT override;  // 具体看实现
        nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
        TRT_NOEXCEPT override;

        //通过PluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来，然后调用上文提到的第一个构造函数：.....
        nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                               size_t serialLength) TRT_NOEXCEPT override;
        // 这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中。
    };
}  // namespace mmdeploy

#endif //TENSORRT_MMLAB_TRT_DYNAMIC_RESIZE_H
