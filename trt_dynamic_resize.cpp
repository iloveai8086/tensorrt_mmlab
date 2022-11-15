// Copyright (c) OpenMMLab. All rights reserved
#include "trt_dynamic_resize.h"

#include <assert.h>

#include <chrono>

#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"
// 引入CUDA核函数bicubic_interpolate在的头文件，会在enqueue中使用
#include "../bicubic_interpolate/trt_bicubic_interpolate_kernel.hpp"

using namespace nvinfer1;

namespace mmdeploy {
    namespace {
        static const char *PLUGIN_VERSION{"1"};
        static const char *PLUGIN_NAME{"DynamicTRTResize"};//插件名需和ONNX节点名一致，在转换TensorRT模型时被触发
    }  // namespace   不知道为什么mmdeploy要在这边加上这个namespace，官方直接就是static const char

    DynamicTRTResize::DynamicTRTResize(const std::string &name, bool align_corners)
            : TRTPluginBase(name), mAlignCorners(align_corners) {}  // 这个构造函数就是把private的变量初始化了一下，多了一层继承，要name

    // 反序列化，用于将序列化好的权重和参数传入该plugin
    DynamicTRTResize::DynamicTRTResize(const std::string name, const void *data,
                                       size_t length)
            : TRTPluginBase(name) {
        deserialize_value(&data, &length, &mAlignCorners);
    }

    nvinfer1::IPluginV2DynamicExt *DynamicTRTResize::clone() const TRT_NOEXCEPT {
    DynamicTRTResize *plugin =
            new DynamicTRTResize(mLayerName, mAlignCorners);  // 这两参数名是随便写，还是和构造函数声明的一样的？
            // TRTPluginBase(const std::string &name) : mLayerName(name) {} 最开始这么声明的，看来是和声明的一样
    plugin->setPluginNamespace(getPluginNamespace());
    //  try
        //    {
        //        auto* plugin = new PillarScatterPlugin(feature_y_size_, feature_x_size_);
        //        plugin->setPluginNamespace(mNamespace.c_str());
        //        return plugin;
        //    }
        //    catch (std::exception const& e)
        //    {
        //        caughtError(e);
        //    }
        //    return nullptr;   这个是官方的写法
    return plugin;
}

nvinfer1::DimsExprs DynamicTRTResize::getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
        nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
nvinfer1::DimsExprs ret;  // 都是return 这个类型
ret.nbDims = 4;
// 输入张量有两个：input和size_tensor，后者只用于计算输出张量形状  输入torch.Size([1, 3, 256, 256]) torch.Size([1, 1, 512, 512])
// 输出 （1，3，512，512）   就是完全根据torch的shape来计算的，就可以写这个api了
ret.d[0] = inputs[0].d[0];
ret.d[1] = inputs[0].d[1];
ret.d[2] = inputs[1].d[2];
ret.d[3] = inputs[1].d[3];
return ret;
}

bool DynamicTRTResize::supportsFormatCombination(int pos,
                                                 const nvinfer1::PluginTensorDesc *ioDesc,
                                                 int nbInputs, int nbOutputs) TRT_NOEXCEPT {
if (pos == 0) {
return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&  // 这两有什么区别 The type of weights and tensors.
        ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);  // Format of the input/output tensors.  布局格式

} else {
return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
}
}

//判断输入和输出类型数量是否正确。 官方已经默认对了？直接return.....
void DynamicTRTResize::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                       int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc *outputs,
                                       int nbOutputs) TRT_NOEXCEPT {}


// 官方也是直接return 0
size_t DynamicTRTResize::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                          int nbInputs,
                                          const nvinfer1::PluginTensorDesc *outputs,
                                          int nbOutputs) const TRT_NOEXCEPT {
return 0;
}

int DynamicTRTResize::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs, void *workSpace,
                              cudaStream_t stream) TRT_NOEXCEPT {
int batch = inputDesc[0].dims.d[0];
int channels = inputDesc[0].dims.d[1];
int height = inputDesc[0].dims.d[2];
int width = inputDesc[0].dims.d[3];

int height_out = outputDesc[0].dims.d[2];
int width_out = outputDesc[0].dims.d[3];
const void *x = inputs[0];
void *output = outputs[0];

// TODO: add fp16 support
auto data_type = inputDesc[0].type;
switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
        bicubic_interpolate<float>((float *)x, (float *)output, batch, channels, height, width,
        height_out, width_out, mAlignCorners, stream);
        break;
    default:
        return 1;
    break;
}

return 0;
}

// 和官方一样
nvinfer1::DataType DynamicTRTResize::getOutputDataType(int index,
                                                       const nvinfer1::DataType *inputTypes,
                                                       int nbInputs) const TRT_NOEXCEPT {
return inputTypes[0];
}

// IPluginV2 Methods   和官方一样
const char *DynamicTRTResize::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *DynamicTRTResize::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int DynamicTRTResize::getNbOutputs() const TRT_NOEXCEPT { return 1; }

//返回序列化时需要写多少字节到buffer中。
size_t DynamicTRTResize::getSerializationSize() const TRT_NOEXCEPT {
return serialized_size(mAlignCorners);
}

//把需要用的数据按照顺序序列化到buffer里头。
void DynamicTRTResize::serialize(void *buffer) const TRT_NOEXCEPT {
serialize_value(&buffer, mAlignCorners);
}  // 官方自己写了个writeToBuffer<size_t>(d, feature_y_size_);

////////////////////// creator /////////////////////////////

DynamicTRTResizeCreator::DynamicTRTResizeCreator() {
    mPluginAttributes.clear();  // 剩余三个都是固定的
    mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners"));  // 这边好像就是DynamicTRTResize类里面的私有成员变量，实际还要指定type
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

//这两函数都是固定写法
const char *DynamicTRTResizeCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *DynamicTRTResizeCreator::getPluginVersion() const TRT_NOEXCEPT {
return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *DynamicTRTResizeCreator::createPlugin(
        const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
    nvinfer1::Dims size{2, {1, 1}};
    bool align_corners = 1;  // 还是之前DynamicTRTResize类的成员变量

    // 下面的好像也是套路了
    for (int i = 0; i < fc->nbFields; i++) {
        if (fc->fields[i].data == nullptr) {
            continue;
        }
        std::string field_name(fc->fields[i].name);
        //获取align_corners值，用于创建插件DynamicTRTResize的实例
        if (field_name.compare("align_corners") == 0) {
            align_corners = static_cast<const int *>(fc->fields[i].data)[0];
        }
    }
    // 创建插件DynamicTRTResize实例并返回
    DynamicTRTResize *plugin = new DynamicTRTResize(name, align_corners);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::IPluginV2 *DynamicTRTResizeCreator::deserializePlugin(
        const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
    auto plugin = new DynamicTRTResize(name, serialData, serialLength);  // 构造函数，传的是size，而不是align_corners
    plugin->setPluginNamespace(getPluginNamespace());
    // 这个函数就和之前的clone 的函数类似
    return plugin;
}

REGISTER_TENSORRT_PLUGIN(DynamicTRTResizeCreator);//真正注册了该插件，为什么官方的plugin里面有些没有注册的
// 老潘讲了，有两种注册的方式，一个是官方的，一个是这个方式，找一下这个宏还是什么在哪里定义的
}  // namespace mmdeploy