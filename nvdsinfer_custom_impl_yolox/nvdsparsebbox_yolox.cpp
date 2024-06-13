/**
 * Copyright (c) 2022, Intflow Inc.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distridbution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "edgefarm_config.h"
#include <cmath>
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomYolox  (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *outputLayer = layerFinder("output0");

    if (!outputLayer){
        std::cerr << "ERROR: output layer missing or unsupported data types in output tensors" << std::endl;
        return false;
    }

    const int det_max_instance = outputLayer->inferDims.d[0];
    const int det_elements_per_instance = 34; // x1, y1, x2, y2, rad  class,class,class class kpt27  // 5 + 4 + 27

    if (outputLayer->inferDims.numDims != 2 || outputLayer->inferDims.d[1] != det_elements_per_instance) {
        std::cerr << "ERROR: output layer dimensions are incorrect" << std::endl;
        return false;
    }
    float image_w = MUXER_OUTPUT_WIDTH;    
    float image_h = MUXER_OUTPUT_HEIGHT;
    float scale_x = 1;
    float scale_y = 1;
    float scale = std::min(512 / (image_w*1.0), 288 / (image_h*1.0));
    float *outputData = (float *)outputLayer->buffer;

  for (int indx = 0; indx < det_max_instance; indx++)
  {
        float cx = outputData[indx * det_elements_per_instance + 0] / scale*1.0;
        float cy = outputData[indx * det_elements_per_instance + 1] / scale*1.0;
        float w = outputData[indx * det_elements_per_instance + 2] /  scale*1.0;
        float h = outputData[indx * det_elements_per_instance + 3] /  scale*1.0;
        float rad = outputData[indx * det_elements_per_instance + 4];
        float x1=cx-(w/2.0);
        float y1=cy-(h/2.0);
        float x2=cx+(w/2.0);
        float y2=cy+(h/2.0);
        float max_score = outputData[indx * det_elements_per_instance + 5];
        float max_index = outputData[indx * det_elements_per_instance + 6];


        float kpt_x1 = outputData[indx * det_elements_per_instance + 7] / scale*1.0;
        float kpt_y1 = outputData[indx * det_elements_per_instance + 8] / scale*1.0;
        float kpt_v1 = outputData[indx * det_elements_per_instance + 9];

        float kpt_x2 = outputData[indx * det_elements_per_instance + 10] / scale*1.0;
        float kpt_y2 = outputData[indx * det_elements_per_instance + 11] / scale*1.0;
        float kpt_v2 = outputData[indx * det_elements_per_instance + 12];

        float kpt_x3 = outputData[indx * det_elements_per_instance + 13] / scale*1.0;
        float kpt_y3 = outputData[indx * det_elements_per_instance + 14] / scale*1.0;
        float kpt_v3 = outputData[indx * det_elements_per_instance + 15];

        float kpt_x4 = outputData[indx * det_elements_per_instance + 16] / scale*1.0;
        float kpt_y4 = outputData[indx * det_elements_per_instance + 17] / scale*1.0;
        float kpt_v4 = outputData[indx * det_elements_per_instance + 18];

        float kpt_x5 = outputData[indx * det_elements_per_instance + 19] / scale*1.0;
        float kpt_y5 = outputData[indx * det_elements_per_instance + 20] / scale*1.0;
        float kpt_v5 = outputData[indx * det_elements_per_instance + 21];

        float kpt_x6 = outputData[indx * det_elements_per_instance + 22] / scale*1.0;
        float kpt_y6 = outputData[indx * det_elements_per_instance + 23] / scale*1.0;
        float kpt_v6 = outputData[indx * det_elements_per_instance + 24];

        float kpt_x7 = outputData[indx * det_elements_per_instance + 25] / scale;
        float kpt_y7 = outputData[indx * det_elements_per_instance + 26] / scale;
        float kpt_v7 = outputData[indx * det_elements_per_instance + 27];

        float kpt_x8 = outputData[indx * det_elements_per_instance + 28] / scale;
        float kpt_y8 = outputData[indx * det_elements_per_instance + 29] / scale;
        float kpt_v8 = outputData[indx * det_elements_per_instance + 30];

        float kpt_x9 = outputData[indx * det_elements_per_instance + 31] / scale;
        float kpt_y9 = outputData[indx * det_elements_per_instance + 32] / scale;
        float kpt_v9 = outputData[indx * det_elements_per_instance + 33];


        float threshold = detectionParams.perClassThreshold[max_index];

        if (max_score>threshold){
//		std::cout<<threshold;
            NvDsInferParseObjectInfo object;
            object.left = x1;
            object.top = y1;
            object.width = x2 - x1;
            object.height = y2 - y1;

            object.theta =rad;
            object.classId = max_index;
            object.detectionConfidence = max_score;
            object.landmarksX1  = kpt_x1;
            object.landmarksY1  = kpt_y1;
            object.landmarksX2  = kpt_x2;
            object.landmarksY2  = kpt_y2;
            object.landmarksX3  = kpt_x3;
            object.landmarksY3  = kpt_y3;
            object.landmarksX4  = kpt_x4;
            object.landmarksY4  = kpt_y4;
            object.landmarksX5  = kpt_x5;
            object.landmarksY5  = kpt_y5;
            object.landmarksX6  = kpt_x6;
            object.landmarksY6  = kpt_y6;
            object.landmarksX7  = kpt_x7;
            object.landmarksY7  = kpt_y7;
            object.landmarksX8  = kpt_x8;
            object.landmarksY8  = kpt_y8;
            object.landmarksX9  = kpt_x9;
            object.landmarksY9  = kpt_y9;
            objectList.push_back(object);
            // std::cout<<"x1:"<<x1 << " y1 : "<<y1<<
            // "  max_index:"<<max_index << " max_score : "<<max_score<<
            // "  kpt_x1:"<<kpt_x1 << " kpt_y1 : "<<kpt_y1<<std::endl;
        }



  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolox);
